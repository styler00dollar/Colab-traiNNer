"""
29-Nov-21
https://github.com/tding1/CDFI/blob/f048b38e0cfbb4644dc0ac7ceb38288b145d22d8/models/compressed_adacof.py
https://github.com/tding1/CDFI/blob/f048b38e0cfbb4644dc0ac7ceb38288b145d22d8/cupy_module/adacof.py
https://github.com/tding1/CDFI/blob/f048b38e0cfbb4644dc0ac7ceb38288b145d22d8/utility.py
"""

import torch

# import cupy_module.adacof as adacof
import sys
from torch.nn import functional as F

# from utility import CharbonnierFunc, moduleNormalize


def CharbonnierFunc(data, epsilon=0.001):
    return torch.mean(torch.sqrt(data**2 + epsilon**2))


def moduleNormalize(frame):
    return torch.cat(
        [
            (frame[:, 0:1, :, :] - 0.4631),
            (frame[:, 1:2, :, :] - 0.4352),
            (frame[:, 2:3, :, :] - 0.3990),
        ],
        1,
    )


import cupy
import re
import math

kernel_AdaCoF_updateOutput = """
    extern "C" __global__ void kernel_AdaCoF_updateOutput(
        const int n,
        const float* input,
        const float* weight,
        const float* offset_i,
        const float* offset_j,
        float* output
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        float dblOutput = 0.0;

        const int intSample = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output) ) % SIZE_0(output);
        const int c         = ( intIndex / SIZE_3(output) / SIZE_2(output)                  ) % SIZE_1(output);
        const int i         = ( intIndex / SIZE_3(output)                                   ) % SIZE_2(output);
        const int j         = ( intIndex                                                    ) % SIZE_3(output);

        for (int k = 0; k < F_SIZE; k += 1) {
        for (int l = 0; l < F_SIZE; l += 1) {
        float w         = VALUE_4(weight, intSample, k*F_SIZE+l, i, j);
        float alpha     = VALUE_4(offset_i, intSample, k*F_SIZE+l, i, j);
        float beta      = VALUE_4(offset_j, intSample, k*F_SIZE+l, i, j);
        int A           = (int) alpha;
        int B           = (int) beta;

        int i_k_A = i+k*DILATION+A;
        if(i_k_A < 0)
            i_k_A = 0;
        if(i_k_A > SIZE_2(input) - 1)
            i_k_A = SIZE_2(input) - 1;

        int j_l_B = j+l*DILATION+B;
        if(j_l_B < 0)
            j_l_B = 0;
        if(j_l_B > SIZE_3(input) - 1)
            j_l_B = SIZE_3(input) - 1;

        int i_k_A_1 = i+k*DILATION+A+1;
        if(i_k_A_1 < 0)
            i_k_A_1 = 0;
        if(i_k_A_1 > SIZE_2(input) - 1)
            i_k_A_1 = SIZE_2(input) - 1;

        int j_l_B_1 = j+l*DILATION+B+1;
        if(j_l_B_1 < 0)
            j_l_B_1 = 0;
        if(j_l_B_1 > SIZE_3(input) - 1)
            j_l_B_1 = SIZE_3(input) - 1;

        dblOutput += w * (
            VALUE_4(input, intSample, c, i_k_A, j_l_B)*(1-(alpha-(float)A))*(1-(beta-(float)B)) + 
            VALUE_4(input, intSample, c, i_k_A_1, j_l_B)*(alpha-(float)A)*(1-(beta-(float)B)) + 
            VALUE_4(input, intSample, c, i_k_A, j_l_B_1)*(1-(alpha-(float)A))*(beta-(float)B) + 
            VALUE_4(input, intSample, c, i_k_A_1, j_l_B_1)*(alpha-(float)A)*(beta-(float)B)
            );
        }
        }

        output[intIndex] = dblOutput;
    } }
"""

kernel_AdaCoF_updateGradWeight = """
    extern "C" __global__ void kernel_AdaCoF_updateGradWeight(
        const int n,
        const float* gradLoss,
        const float* input,
        const float* offset_i,
        const float* offset_j,
        float* gradWeight
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        float floatOutput = 0.0;

        const int intSample  = ( intIndex / SIZE_3(gradWeight) / SIZE_2(gradWeight) / SIZE_1(gradWeight) ) % SIZE_0(gradWeight);
        const int intDepth   = ( intIndex / SIZE_3(gradWeight) / SIZE_2(gradWeight)                      ) % SIZE_1(gradWeight);
        const int i          = ( intIndex / SIZE_3(gradWeight)                                           ) % SIZE_2(gradWeight);
        const int j          = ( intIndex                                                                ) % SIZE_3(gradWeight);

        int k = intDepth / F_SIZE;
        int l = intDepth % F_SIZE;

        for (int c = 0; c < 3; c++) 
        {
        float delta     = VALUE_4(gradLoss, intSample, c, i, j);
        float alpha     = VALUE_4(offset_i, intSample, k*F_SIZE+l, i, j);
        float beta      = VALUE_4(offset_j, intSample, k*F_SIZE+l, i, j);
        int A           = (int) alpha;
        int B           = (int) beta;

        int i_k_A = i+k*DILATION+A;
        if(i_k_A < 0)
            i_k_A = 0;
        if(i_k_A > SIZE_2(input) - 1)
            i_k_A = SIZE_2(input) - 1;

        int j_l_B = j+l*DILATION+B;
        if(j_l_B < 0)
            j_l_B = 0;
        if(j_l_B > SIZE_3(input) - 1)
            j_l_B = SIZE_3(input) - 1;

        int i_k_A_1 = i+k*DILATION+A+1;
        if(i_k_A_1 < 0)
            i_k_A_1 = 0;
        if(i_k_A_1 > SIZE_2(input) - 1)
            i_k_A_1 = SIZE_2(input) - 1;

        int j_l_B_1 = j+l*DILATION+B+1;
        if(j_l_B_1 < 0)
            j_l_B_1 = 0;
        if(j_l_B_1 > SIZE_3(input) - 1)
            j_l_B_1 = SIZE_3(input) - 1;
        
        floatOutput += delta * (
            VALUE_4(input, intSample, c, i_k_A, j_l_B)*(1-(alpha-(float)A))*(1-(beta-(float)B)) + 
            VALUE_4(input, intSample, c, i_k_A_1, j_l_B)*(alpha-(float)A)*(1-(beta-(float)B)) + 
            VALUE_4(input, intSample, c, i_k_A, j_l_B_1)*(1-(alpha-(float)A))*(beta-(float)B) + 
            VALUE_4(input, intSample, c, i_k_A_1, j_l_B_1)*(alpha-(float)A)*(beta-(float)B)
            );
        }

        gradWeight[intIndex] = floatOutput;
    } }
"""

kernel_AdaCoF_updateGradAlpha = """
    extern "C" __global__ void kernel_AdaCoF_updateGradAlpha(
        const int n,
        const float* gradLoss,
        const float* input,
        const float* weight,
        const float* offset_i,
        const float* offset_j,
        float* gradOffset_i
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        float floatOutput = 0.0;

        const int intSample  = ( intIndex / SIZE_3(gradOffset_i) / SIZE_2(gradOffset_i) / SIZE_1(gradOffset_i) ) % SIZE_0(gradOffset_i);
        const int intDepth   = ( intIndex / SIZE_3(gradOffset_i) / SIZE_2(gradOffset_i)                        ) % SIZE_1(gradOffset_i);
        const int i          = ( intIndex / SIZE_3(gradOffset_i)                                               ) % SIZE_2(gradOffset_i);
        const int j          = ( intIndex                                                                      ) % SIZE_3(gradOffset_i);

        int k = intDepth / F_SIZE;
        int l = intDepth % F_SIZE;

        for (int c = 0; c < 3; c++) 
        {
        float delta     = VALUE_4(gradLoss, intSample, c, i, j);
        float w         = VALUE_4(weight, intSample, k*F_SIZE+l, i, j);
        float alpha     = VALUE_4(offset_i, intSample, k*F_SIZE+l, i, j);
        float beta      = VALUE_4(offset_j, intSample, k*F_SIZE+l, i, j);
        int A           = (int) alpha;
        int B           = (int) beta;

        int i_k_A = i+k*DILATION+A;
        if(i_k_A < 0)
            i_k_A = 0;
        if(i_k_A > SIZE_2(input) - 1)
            i_k_A = SIZE_2(input) - 1;

        int j_l_B = j+l*DILATION+B;
        if(j_l_B < 0)
            j_l_B = 0;
        if(j_l_B > SIZE_3(input) - 1)
            j_l_B = SIZE_3(input) - 1;

        int i_k_A_1 = i+k*DILATION+A+1;
        if(i_k_A_1 < 0)
            i_k_A_1 = 0;
        if(i_k_A_1 > SIZE_2(input) - 1)
            i_k_A_1 = SIZE_2(input) - 1;

        int j_l_B_1 = j+l*DILATION+B+1;
        if(j_l_B_1 < 0)
            j_l_B_1 = 0;
        if(j_l_B_1 > SIZE_3(input) - 1)
            j_l_B_1 = SIZE_3(input) - 1;

        floatOutput += delta * w * (
            - VALUE_4(input, intSample, c, i_k_A, j_l_B)*(1-(beta-(float)B)) + 
            VALUE_4(input, intSample, c, i_k_A_1, j_l_B)*(1-(beta-(float)B)) - 
            VALUE_4(input, intSample, c, i_k_A, j_l_B_1)*(beta-(float)B) + 
            VALUE_4(input, intSample, c, i_k_A_1, j_l_B_1)*(beta-(float)B)
            );
        }

        gradOffset_i[intIndex] = floatOutput;
    } }
"""

kernel_AdaCoF_updateGradBeta = """
    extern "C" __global__ void kernel_AdaCoF_updateGradBeta(
        const int n,
        const float* gradLoss,
        const float* input,
        const float* weight,
        const float* offset_i,
        const float* offset_j,
        float* gradOffset_j
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        float floatOutput = 0.0;

        const int intSample  = ( intIndex / SIZE_3(gradOffset_j) / SIZE_2(gradOffset_j) / SIZE_1(gradOffset_j) ) % SIZE_0(gradOffset_j);
        const int intDepth   = ( intIndex / SIZE_3(gradOffset_j) / SIZE_2(gradOffset_j)                        ) % SIZE_1(gradOffset_j);
        const int i          = ( intIndex / SIZE_3(gradOffset_j)                                               ) % SIZE_2(gradOffset_j);
        const int j          = ( intIndex                                                                      ) % SIZE_3(gradOffset_j);

        int k = intDepth / F_SIZE;
        int l = intDepth % F_SIZE;

        for (int c = 0; c < 3; c++) 
        {
        float delta     = VALUE_4(gradLoss, intSample, c, i, j);
        float w         = VALUE_4(weight, intSample, k*F_SIZE+l, i, j);
        float alpha     = VALUE_4(offset_i, intSample, k*F_SIZE+l, i, j);
        float beta      = VALUE_4(offset_j, intSample, k*F_SIZE+l, i, j);
        int A           = (int) alpha;
        int B           = (int) beta;

        int i_k_A = i+k*DILATION+A;
        if(i_k_A < 0)
            i_k_A = 0;
        if(i_k_A > SIZE_2(input) - 1)
            i_k_A = SIZE_2(input) - 1;

        int j_l_B = j+l*DILATION+B;
        if(j_l_B < 0)
            j_l_B = 0;
        if(j_l_B > SIZE_3(input) - 1)
            j_l_B = SIZE_3(input) - 1;

        int i_k_A_1 = i+k*DILATION+A+1;
        if(i_k_A_1 < 0)
            i_k_A_1 = 0;
        if(i_k_A_1 > SIZE_2(input) - 1)
            i_k_A_1 = SIZE_2(input) - 1;

        int j_l_B_1 = j+l*DILATION+B+1;
        if(j_l_B_1 < 0)
            j_l_B_1 = 0;
        if(j_l_B_1 > SIZE_3(input) - 1)
            j_l_B_1 = SIZE_3(input) - 1;

        floatOutput += delta * w * (
            - VALUE_4(input, intSample, c, i_k_A, j_l_B)*(1-(alpha-(float)A)) - 
            VALUE_4(input, intSample, c, i_k_A_1, j_l_B)*(alpha-(float)A) + 
            VALUE_4(input, intSample, c, i_k_A, j_l_B_1)*(1-(alpha-(float)A)) + 
            VALUE_4(input, intSample, c, i_k_A_1, j_l_B_1)*(alpha-(float)A)
            );
        }

        gradOffset_j[intIndex] = floatOutput;
    } }
"""


def cupy_kernel(strFunction, intFilterSize, intDilation, objectVariables):
    strKernel = globals()[strFunction]

    while True:
        objectMatch = re.search("(SIZE_)([0-4])(\()([^\)]*)(\))", strKernel)

        if objectMatch is None:
            break
        # end

        intArg = int(objectMatch.group(2))

        strTensor = objectMatch.group(4)
        intSizes = objectVariables[strTensor].size()

        strKernel = strKernel.replace(objectMatch.group(), str(intSizes[intArg]))
    # end

    while True:
        objectMatch = re.search("(VALUE_)([0-4])(\()([^\)]+)(\))", strKernel)

        if objectMatch is None:
            break
        # end

        intArgs = int(objectMatch.group(2))
        strArgs = objectMatch.group(4).split(",")

        strTensor = strArgs[0]
        intStrides = objectVariables[strTensor].stride()
        strIndex = [
            "(("
            + strArgs[intArg + 1].replace("{", "(").replace("}", ")").strip()
            + ")*"
            + str(intStrides[intArg])
            + ")"
            for intArg in range(intArgs)
        ]

        strKernel = strKernel.replace(
            objectMatch.group(0), strTensor + "[" + str.join("+", strIndex) + "]"
        )
    # end

    strKernel = strKernel.replace("F_SIZE", str(intFilterSize))
    strKernel = strKernel.replace("DILATION", str(intDilation))

    return strKernel


# end


@cupy.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)


# end


class FunctionAdaCoF(torch.autograd.Function):
    # end
    @staticmethod
    def forward(ctx, input, weight, offset_i, offset_j, dilation):
        ctx.save_for_backward(input, weight, offset_i, offset_j)
        ctx.dilation = dilation

        intSample = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intFilterSize = int(math.sqrt(weight.size(1)))
        intOutputHeight = weight.size(2)
        intOutputWidth = weight.size(3)

        assert (
            intInputHeight - ((intFilterSize - 1) * dilation + 1) == intOutputHeight - 1
        )
        assert (
            intInputWidth - ((intFilterSize - 1) * dilation + 1) == intOutputWidth - 1
        )

        assert input.is_contiguous() is True
        assert weight.is_contiguous() is True
        assert offset_i.is_contiguous() is True
        assert offset_j.is_contiguous() is True

        output = input.new_zeros(
            intSample, intInputDepth, intOutputHeight, intOutputWidth
        )

        if input.is_cuda is True:

            class Stream:
                ptr = torch.cuda.current_stream().cuda_stream

            # end

            n = output.nelement()
            cupy_launch(
                "kernel_AdaCoF_updateOutput",
                cupy_kernel(
                    "kernel_AdaCoF_updateOutput",
                    intFilterSize,
                    dilation,
                    {
                        "input": input,
                        "weight": weight,
                        "offset_i": offset_i,
                        "offset_j": offset_j,
                        "output": output,
                    },
                ),
            )(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    n,
                    input.data_ptr(),
                    weight.data_ptr(),
                    offset_i.data_ptr(),
                    offset_j.data_ptr(),
                    output.data_ptr(),
                ],
                stream=Stream,
            )

        elif input.is_cuda is False:
            raise NotImplementedError()

        # end

        return output

    # end
    @staticmethod
    def backward(ctx, gradOutput):
        input, weight, offset_i, offset_j = ctx.saved_tensors
        dilation = ctx.dilation

        intSample = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intFilterSize = int(math.sqrt(weight.size(1)))
        intOutputHeight = weight.size(2)
        intOutputWidth = weight.size(3)

        assert (
            intInputHeight - ((intFilterSize - 1) * dilation + 1) == intOutputHeight - 1
        )
        assert (
            intInputWidth - ((intFilterSize - 1) * dilation + 1) == intOutputWidth - 1
        )

        assert gradOutput.is_contiguous() is True

        gradInput = (
            input.new_zeros(intSample, intInputDepth, intInputHeight, intInputWidth)
            if ctx.needs_input_grad[0] is True
            else None
        )
        gradWeight = (
            input.new_zeros(
                intSample, intFilterSize**2, intOutputHeight, intOutputWidth
            )
            if ctx.needs_input_grad[1] is True
            else None
        )
        gradOffset_i = (
            input.new_zeros(
                intSample, intFilterSize**2, intOutputHeight, intOutputWidth
            )
            if ctx.needs_input_grad[2] is True
            else None
        )
        gradOffset_j = (
            input.new_zeros(
                intSample, intFilterSize**2, intOutputHeight, intOutputWidth
            )
            if ctx.needs_input_grad[2] is True
            else None
        )

        if input.is_cuda is True:

            class Stream:
                ptr = torch.cuda.current_stream().cuda_stream

            # end

            # weight grad
            n_w = gradWeight.nelement()
            cupy_launch(
                "kernel_AdaCoF_updateGradWeight",
                cupy_kernel(
                    "kernel_AdaCoF_updateGradWeight",
                    intFilterSize,
                    dilation,
                    {
                        "gradLoss": gradOutput,
                        "input": input,
                        "offset_i": offset_i,
                        "offset_j": offset_j,
                        "gradWeight": gradWeight,
                    },
                ),
            )(
                grid=tuple([int((n_w + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    n_w,
                    gradOutput.data_ptr(),
                    input.data_ptr(),
                    offset_i.data_ptr(),
                    offset_j.data_ptr(),
                    gradWeight.data_ptr(),
                ],
                stream=Stream,
            )

            # alpha grad
            n_i = gradOffset_i.nelement()
            cupy_launch(
                "kernel_AdaCoF_updateGradAlpha",
                cupy_kernel(
                    "kernel_AdaCoF_updateGradAlpha",
                    intFilterSize,
                    dilation,
                    {
                        "gradLoss": gradOutput,
                        "input": input,
                        "weight": weight,
                        "offset_i": offset_i,
                        "offset_j": offset_j,
                        "gradOffset_i": gradOffset_i,
                    },
                ),
            )(
                grid=tuple([int((n_i + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    n_i,
                    gradOutput.data_ptr(),
                    input.data_ptr(),
                    weight.data_ptr(),
                    offset_i.data_ptr(),
                    offset_j.data_ptr(),
                    gradOffset_i.data_ptr(),
                ],
                stream=Stream,
            )

            # beta grad
            n_j = gradOffset_j.nelement()
            cupy_launch(
                "kernel_AdaCoF_updateGradBeta",
                cupy_kernel(
                    "kernel_AdaCoF_updateGradBeta",
                    intFilterSize,
                    dilation,
                    {
                        "gradLoss": gradOutput,
                        "input": input,
                        "weight": weight,
                        "offset_i": offset_i,
                        "offset_j": offset_j,
                        "gradOffset_j": gradOffset_j,
                    },
                ),
            )(
                grid=tuple([int((n_j + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    n_j,
                    gradOutput.data_ptr(),
                    input.data_ptr(),
                    weight.data_ptr(),
                    offset_i.data_ptr(),
                    offset_j.data_ptr(),
                    gradOffset_j.data_ptr(),
                ],
                stream=Stream,
            )

        elif input.is_cuda is False:
            raise NotImplementedError()

        # end

        return gradInput, gradWeight, gradOffset_i, gradOffset_j, None


# end


# end


class AdaCoFNet(torch.nn.Module):
    def __init__(self):
        super(AdaCoFNet, self).__init__()
        self.kernel_size = 5
        self.dilation = 1

        self.kernel_pad = int(((self.kernel_size - 1) * self.dilation) / 2.0)

        self.get_kernel = PrunedKernelEstimation(self.kernel_size)

        self.modulePad = torch.nn.ReplicationPad2d(
            [self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad]
        )

        self.moduleAdaCoF = FunctionAdaCoF.apply

    def forward(self, frame0, frame2):
        h0 = int(list(frame0.size())[2])
        w0 = int(list(frame0.size())[3])
        h2 = int(list(frame2.size())[2])
        w2 = int(list(frame2.size())[3])
        if h0 != h2 or w0 != w2:
            sys.exit("Frame sizes do not match")

        h_padded = False
        w_padded = False
        if h0 % 32 != 0:
            pad_h = 32 - (h0 % 32)
            frame0 = F.pad(frame0, [0, 0, 0, pad_h], mode="reflect")
            frame2 = F.pad(frame2, [0, 0, 0, pad_h], mode="reflect")
            h_padded = True

        if w0 % 32 != 0:
            pad_w = 32 - (w0 % 32)
            frame0 = F.pad(frame0, [0, pad_w, 0, 0], mode="reflect")
            frame2 = F.pad(frame2, [0, pad_w, 0, 0], mode="reflect")
            w_padded = True
        Weight1, Alpha1, Beta1, Weight2, Alpha2, Beta2, Occlusion = self.get_kernel(
            moduleNormalize(frame0), moduleNormalize(frame2)
        )

        tensorAdaCoF1 = self.moduleAdaCoF(
            self.modulePad(frame0), Weight1, Alpha1, Beta1, self.dilation
        )
        tensorAdaCoF2 = self.moduleAdaCoF(
            self.modulePad(frame2), Weight2, Alpha2, Beta2, self.dilation
        )

        frame1 = Occlusion * tensorAdaCoF1 + (1 - Occlusion) * tensorAdaCoF2
        if h_padded:
            frame1 = frame1[:, :, 0:h0, :]
        if w_padded:
            frame1 = frame1[:, :, :, 0:w0]
        """
        if self.training:
            # Smoothness Terms
            m_Alpha1 = torch.mean(Weight1 * Alpha1, dim=1, keepdim=True)
            m_Alpha2 = torch.mean(Weight2 * Alpha2, dim=1, keepdim=True)
            m_Beta1 = torch.mean(Weight1 * Beta1, dim=1, keepdim=True)
            m_Beta2 = torch.mean(Weight2 * Beta2, dim=1, keepdim=True)

            g_Alpha1 = CharbonnierFunc(m_Alpha1[:, :, :, :-1] - m_Alpha1[:, :, :, 1:]) + CharbonnierFunc(m_Alpha1[:, :, :-1, :] - m_Alpha1[:, :, 1:, :])
            g_Beta1 = CharbonnierFunc(m_Beta1[:, :, :, :-1] - m_Beta1[:, :, :, 1:]) + CharbonnierFunc(m_Beta1[:, :, :-1, :] - m_Beta1[:, :, 1:, :])
            g_Alpha2 = CharbonnierFunc(m_Alpha2[:, :, :, :-1] - m_Alpha2[:, :, :, 1:]) + CharbonnierFunc(m_Alpha2[:, :, :-1, :] - m_Alpha2[:, :, 1:, :])
            g_Beta2 = CharbonnierFunc(m_Beta2[:, :, :, :-1] - m_Beta2[:, :, :, 1:]) + CharbonnierFunc(m_Beta2[:, :, :-1, :] - m_Beta2[:, :, 1:, :])
            g_Occlusion = CharbonnierFunc(Occlusion[:, :, :, :-1] - Occlusion[:, :, :, 1:]) + CharbonnierFunc(Occlusion[:, :, :-1, :] - Occlusion[:, :, 1:, :])

            g_Spatial = g_Alpha1 + g_Beta1 + g_Alpha2 + g_Beta2

            return {'frame1': frame1, 'g_Spatial': g_Spatial, 'g_Occlusion': g_Occlusion}
        else:
        """
        return frame1


class PrunedKernelEstimation(torch.nn.Module):
    def __init__(self, kernel_size):
        super(PrunedKernelEstimation, self).__init__()
        self.kernel_size = kernel_size

        self.moduleConv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=6, out_channels=24, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )
        self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=48, out_channels=51, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )
        self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=51, out_channels=99, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=99, out_channels=97, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=97, out_channels=94, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=94, out_channels=156, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=156, out_channels=142, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=142, out_channels=159, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )
        self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv5 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=159, out_channels=92, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=92, out_channels=72, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=72, out_channels=121, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )
        self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleDeconv5 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=121, out_channels=99, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=99, out_channels=69, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=69, out_channels=36, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )
        self.moduleUpsample5 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=36, out_channels=121, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.moduleDeconv4 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=121, out_channels=74, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=74, out_channels=83, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=83, out_channels=81, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )
        self.moduleUpsample4 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=81, out_channels=159, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.moduleDeconv3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=159, out_channels=83, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=83, out_channels=88, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=88, out_channels=72, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )
        self.moduleUpsample3 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=72, out_channels=94, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.moduleDeconv2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=94, out_channels=45, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=45, out_channels=47, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=47, out_channels=44, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )
        self.moduleUpsample2 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=44, out_channels=51, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.moduleWeight1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=51, out_channels=49, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=49, out_channels=21, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=21,
                out_channels=self.kernel_size**2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.Softmax(dim=1),
        )
        self.moduleAlpha1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=51, out_channels=50, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=50, out_channels=48, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=48, out_channels=20, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=20,
                out_channels=self.kernel_size**2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.moduleBeta1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=51, out_channels=50, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=50, out_channels=20, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=20,
                out_channels=self.kernel_size**2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.moduleWeight2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=51, out_channels=50, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=50, out_channels=49, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=49, out_channels=20, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=20,
                out_channels=self.kernel_size**2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.Softmax(dim=1),
        )
        self.moduleAlpha2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=51, out_channels=50, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=50, out_channels=49, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=49, out_channels=20, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=20,
                out_channels=self.kernel_size**2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.moduleBeta2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=51, out_channels=50, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=50, out_channels=20, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=20,
                out_channels=self.kernel_size**2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.moduleOcclusion = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=51, out_channels=52, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=52, out_channels=51, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=51, out_channels=48, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Conv2d(
                in_channels=48, out_channels=1, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.Sigmoid(),
        )

    def forward(self, rfield0, rfield2):
        tensorJoin = torch.cat([rfield0, rfield2], 1)

        tensorConv1 = self.moduleConv1(tensorJoin)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        tensorPool4 = self.modulePool4(tensorConv4)

        tensorConv5 = self.moduleConv5(tensorPool4)
        tensorPool5 = self.modulePool5(tensorConv5)

        tensorDeconv5 = self.moduleDeconv5(tensorPool5)
        tensorUpsample5 = self.moduleUpsample5(tensorDeconv5)

        tensorCombine = tensorUpsample5 + tensorConv5

        tensorDeconv4 = self.moduleDeconv4(tensorCombine)
        tensorUpsample4 = self.moduleUpsample4(tensorDeconv4)

        tensorCombine = tensorUpsample4 + tensorConv4

        tensorDeconv3 = self.moduleDeconv3(tensorCombine)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)

        tensorCombine = tensorUpsample3 + tensorConv3

        tensorDeconv2 = self.moduleDeconv2(tensorCombine)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)

        tensorCombine = tensorUpsample2 + tensorConv2

        Weight1 = self.moduleWeight1(tensorCombine)
        Alpha1 = self.moduleAlpha1(tensorCombine)
        Beta1 = self.moduleBeta1(tensorCombine)
        Weight2 = self.moduleWeight2(tensorCombine)
        Alpha2 = self.moduleAlpha2(tensorCombine)
        Beta2 = self.moduleBeta2(tensorCombine)
        Occlusion = self.moduleOcclusion(tensorCombine)

        return Weight1, Alpha1, Beta1, Weight2, Alpha2, Beta2, Occlusion
