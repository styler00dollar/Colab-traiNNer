"""
https://github.com/sagarb96/Video-Frame-Interpolation-via-Adaptive-Seperable-Convolution
net.py
sepconv.py
fe.py
rt_fe.py
subnet.py
init_blk.py
asymmetric_blk.py
downsampling_blk.py
upsampling_blk.py
bottleneck_blk.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cupy
import re


class BottleNeck(nn.Module):
    """
    Default BottleNeck Convolution Block for the ENet Model, with optional dilation
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_1x1_channels,
        dilation=(1, 1),
        dropout_p=0.1,
    ):
        super().__init__()

        self.conv_blk = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=conv_1x1_channels,
                kernel_size=(1, 1),
            ),
            nn.BatchNorm2d(num_features=conv_1x1_channels),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=conv_1x1_channels,
                out_channels=conv_1x1_channels,
                kernel_size=(3, 3),
                padding="same",
                dilation=dilation,
            ),
            nn.BatchNorm2d(num_features=conv_1x1_channels),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=conv_1x1_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.PReLU(),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, X):
        """
        X: (batch, 3, height, width)
        """
        y_conv = self.conv_blk(X)
        return y_conv


class UpsampleBottleNeck(nn.Module):
    """
    UpSample BottleNeck Convolution Block for the ENet Model
    """

    def __init__(self, in_channels, out_channels, conv_1x1_channels, dropout_p=0.1):
        super().__init__()

        self.conv_blk = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=conv_1x1_channels,
                kernel_size=(2, 2),
                stride=(2, 2),
            ),
            nn.BatchNorm2d(num_features=conv_1x1_channels),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=conv_1x1_channels,
                out_channels=conv_1x1_channels,
                kernel_size=(3, 3),
                padding="same",
            ),
            nn.BatchNorm2d(num_features=conv_1x1_channels),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=conv_1x1_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.PReLU(),
            nn.Dropout(p=dropout_p),
        )

        self.unpool = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))
        self.unpool_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1)
        )

    def forward(self, X, pool_indices):
        """
        X: (batch, 3, height, width)
        """

        y_unpool = self.unpool_conv(X)
        y_unpool = self.unpool(y_unpool, pool_indices)

        y_conv = self.conv_blk(X)

        return y_conv + y_unpool


class DownsampleBottleNeck(nn.Module):
    """
    Downsampling BottleNeck Convolution Block for the ENet Model
    """

    def __init__(self, in_channels, out_channels, device, dropout_p=0.1):
        super().__init__()
        self.device = device

        # Number of channels required to pad the max-pool operation
        self.n_pad_channels_req = out_channels - in_channels

        self.conv_blk = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(2, 2),
                stride=(2, 2),
            ),
            nn.BatchNorm2d(num_features=in_channels),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(3, 3),
                padding="same",
            ),
            nn.BatchNorm2d(num_features=in_channels),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1)
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.PReLU(),
            nn.Dropout(p=dropout_p),
        )

        self.max_pool = nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2), return_indices=True
        )

    def forward(self, X):
        """
        X: (batch, 3, height, width)
        """
        y_conv = self.conv_blk(X)
        y_pool, pool_indices = self.max_pool(X)

        # Now need to pad the channel dimension of max-pool with zeros so that
        # it matches with the output of the convolution block
        ip_shape = list(y_conv.shape)
        ip_shape[1] = self.n_pad_channels_req  # Change the channel dimension shape

        zero_pads = torch.zeros(ip_shape).to(self.device)

        y_pool_padded = torch.cat([y_pool, zero_pads], dim=1)

        # Need to do a sum of the (padded) pool layer and the convolution layer
        return y_pool_padded + y_conv, pool_indices


class AsymmetricBottleNeck(nn.Module):
    """
    Asymmetric BottleNeck Convolution Block for the ENet Model that uses (5, 1) and (1, 5) kernels
    """

    def __init__(self, in_channels, out_channels, conv_1x1_channels, dropout_p=0.1):
        super().__init__()

        self.conv_blk = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=conv_1x1_channels,
                kernel_size=(1, 1),
            ),
            nn.BatchNorm2d(num_features=conv_1x1_channels),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=conv_1x1_channels,
                out_channels=conv_1x1_channels,
                kernel_size=(5, 1),
                padding="same",
            ),
            nn.BatchNorm2d(num_features=conv_1x1_channels),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=conv_1x1_channels,
                out_channels=conv_1x1_channels,
                kernel_size=(1, 5),
                padding="same",
            ),
            nn.BatchNorm2d(num_features=conv_1x1_channels),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=conv_1x1_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.PReLU(),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, X):
        """
        X: (batch, 3, height, width)
        """
        y_conv = self.conv_blk(X)
        return y_conv


class InitialConvBlock(nn.Module):
    """
    Initial Convolution Block for the ENet Model
    """

    def __init__(self, in_channels=6, out_channels=13):
        super().__init__()
        self.conv_blk = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.PReLU(),
            nn.BatchNorm2d(num_features=out_channels),
        )

        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, X):
        """
        X: (batch, 6, height, width)
        """
        y_conv = self.conv_blk(X)
        y_pool = self.max_pool(X)

        # Concatenate on the channel dimension
        # Convolution has 13 channels, and maxpool has 6 channels so the resulting channel dimension has 19 channels
        y_comb = torch.cat([y_conv, y_pool], dim=1)

        return y_comb


class SubNet(nn.Module):
    """
    The branching network after the backbone feature-extracting network
    NOTE: Input channels must be 64
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(3, 3),
                padding="same",
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(3, 3),
                padding="same",
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding="same",
            ),
            nn.ReLU(),
            # Final upsample to match the input frame shape
            # The backbone network is responsible for upscaling upto half the original frame size
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding="same",
            ),
        ]

        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)


def build_bottleneck_2(channels):
    """
    A function to build the bottleneck 2.X blocks
    Because there is a repeat, it is better to have it in the same place
    """
    conv_blk = nn.Sequential(
        BottleNeck(
            in_channels=channels,
            out_channels=channels,
            conv_1x1_channels=(channels // 2),
        ),
        BottleNeck(
            in_channels=channels,
            out_channels=channels,
            conv_1x1_channels=(channels // 2),
            dilation=(2, 2),
        ),
        AsymmetricBottleNeck(
            in_channels=channels,
            out_channels=channels,
            conv_1x1_channels=(channels // 2),
        ),
        BottleNeck(
            in_channels=channels,
            out_channels=channels,
            conv_1x1_channels=(channels // 2),
            dilation=(4, 4),
        ),
        BottleNeck(
            in_channels=channels,
            out_channels=channels,
            conv_1x1_channels=(channels // 2),
        ),
        BottleNeck(
            in_channels=channels,
            out_channels=channels,
            conv_1x1_channels=(channels // 2),
            dilation=(8, 8),
        ),
        AsymmetricBottleNeck(
            in_channels=channels,
            out_channels=channels,
            conv_1x1_channels=(channels // 2),
        ),
        BottleNeck(
            in_channels=channels,
            out_channels=channels,
            conv_1x1_channels=(channels // 2),
            dilation=(16, 16),
        ),
    )

    return conv_blk


class ENet_FeatureExtractor(nn.Module):
    """
    Class for the ENet model
    Paper: https://arxiv.org/abs/1606.02147
    """

    def __init__(self, device):
        super().__init__()

        self.device = device
        self.init_conv_blk = InitialConvBlock(in_channels=6)
        self.downsample_1 = DownsampleBottleNeck(
            in_channels=19, out_channels=64, dropout_p=0.01, device=device
        )

        # 4 bottleneck 1.X blocks
        self.bottleneck_1 = nn.Sequential(
            *[
                BottleNeck(
                    in_channels=64,
                    out_channels=64,
                    conv_1x1_channels=32,
                    dropout_p=0.01,
                )
                for _ in range(4)
            ]
        )

        self.downsample_2 = DownsampleBottleNeck(
            in_channels=64, out_channels=128, device=device
        )  # Using default dropout (0.1)
        self.bottleneck_2 = nn.Sequential(*[build_bottleneck_2(128) for _ in range(2)])

        # Naming the upsample in reverse to know which downsample operation it undoes
        self.upsample_2 = UpsampleBottleNeck(
            in_channels=128, out_channels=64, conv_1x1_channels=64
        )
        self.bottleneck_4 = nn.Sequential(
            *[
                BottleNeck(in_channels=64, out_channels=64, conv_1x1_channels=32)
                for _ in range(2)
            ]
        )

        self.upsample_1 = UpsampleBottleNeck(
            in_channels=64, out_channels=19, conv_1x1_channels=32
        )
        self.bottleneck_5 = BottleNeck(
            in_channels=19, out_channels=64, conv_1x1_channels=32
        )

        # Final layer -- Used a transposed convolution to upsample the image to the full resolution
        # NOTE: There is no final layer for this implementation. Also instead of 16 channels as output in
        #       self.bottleneck_5, 64 channels are used here (to be compliant with the Interpolation network)

    def forward(self, X):
        """
        X: (batch, channels, height, width)
        """
        y_init = self.init_conv_blk(X)

        y_ds_1, pool_idx_1 = self.downsample_1(y_init)
        y_bneck_1 = self.bottleneck_1(y_ds_1)

        y_ds_2, pool_idx_2 = self.downsample_2(y_bneck_1)
        y_bneck_2 = self.bottleneck_2(y_ds_2)

        y_us_2 = self.upsample_2(y_bneck_2, pool_idx_2)
        y_bneck_4 = self.bottleneck_4(y_us_2)

        y_us_1 = self.upsample_1(y_bneck_4, pool_idx_1)
        y_bneck_5 = self.bottleneck_5(y_us_1)

        return y_bneck_5


class DownSampleBlock(nn.Module):
    """
    Simple convolution block with ReLU activation for down-sampling

    Input size:  (batch, in_channels, height, width)
    Output size: (batch, out_channels, height//2, width//2)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Padding set to 'same' to ensure that the height/width remains the same as the input
        self.layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding="same",
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding="same",
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding="same",
            ),
            nn.ReLU(),
        ]

        # Now, down-sample the output by a factor of 2
        # Original paper (from their Github) uses Average Pooling, so using it here
        # NOTE: This is being written separately because we need to return the output BEFORE down-sampling
        #       for the skip-connection with up-sampling blocks
        self.pool = nn.AvgPool2d(
            kernel_size=(2, 2), stride=(2, 2), count_include_pad=False
        )

        # Build the network of convolutions
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        output = self.net(x)
        pooled_output = self.pool(output)

        return output, pooled_output


class UpSampleBlock(nn.Module):
    """
    Simple convolution block with ReLU activation for up-sampling

    Input size:  (batch, in_channels, height, width)
    Output size: (batch, out_channels, height*2, width*2)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Padding set to 'same' to ensure that the height/width remains the same as the input
        self.layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding="same",
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding="same",
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding="same",
            ),
            nn.ReLU(),
            # Now, up-sample the output by a factor of 2
            # Original paper (from their Github) uses BiLinear interpolation, so using it here
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding="same",
            ),
            nn.ReLU(),
        ]

        # Build the network of convolutions
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        output = self.net(x)
        return output


class FeatureExtractorNet(nn.Module):
    """
    Feature extraction network, based on the architecture provided in the above linked paper
    and their corresponding GitHub code
    """

    def __init__(self, device):
        super().__init__()

        # Series of down-sampling blocks
        self.ds_blk_1 = DownSampleBlock(6, 32)
        self.ds_blk_2 = DownSampleBlock(32, 64)
        self.ds_blk_3 = DownSampleBlock(64, 128)
        self.ds_blk_4 = DownSampleBlock(128, 256)
        self.ds_blk_5 = DownSampleBlock(256, 512)

        # Series of up-sampling blocks
        self.us_blk_5 = UpSampleBlock(512, 512)
        self.us_blk_4 = UpSampleBlock(512, 256)
        self.us_blk_3 = UpSampleBlock(256, 128)
        self.us_blk_2 = UpSampleBlock(128, 64)

    def forward(self, frames_comb):
        """
        Extracts the features from the two input (RGB) frames and returns it

        Input:  (batch, 3, height, width)
        Output: (batch, 64, height//2, width//2)

        NOTE: It is important to note that the height/width dimensions must be padded
              appropriately so that they are an exact multiple of 32. Or else, there will be dimension
              mismatch when doing the skip-connections (due to downsampling/upsampling layers)

              Why 32 ? There are 5 down-sampling blocks, so need input to be atleast multiple of 2^5 = 32
              They can be multiple of 64, 128 etc. as well, but multiple of 32 is the least we expect to avoid
              rounding-off when downsampling
        """

        # Down-sample the frames using the encoder net
        o1, ds_o1 = self.ds_blk_1(frames_comb)
        o2, ds_o2 = self.ds_blk_2(ds_o1)
        o3, ds_o3 = self.ds_blk_3(ds_o2)
        o4, ds_o4 = self.ds_blk_4(ds_o3)
        o5, ds_o5 = self.ds_blk_5(ds_o4)

        # Up-Sample the frames using the decoder net
        us_o5 = self.us_blk_5(ds_o5) + o5
        us_o4 = self.us_blk_4(us_o5) + o4
        us_o3 = self.us_blk_3(us_o4) + o3
        us_o2 = self.us_blk_2(us_o3) + o2

        # We stop here (no use for o1)
        return us_o2


kernel_Sepconv_updateOutput = """
    extern "C" __global__ void kernel_Sepconv_updateOutput(
        const int n,
        const float* input,
        const float* vertical,
        const float* horizontal,
        float* output
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        float dblOutput = 0.0;

        const int intSample = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output) ) % SIZE_0(output);
        const int intDepth  = ( intIndex / SIZE_3(output) / SIZE_2(output)                  ) % SIZE_1(output);
        const int intY      = ( intIndex / SIZE_3(output)                                   ) % SIZE_2(output);
        const int intX      = ( intIndex                                                    ) % SIZE_3(output);

        for (int intFilterY = 0; intFilterY < SIZE_1(vertical); intFilterY += 1) {
        for (int intFilterX = 0; intFilterX < SIZE_1(horizontal); intFilterX += 1) {
        dblOutput += VALUE_4(input, intSample, intDepth, intY + intFilterY, intX + intFilterX) * VALUE_4(vertical, intSample, intFilterY, intY, intX) * VALUE_4(horizontal, intSample, intFilterX, intY, intX);
            }
        }

        output[intIndex] = dblOutput;
    } }
"""

kernel_SeparableConvolution_updateGradVertical = """
    extern "C" __global__ void kernel_SeparableConvolution_updateGradVertical(
        const int n,
        const float* gradLoss,
        const float* input,
        const float* horizontal,
        float* gradVertical
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        float floatOutput = 0.0;
        float c = 0.0;

        const int intBatch   = ( intIndex / SIZE_3(gradVertical) / SIZE_2(gradVertical) / SIZE_1(gradVertical) ) % SIZE_0(gradVertical);
        const int intFilterY = ( intIndex / SIZE_3(gradVertical) / SIZE_2(gradVertical)                        ) % SIZE_1(gradVertical);
        const int intY       = ( intIndex / SIZE_3(gradVertical)                                               ) % SIZE_2(gradVertical);
        const int intX       = ( intIndex                                                                      ) % SIZE_3(gradVertical);

        for (int intFilterX = 0; intFilterX < SIZE_1(horizontal); intFilterX++) 
        {
            float product = VALUE_4(gradLoss, intBatch, 0, intY, intX)*              // channel 0
            VALUE_4(input, intBatch, 0, intY + intFilterY, intX + intFilterX)* 
            VALUE_4(horizontal, intBatch, intFilterX, intY, intX) +
            VALUE_4(gradLoss, intBatch, 1, intY, intX)*                          // channel 1     
            VALUE_4(input, intBatch, 1, intY + intFilterY, intX + intFilterX)* 
            VALUE_4(horizontal, intBatch, intFilterX, intY, intX) +
            VALUE_4(gradLoss, intBatch, 2, intY, intX)*                          // channel 2
            VALUE_4(input, intBatch, 2, intY + intFilterY, intX + intFilterX)* 
            VALUE_4(horizontal, intBatch, intFilterX, intY, intX);

            floatOutput += product;
        }

        gradVertical[intIndex] = floatOutput;
    } }
"""

kernel_SeparableConvolution_updateGradHorizontal = """
    extern "C" __global__ void kernel_SeparableConvolution_updateGradHorizontal(
        const int n,
        const float* gradLoss,
        const float* input,
        const float* vertical,
        float* gradHorizontal
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        float floatOutput = 0.0;
        float c = 0.0;

        const int intBatch   = ( intIndex / SIZE_3(gradHorizontal) / SIZE_2(gradHorizontal) / SIZE_1(gradHorizontal) ) % SIZE_0(gradHorizontal);
        const int intFilterX = ( intIndex / SIZE_3(gradHorizontal) / SIZE_2(gradHorizontal)                          ) % SIZE_1(gradHorizontal);
        const int intY       = ( intIndex / SIZE_3(gradHorizontal)                                                   ) % SIZE_2(gradHorizontal);
        const int intX       = ( intIndex                                                                            ) % SIZE_3(gradHorizontal);

        for (int intFilterY = 0; intFilterY < SIZE_1(vertical); intFilterY++)
        {
            float product = VALUE_4(gradLoss, intBatch, 0, intY, intX)*             // channel 0
            VALUE_4(input, intBatch, 0, intY + intFilterY, intX + intFilterX)* 
            VALUE_4(vertical, intBatch, intFilterY, intY, intX) + 
            VALUE_4(gradLoss, intBatch, 1, intY, intX)*                         // channel 1
            VALUE_4(input, intBatch, 1, intY + intFilterY, intX + intFilterX)* 
            VALUE_4(vertical, intBatch, intFilterY, intY, intX) + 
            VALUE_4(gradLoss, intBatch, 2, intY, intX)*                         // channel 2
            VALUE_4(input, intBatch, 2, intY + intFilterY, intX + intFilterX)* 
            VALUE_4(vertical, intBatch, intFilterY, intY, intX);

            float y = product - c;
            float t = floatOutput + y;
            c = (t - floatOutput) - y;
            floatOutput = t;
        }

        gradHorizontal[intIndex] = floatOutput;
    } }
"""


def cupy_kernel(strFunction, objectVariables):
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

    return strKernel


# end


@cupy.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)


# end


class FunctionSepconv(torch.autograd.Function):
    def __init__(self):
        super(FunctionSepconv, self).__init__()

    # end

    @staticmethod
    def forward(self, input, vertical, horizontal):
        self.save_for_backward(input, vertical, horizontal)

        intSample = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intFilterSize = min(vertical.size(1), horizontal.size(1))
        intOutputHeight = min(vertical.size(2), horizontal.size(2))
        intOutputWidth = min(vertical.size(3), horizontal.size(3))

        assert intInputHeight - intFilterSize == intOutputHeight - 1
        assert intInputWidth - intFilterSize == intOutputWidth - 1

        assert input.is_contiguous() is True
        assert vertical.is_contiguous() is True
        assert horizontal.is_contiguous() is True

        output = input.new_zeros(
            intSample, intInputDepth, intOutputHeight, intOutputWidth
        )

        if input.is_cuda is True:

            class Stream:
                ptr = torch.cuda.current_stream().cuda_stream

            # end

            n = output.nelement()
            cupy_launch(
                "kernel_Sepconv_updateOutput",
                cupy_kernel(
                    "kernel_Sepconv_updateOutput",
                    {
                        "input": input,
                        "vertical": vertical,
                        "horizontal": horizontal,
                        "output": output,
                    },
                ),
            )(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    n,
                    input.data_ptr(),
                    vertical.data_ptr(),
                    horizontal.data_ptr(),
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
    def backward(self, gradOutput):
        input, vertical, horizontal = self.saved_tensors

        intSample = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intFilterSize = min(vertical.size(1), horizontal.size(1))
        intOutputHeight = min(vertical.size(2), horizontal.size(2))
        intOutputWidth = min(vertical.size(3), horizontal.size(3))

        assert intInputHeight - intFilterSize == intOutputHeight - 1
        assert intInputWidth - intFilterSize == intOutputWidth - 1

        assert gradOutput.is_contiguous() is True

        gradInput = (
            input.new_zeros(intSample, intInputDepth, intInputHeight, intInputWidth)
            if self.needs_input_grad[0] is True
            else None
        )
        gradVertical = (
            input.new_zeros(intSample, intFilterSize, intOutputHeight, intOutputWidth)
            if self.needs_input_grad[1] is True
            else None
        )
        gradHorizontal = (
            input.new_zeros(intSample, intFilterSize, intOutputHeight, intOutputWidth)
            if self.needs_input_grad[2] is True
            else None
        )

        if input.is_cuda is True:

            class Stream:
                ptr = torch.cuda.current_stream().cuda_stream

            # end

            # vertical grad
            n_v = gradVertical.nelement()
            cupy_launch(
                "kernel_SeparableConvolution_updateGradVertical",
                cupy_kernel(
                    "kernel_SeparableConvolution_updateGradVertical",
                    {
                        "gradLoss": gradOutput,
                        "input": input,
                        "horizontal": horizontal,
                        "gradVertical": gradVertical,
                    },
                ),
            )(
                grid=tuple([int((n_v + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    n_v,
                    gradOutput.data_ptr(),
                    input.data_ptr(),
                    horizontal.data_ptr(),
                    gradVertical.data_ptr(),
                ],
                stream=Stream,
            )

            # horizontal grad
            n_h = gradHorizontal.nelement()
            cupy_launch(
                "kernel_SeparableConvolution_updateGradHorizontal",
                cupy_kernel(
                    "kernel_SeparableConvolution_updateGradHorizontal",
                    {
                        "gradLoss": gradOutput,
                        "input": input,
                        "vertical": vertical,
                        "gradHorizontal": gradHorizontal,
                    },
                ),
            )(
                grid=tuple([int((n_h + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    n_h,
                    gradOutput.data_ptr(),
                    input.data_ptr(),
                    vertical.data_ptr(),
                    gradHorizontal.data_ptr(),
                ],
                stream=Stream,
            )

        elif input.is_cuda is False:
            raise NotImplementedError()

        # end

        return gradInput, gradVertical, gradHorizontal


# end
# end

# ================= CONSTANTS ===========================================
FRAME_DIM_MULTIPLE = 32  # Frame dimensions must be a multiple of this
# =======================================================================


class InterpolationNet(nn.Module):
    """Main network that is responsible for outputting the kernels per pixel"""

    def __init__(self, real_time, device, in_channels=64, out_channels=51):
        super().__init__()

        self.device = device
        self.sep_conv_net = FunctionSepconv()
        self.input_pad_pixels = out_channels // 2
        self.input_pad = nn.ReplicationPad2d([self.input_pad_pixels] * 4)

        # Set the appropriate class references
        BackboneNet = ENet_FeatureExtractor if real_time else FeatureExtractorNet

        # TODO: Change the subnet as well ?
        # SubNet = subnet.SubNet

        self.backbone = BackboneNet(device)
        self.vnet_1 = SubNet(in_channels, out_channels)
        self.hnet_1 = SubNet(in_channels, out_channels)
        self.vnet_2 = SubNet(in_channels, out_channels)
        self.hnet_2 = SubNet(in_channels, out_channels)

    def forward(self, frame_prev, frame_next):
        """
        Returns the interpolated frame between frame_prev and frame_next
        Shape of each frame: (batch, channels, height, width)
        """
        h_prev, w_prev = frame_prev.shape[2:]
        h_next, w_next = frame_next.shape[2:]

        # Some sanity checks
        assert (h_prev == h_next) and (w_prev == w_next), "Frame sizes doesn't match"

        # Pad the frames appropriately in height/width, if they are not multiples of 32
        need_h_pad = (h_prev % FRAME_DIM_MULTIPLE) != 0
        need_w_pad = (w_prev % FRAME_DIM_MULTIPLE) != 0

        # If height is not a multiple of 32, pad it so that it is
        # They will be un-padded later on, from the resulting output
        # Padding has the following semantics: (left, right, top, down) from the LAST (right-most) dimension
        if need_h_pad:
            n_pad_pixels = FRAME_DIM_MULTIPLE - (h_prev % FRAME_DIM_MULTIPLE)
            frame_prev = F.pad(
                frame_prev, (0, 0, 0, n_pad_pixels)
            )  # Pad the bottom of the frame
            frame_next = F.pad(
                frame_next, (0, 0, 0, n_pad_pixels)
            )  # Pad the bottom of the frame

        # If the width is not a multiple of 32, pad it so that is is
        # They will be un-padded from the resulting output
        if need_w_pad:
            n_pad_pixels = FRAME_DIM_MULTIPLE - (w_prev % FRAME_DIM_MULTIPLE)
            frame_prev = F.pad(
                frame_prev, (0, n_pad_pixels, 0, 0)
            )  # Pad the right part of the frame
            frame_next = F.pad(
                frame_next, (0, n_pad_pixels, 0, 0)
            )  # Pad the right part of the frame

        # Now extract the features from the frames
        # Then send them to the corresponding subnets

        # Need to concatenate the frames in the channel-axis (which is axis-1)
        # That's what the paper does
        frames_comb = torch.cat([frame_prev, frame_next], axis=1)
        output_features = self.backbone(frames_comb)
        k1_v = self.vnet_1(output_features)
        k1_h = self.hnet_1(output_features)
        k2_v = self.vnet_2(output_features)
        k2_h = self.hnet_2(output_features)

        # Pad the input frames
        padded_frame_prev = self.input_pad(frame_prev)
        padded_frame_next = self.input_pad(frame_next)

        # NOTE: Following below requires CUDA or else will not work
        inter_frame_prev = self.sep_conv_net.apply(padded_frame_prev, k1_v, k1_h)
        inter_frame_next = self.sep_conv_net.apply(padded_frame_next, k2_v, k2_h)

        # Add the resulting outputs to get the final interpolated frame
        inter_frame = inter_frame_prev + inter_frame_next

        # If we had padded previously, remove the paddings
        if need_h_pad:
            inter_frame = inter_frame[:, :, :h_prev, :]
        if need_w_pad:
            inter_frame = inter_frame[:, :, :, :w_prev]

        return inter_frame
