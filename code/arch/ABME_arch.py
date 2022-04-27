"""
25-Sep-21
https://github.com/JunHeum/ABME/blob/master/model/ABMNet.py
https://github.com/JunHeum/ABME/blob/master/model/SBMNet.py
https://github.com/JunHeum/ABME/blob/master/model/SynthesisNet.py
https://github.com/JunHeum/ABME/blob/master/Upsample.py
https://github.com/JunHeum/ABME/blob/d9f04d160d6806204a384b29dc6a4821152bb77b/Bilateral_Correlation.py
https://github.com/JunHeum/ABME/blob/d9f04d160d6806204a384b29dc6a4821152bb77b/correlation_package/correlation.py
https://github.com/JunHeum/ABME/blob/d9f04d160d6806204a384b29dc6a4821152bb77b/utils.py
"""
from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F

# from Upsample import Upsample

import torch
from torch.nn.modules.module import Module
from torch.autograd import Function
import correlation_cuda

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms.functional as TF

import os
import torch
import torch.nn as nn
import time
import numpy as np

TAG_CHAR = np.array([202021.25], np.float32)


def load_flow(path):
    with open(path, "rb") as f:
        magic = float(np.fromfile(f, np.float32, count=1)[0])
        if magic == 202021.25:
            w, h = (
                np.fromfile(f, np.int32, count=1)[0],
                np.fromfile(f, np.int32, count=1)[0],
            )
            data = np.fromfile(f, np.float32, count=h * w * 2)
            data.resize((h, w, 2))
            return data
        return None


def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    if "startTime_for_tictoc" in globals():
        print(
            "Elapsed time is  " + str(time.time() - startTime_for_tictoc) + "  seconds"
        )
        # str(time.time() - startTime_for tictoc)
    else:
        print("Toc: start time not set")


def warp(x, flo, return_mask=True):
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
    yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)

    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.to(x.device)

    vgrid = torch.autograd.Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

    mask = mask.masked_fill_(mask < 0.999, 0)
    mask = mask.masked_fill_(mask > 0, 1)

    if return_mask:
        return output * mask, mask
    else:
        return output * mask


class BilateralCorrelation(nn.Module):
    def __init__(self, md=4):
        super(BilateralCorrelation, self).__init__()
        self.md = md  # displacement (default = 4pixels)
        self.grid = torch.ones(1).cuda()
        # default intermediate time step is 0.5 [Half]

        # per pixel displacement
        self.range = (md * 2 + 1) ** 2  # (default = 9*9 = 81)
        d_u = (
            torch.linspace(-self.md, self.md, 2 * self.md + 1)
            .view(1, -1)
            .repeat((2 * self.md + 1, 1))
            .view(self.range, 1)
        )  # (25,1)
        d_v = (
            torch.linspace(-self.md, self.md, 2 * self.md + 1)
            .view(-1, 1)
            .repeat((1, 2 * self.md + 1))
            .view(self.range, 1)
        )  # (25,1)

        self.d = torch.cat(
            (d_u, d_v), dim=1
        ).cuda()  # Per-pixel:(25,2) | Half-pixel: (81,2)

    def L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x**2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return x / norm

    def UniformGrid(self, Input):
        """
        Make uniform grid
        :param Input: tensor(N,C,H,W)
        :return grid: (N,2,H,W)
        """

        B, _, H, W = Input.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, 1, 1, W).expand(self.range, 1, H, W)
        yy = torch.arange(0, H).view(1, 1, H, 1).expand(self.range, 1, H, W)

        grid = torch.cat((xx, yy), 1).float()

        if Input.is_cuda:
            grid = grid.to(Input.device)

        return grid

    def warp(self, x, BM_d):
        vgrid = self.grid + BM_d
        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(x.size(3) - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(x.size(2) - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(
            x, vgrid, mode="bilinear", padding_mode="border", align_corners=True
        )  # 800MB memory occupied (d=2,C=64,H=256,W=256)
        mask = torch.ones(x.size(), device=x.device)
        # mask = torch.autograd.Variable(torch.ones(x.size())).to(x.Device)
        mask = nn.functional.grid_sample(
            mask, vgrid, align_corners=True
        )  # 300MB memory occpied (d=2,C=64,H=256,W=256)

        mask = mask.masked_fill_(mask < 0.999, 0)
        mask = mask.masked_fill_(mask > 0, 1)

        return output * mask

    def forward(self, feature1, feature2, SBM, time=0.5):
        """
        Return bilateral cost volume(Set of bilateral correlation map)
        :param feature1: feature at time t-1(N,C,H,W)
        :param feature2: feature at time t+1(N,C,H,W)
        :param SBM: (N,2,H,W)
        :param time(float): intermediate time step from 0 to 1 (default: 0.5 [half])
        :return BC: (N,(2d+1)^2,H,W)
        """
        feature1 = self.L2normalize(feature1)
        feature2 = self.L2normalize(feature2)

        if torch.equal(self.grid, torch.ones(1).cuda()):
            self.grid = torch.autograd.Variable(self.UniformGrid(SBM))

        if SBM.size(2) != self.grid.size(2) or SBM.size(3) != self.grid.size(3):
            self.grid = torch.autograd.Variable(self.UniformGrid(SBM))

        D_vol = self.d.view(1, self.range, 2, 1, 1).expand(
            SBM.size(0), -1, -1, SBM.size(2), SBM.size(3)
        )

        SBM_d = (
            SBM.view(SBM.size(0), 1, SBM.size(1), SBM.size(2), SBM.size(3)).expand(
                -1, self.range, -1, -1, -1
            )
            + D_vol
        )

        BC_list = []

        for i in range(SBM.size(0)):
            bw_feature = self.warp(
                feature1[i]
                .view((1,) + feature1[i].size())
                .expand(self.range, -1, -1, -1),
                2 * (-time) * SBM_d[i],
            )  # (D**2,C,H,W)
            fw_feature = self.warp(
                feature2[i]
                .view((1,) + feature2[i].size())
                .expand(self.range, -1, -1, -1),
                2 * (1 - time) * SBM_d[i],
            )  # (D**2,C,H,W)

            BC_list.append(
                torch.sum(torch.mul(fw_feature, bw_feature), dim=1).view(
                    1, self.range, SBM.size(2), SBM.size(3)
                )
            )

        return torch.cat(BC_list)


class CorrelationFunction(Function):

    # def __init__(self, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
    #     super(CorrelationFunction, self).__init__()
    #     self.pad_size = pad_size
    #     self.kernel_size = kernel_size
    #     self.max_displacement = max_displacement
    #     self.stride1 = stride1
    #     self.stride2 = stride2
    #     self.corr_multiply = corr_multiply
    #     # self.out_channel = ((max_displacement/stride2)*2 + 1) * ((max_displacement/stride2)*2 + 1)

    @staticmethod
    def forward(
        ctx,
        input1,
        input2,
        pad_size,
        kernel_size,
        max_displacement,
        stride1,
        stride2,
        corr_multiply,
    ):
        ctx.save_for_backward(input1, input2)
        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.corr_multiply = corr_multiply

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()

            correlation_cuda.forward(
                input1,
                input2,
                rbot1,
                rbot2,
                output,
                pad_size,
                kernel_size,
                max_displacement,
                stride1,
                stride2,
                corr_multiply,
            )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()

            grad_input1 = input1.new()
            grad_input2 = input2.new()

            correlation_cuda.backward(
                input1,
                input2,
                rbot1,
                rbot2,
                grad_output,
                grad_input1,
                grad_input2,
                ctx.pad_size,
                ctx.kernel_size,
                ctx.max_displacement,
                ctx.stride1,
                ctx.stride2,
                ctx.corr_multiply,
            )

        return grad_input1, grad_input2, None, None, None, None, None, None


class Correlation(Module):
    def __init__(
        self,
        pad_size=0,
        kernel_size=0,
        max_displacement=0,
        stride1=1,
        stride2=2,
        corr_multiply=1,
    ):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):

        input1 = input1.contiguous()
        input2 = input2.contiguous()
        # result = CorrelationFunction(self.pad_size, self.kernel_size, self.max_displacement,self.stride1, self.stride2, self.corr_multiply)(input1, input2)
        result = CorrelationFunction.apply(
            input1,
            input2,
            self.pad_size,
            self.kernel_size,
            self.max_displacement,
            self.stride1,
            self.stride2,
            self.corr_multiply,
        )

        return result


import numpy as np


def upsample_kernel2d(w, device):
    c = w // 2
    kernel = 1 - torch.abs(c - torch.arange(w, dtype=torch.float32, device=device)) / (
        c + 1
    )
    kernel = kernel.repeat(w).view(w, -1) * kernel.unsqueeze(1)
    return kernel.view(1, 1, w, w)


def Upsample(img, factor):
    if factor == 1:
        return img
    B, C, H, W = img.shape
    batch_img = img.view(B * C, 1, H, W)
    batch_img = F.pad(batch_img, [0, 1, 0, 1], mode="replicate")
    kernel = upsample_kernel2d(factor * 2 - 1, img.device)
    upsamp_img = F.conv_transpose2d(
        batch_img, kernel, stride=factor, padding=(factor - 1)
    )
    upsamp_img = upsamp_img[:, :, :-1, :-1]
    _, _, H_up, W_up = upsamp_img.shape
    return upsamp_img.view(B, C, H_up, W_up)


def conv_a(
    in_planes,
    out_planes,
    kernel_size=3,
    stride=1,
    padding=1,
    dilation=1,
    activation=True,
):
    if activation:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=True,
            ),
            nn.LeakyReLU(0.1),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=True,
            )
        )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def predict_mask(in_planes):
    return nn.Conv2d(in_planes, 1, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(
        in_planes, out_planes, kernel_size, stride, padding, bias=True
    )


class ABMRNet(nn.Module):
    """
    Asymmetric Bilateral Motion Refinement netwrok
    """

    def __init__(self, md=2):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping
        """
        super(ABMRNet, self).__init__()

        self.conv1a = conv(3, 16, kernel_size=3, stride=2)
        self.conv1aa = conv(16, 16, kernel_size=3, stride=1)
        self.conv1b = conv(16, 16, kernel_size=3, stride=1)
        self.conv2a = conv(16, 32, kernel_size=3, stride=2)
        self.conv2aa = conv(32, 32, kernel_size=3, stride=1)
        self.conv2b = conv(32, 32, kernel_size=3, stride=1)

        self.conv1_ASFM = conv(16, 16, kernel_size=3, stride=1)
        self.conv2_ASFM = conv(32, 32, kernel_size=3, stride=1)

        self.corr = Correlation(
            pad_size=md,
            kernel_size=1,
            max_displacement=md,
            stride1=1,
            stride2=1,
            corr_multiply=1,
        )
        self.leakyRELU = nn.LeakyReLU(0.1)

        nd = (2 * md + 1) ** 2
        dd = np.cumsum([64, 64, 48, 32, 16])

        od = nd + 32 + 2
        self.conv2_0 = conv(od, 64, kernel_size=3, stride=1)
        self.conv2_1 = conv(od + dd[0], 64, kernel_size=3, stride=1)
        self.conv2_2 = conv(od + dd[1], 48, kernel_size=3, stride=1)
        self.conv2_3 = conv(od + dd[2], 32, kernel_size=3, stride=1)
        self.conv2_4 = conv(od + dd[3], 16, kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od + dd[4])
        self.predict_mask2 = predict_mask(od + dd[4])
        self.upfeat2 = deconv(
            od + dd[4], 16, kernel_size=4, stride=2, padding=1
        )  # Updates at ASFM_Occ.py

        od = nd + 16 + 18
        self.conv1_0 = conv(od, 64, kernel_size=3, stride=1)
        self.conv1_1 = conv(od + dd[0], 64, kernel_size=3, stride=1)
        self.conv1_2 = conv(od + dd[1], 48, kernel_size=3, stride=1)
        self.conv1_3 = conv(od + dd[2], 32, kernel_size=3, stride=1)
        self.conv1_4 = conv(od + dd[3], 16, kernel_size=3, stride=1)
        self.predict_flow1 = predict_flow(od + dd[4])
        # self.deconv1 = deconv(2, 2, kernel_size=4, stride=2, padding=1) # Updates at ASFM_Occ.py

        self.dc_conv1 = conv(
            od + dd[4], 64, kernel_size=3, stride=1, padding=1, dilation=1
        )
        self.dc_conv2 = conv(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dc_conv3 = conv(64, 64, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dc_conv4 = conv(64, 48, kernel_size=3, stride=1, padding=8, dilation=8)
        self.dc_conv5 = conv(48, 32, kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(32, 16, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv7 = predict_flow(16)

        self.conv1_Res = conv_a(
            16, 16, kernel_size=3, stride=1, padding=1, activation=False
        )
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
        """

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
        yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)

        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.to(x.device)

        vgrid = torch.autograd.Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        mask = mask.masked_fill_(mask < 0.999, 0)
        mask = mask.masked_fill_(mask > 0, 1)

        return output * mask

    def forward(self, x, V_2, Z_2):
        """
        :param x: two input frames
        :param V: symmetric bilateral motion vector field
        :param Z: initial reliability map (1 denotes Non-occ & 0 denotes Occ)
        """

        im1 = x[:, :3, :, :]
        im2 = x[:, 3:, :, :]

        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))

        warp2 = self.warp(self.conv2_ASFM(c22), V_2 * 5.0)
        corr2 = self.corr(c12 * Z_2, warp2)
        corr2 = self.leakyRELU(corr2)

        x = torch.cat((corr2, c12, V_2), 1)
        x = torch.cat((self.conv2_0(x), x), 1)
        x = torch.cat((self.conv2_1(x), x), 1)
        x = torch.cat((self.conv2_2(x), x), 1)
        x = torch.cat((self.conv2_3(x), x), 1)
        x = torch.cat((self.conv2_4(x), x), 1)
        V_2 = V_2 + self.predict_flow2(x)
        Z_2 = self.predict_mask2(x)

        feat1 = self.leakyRELU(self.upfeat2(x))
        V_1 = Upsample(V_2, 2)
        Z_1 = Upsample(Z_2, 2)
        warp1 = self.warp(self.conv1_ASFM(c21), V_1 * 10.0)

        c11_Res = feat1
        c11 = (c11 * F.sigmoid(Z_1)) + self.conv1_Res(c11_Res)
        c11 = self.leakyRELU(c11)
        corr1 = self.corr(c11, warp1)
        corr1 = self.leakyRELU(corr1)
        x = torch.cat((corr1, c11, V_1, feat1), 1)
        x = torch.cat((self.conv1_0(x), x), 1)
        x = torch.cat((self.conv1_1(x), x), 1)
        x = torch.cat((self.conv1_2(x), x), 1)
        x = torch.cat((self.conv1_3(x), x), 1)
        x = torch.cat((self.conv1_4(x), x), 1)
        V_1 = V_1 + self.predict_flow1(x)

        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow = V_1 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))  # Normalized flow

        return F.interpolate(flow, scale_factor=2, mode="bilinear"), F.sigmoid(Z_1)


import torch
import torch.nn as nn
import numpy as np

# from Bilateral_Correlation import BilateralCorrelation


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.1),
    )


def predict_flow(in_channels):
    return nn.Conv2d(in_channels, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size, stride, padding, bias=True
    )


class SBMENet(nn.Module):
    """
    Symmetric Bilateral Motion Estimation network in BMBC
    """

    def __init__(self):
        super(SBMENet, self).__init__()

        self.conv1a = conv(3, 16, kernel_size=3, stride=2)  # Stride is '1' in BMBC
        self.conv1aa = conv(16, 16, kernel_size=3, stride=1)
        self.conv1b = conv(16, 16, kernel_size=3, stride=1)
        self.conv2a = conv(16, 32, kernel_size=3, stride=2)
        self.conv2aa = conv(32, 32, kernel_size=3, stride=1)
        self.conv2b = conv(32, 32, kernel_size=3, stride=1)
        self.conv3a = conv(32, 64, kernel_size=3, stride=2)
        self.conv3aa = conv(64, 64, kernel_size=3, stride=1)
        self.conv3b = conv(64, 64, kernel_size=3, stride=1)
        self.conv4a = conv(64, 96, kernel_size=3, stride=2)
        self.conv4aa = conv(96, 96, kernel_size=3, stride=1)
        self.conv4b = conv(96, 96, kernel_size=3, stride=1)
        self.conv5a = conv(96, 128, kernel_size=3, stride=2)
        self.conv5aa = conv(128, 128, kernel_size=3, stride=1)
        self.conv5b = conv(128, 128, kernel_size=3, stride=1)
        self.conv6aa = conv(128, 196, kernel_size=3, stride=2)
        self.conv6a = conv(196, 196, kernel_size=3, stride=1)
        self.conv6b = conv(196, 196, kernel_size=3, stride=1)

        self.leakyRELU = nn.LeakyReLU(0.1)

        # nd = (2*md + 1)**2
        dd = np.cumsum([128, 128, 96, 64, 32])

        od = (2 * 6 + 1) ** 2
        self.bilateral_corr6 = BilateralCorrelation(md=6)
        self.conv6_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv6_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv6_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow6 = predict_flow(od + dd[4])
        self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat6 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = (2 * 4 + 1) ** 2 + 128 * 2 + 4
        self.bilateral_corr5 = BilateralCorrelation(md=4)
        self.conv5_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv5_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv5_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow5 = predict_flow(od + dd[4])
        self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat5 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = (2 * 4 + 1) ** 2 + 96 * 2 + 4
        self.bilateral_corr4 = BilateralCorrelation(md=4)
        self.conv4_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv4_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv4_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(od + dd[4])
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat4 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = (2 * 2 + 1) ** 2 + 64 * 2 + 4
        self.bilateral_corr3 = BilateralCorrelation(md=2)
        self.conv3_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv3_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv3_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(od + dd[4])
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat3 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = (2 * 2 + 1) ** 2 + 32 * 2 + 4
        self.bilateral_corr2 = BilateralCorrelation(md=2)
        self.conv2_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv2_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv2_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od + dd[4])
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        self.dc_conv1 = conv(
            od + dd[4], 128, kernel_size=3, stride=1, padding=1, dilation=1
        )
        self.dc_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dc_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dc_conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8)
        self.dc_conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv7 = predict_flow(32)
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
        """

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
        yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)

        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.to(x.device)

        vgrid = torch.autograd.Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        mask = mask.masked_fill_(mask < 0.999, 0)
        mask = mask.masked_fill_(mask > 0, 1)

        return output * mask

    def forward(self, x):
        im1 = x[:, :3, :, :]
        im2 = x[:, 3:, :, :]

        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))

        bicorr6 = self.bilateral_corr6(
            c16, c26, torch.zeros(c16.size(0), 2, c16.size(2), c16.size(3)).to(c16)
        )
        bicorr6 = self.leakyRELU(bicorr6)

        x = torch.cat((self.conv6_0(bicorr6), bicorr6), 1)
        x = torch.cat((self.conv6_1(x), x), 1)
        x = torch.cat((self.conv6_2(x), x), 1)
        x = torch.cat((self.conv6_3(x), x), 1)
        x = torch.cat((self.conv6_4(x), x), 1)
        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        warp1_5 = self.warp(c15, up_flow6 * (-0.625))
        warp2_5 = self.warp(c25, up_flow6 * 0.625)
        bicorr5 = self.bilateral_corr5(c15, c25, up_flow6 * 0.625)
        bicorr5 = self.leakyRELU(bicorr5)
        x = torch.cat((bicorr5, warp1_5, warp2_5, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x), 1)
        x = torch.cat((self.conv5_1(x), x), 1)
        x = torch.cat((self.conv5_2(x), x), 1)
        x = torch.cat((self.conv5_3(x), x), 1)
        x = torch.cat((self.conv5_4(x), x), 1)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

        warp1_4 = self.warp(c14, up_flow5 * (-1.25))
        warp2_4 = self.warp(c24, up_flow5 * 1.25)
        bicorr4 = self.bilateral_corr4(c14, c24, up_flow5 * 1.25)
        bicorr4 = self.leakyRELU(bicorr4)
        x = torch.cat((bicorr4, warp1_4, warp2_4, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x), 1)
        x = torch.cat((self.conv4_1(x), x), 1)
        x = torch.cat((self.conv4_2(x), x), 1)
        x = torch.cat((self.conv4_3(x), x), 1)
        x = torch.cat((self.conv4_4(x), x), 1)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)

        warp1_3 = self.warp(c13, up_flow4 * (-2.5))
        warp2_3 = self.warp(c23, up_flow4 * 2.5)
        bicorr3 = self.bilateral_corr3(c13, c23, up_flow4 * 2.5)
        # temp_time += check_time() - cur_time
        bicorr3 = self.leakyRELU(bicorr3)
        x = torch.cat((bicorr3, warp1_3, warp2_3, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x), 1)
        x = torch.cat((self.conv3_1(x), x), 1)
        x = torch.cat((self.conv3_2(x), x), 1)
        x = torch.cat((self.conv3_3(x), x), 1)
        x = torch.cat((self.conv3_4(x), x), 1)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)

        warp1_2 = self.warp(c12, up_flow3 * (-5.0))
        warp2_2 = self.warp(c22, up_flow3 * 5.0)
        bicorr2 = self.bilateral_corr2(c12, c22, up_flow3 * 5.0)
        bicorr2 = self.leakyRELU(bicorr2)
        x = torch.cat((bicorr2, warp1_2, warp2_2, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x), 1)
        x = torch.cat((self.conv2_1(x), x), 1)
        x = torch.cat((self.conv2_2(x), x), 1)
        x = torch.cat((self.conv2_3(x), x), 1)
        x = torch.cat((self.conv2_4(x), x), 1)
        flow2 = self.predict_flow2(x)

        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        return flow2, flow3, flow4, flow5, flow6


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DynFilter(nn.Module):
    def __init__(self, kernel_size=(3, 3), padding=1, DDP=False):
        super(DynFilter, self).__init__()

        self.padding = padding

        filter_localexpand_np = np.reshape(
            np.eye(np.prod(kernel_size), np.prod(kernel_size)),
            (np.prod(kernel_size), 1, kernel_size[0], kernel_size[1]),
        )
        if DDP:
            self.register_buffer(
                "filter_localexpand", torch.FloatTensor(filter_localexpand_np)
            )  # for DDP model
        else:
            self.filter_localexpand = torch.FloatTensor(
                filter_localexpand_np
            ).cuda()  # for single model

    def forward(self, x, filter):
        x_localexpand = []

        for c in range(x.size(1)):
            x_localexpand.append(
                F.conv2d(
                    x[:, c : c + 1, :, :], self.filter_localexpand, padding=self.padding
                )
            )

        x_localexpand = torch.cat(x_localexpand, dim=1)
        x = torch.sum(torch.mul(x_localexpand, filter), dim=1).unsqueeze(1)

        return x


class Feature_Pyramid(nn.Module):
    def __init__(self):
        super(Feature_Pyramid, self).__init__()

        self.Feature_First = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
        )

        self.Feature_Second = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
        )

        self.Feature_Third = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
        )

    def forward(self, Input):
        Feature_1 = self.Feature_First(Input)
        Feature_2 = self.Feature_Second(Feature_1)
        Feature_3 = self.Feature_Third(Feature_2)

        return Feature_1, Feature_2, Feature_3


class GridNet_Filter(nn.Module):
    def __init__(self, output_channel):
        super(GridNet_Filter, self).__init__()

        def First(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=intInput,
                    out_channels=intOutput,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
                torch.nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=intOutput,
                    out_channels=intOutput,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
            )

        def lateral(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=intInput,
                    out_channels=intOutput,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
                torch.nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=intOutput,
                    out_channels=intOutput,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
            )

        def downsampling(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=intInput,
                    out_channels=intOutput,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                ),
                torch.nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=intOutput,
                    out_channels=intOutput,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
            )

        def upsampling(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                torch.nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=intInput,
                    out_channels=intOutput,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
                torch.nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=intOutput,
                    out_channels=intOutput,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
            )

        def Last(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=intInput,
                    out_channels=intOutput,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
                torch.nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=intOutput,
                    out_channels=intOutput,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
            )

        self.First_Block = First(4 * (3 + 32), 32)  # 4*RGB(3) + 4* 1st features(32)

        self.Row1_1 = lateral(32, 32)
        self.Row1_2 = lateral(32, 32)
        self.Row1_3 = lateral(32, 32)
        self.Row1_4 = lateral(32, 32)
        self.Row1_5 = lateral(32, 32)
        self.Last_Block = Last(32, output_channel)

        self.Row2_0 = First(4 * 64, 64)

        self.Row2_1 = lateral(64, 64)
        self.Row2_2 = lateral(64, 64)
        self.Row2_3 = lateral(64, 64)
        self.Row2_4 = lateral(64, 64)
        self.Row2_5 = lateral(64, 64)

        self.Row3_0 = First(4 * 96, 96)

        self.Row3_1 = lateral(96, 96)
        self.Row3_2 = lateral(96, 96)
        self.Row3_3 = lateral(96, 96)
        self.Row3_4 = lateral(96, 96)
        self.Row3_5 = lateral(96, 96)

        self.Col1_1 = downsampling(32, 64)
        self.Col2_1 = downsampling(64, 96)
        self.Col1_2 = downsampling(32, 64)
        self.Col2_2 = downsampling(64, 96)
        self.Col1_3 = downsampling(32, 64)
        self.Col2_3 = downsampling(64, 96)

        self.Col1_4 = upsampling(64, 32)
        self.Col2_4 = upsampling(96, 64)
        self.Col1_5 = upsampling(64, 32)
        self.Col2_5 = upsampling(96, 64)
        self.Col1_6 = upsampling(64, 32)
        self.Col2_6 = upsampling(96, 64)

    def forward(self, V_0_t_SBM, V_0_t_ABM, V_1_t_SBM, V_1_t_ABM):
        Variable1_1 = self.First_Block(
            torch.cat((V_0_t_SBM[0], V_0_t_ABM[0], V_1_t_SBM[0], V_1_t_ABM[0]), dim=1)
        )  # 1
        Variable1_2 = self.Row1_1(Variable1_1) + Variable1_1  # 2
        Variable1_3 = self.Row1_2(Variable1_2) + Variable1_2  # 3

        Variable2_0 = self.Row2_0(
            torch.cat(
                (
                    V_0_t_SBM[1][:, 3:, :, :],
                    V_0_t_ABM[1][:, 3:, :, :],
                    V_1_t_SBM[1][:, 3:, :, :],
                    V_1_t_ABM[1][:, 3:, :, :],
                ),
                dim=1,
            )
        )  # 4
        Variable2_1 = self.Col1_1(Variable1_1) + Variable2_0  # 5
        Variable2_2 = (
            self.Col1_2(Variable1_2) + self.Row2_1(Variable2_1) + Variable2_1
        )  # 6
        Variable2_3 = (
            self.Col1_3(Variable1_3) + self.Row2_2(Variable2_2) + Variable2_2
        )  # 7

        Variable3_0 = self.Row3_0(
            torch.cat(
                (
                    V_0_t_SBM[2][:, 3:, :, :],
                    V_0_t_ABM[2][:, 3:, :, :],
                    V_1_t_SBM[2][:, 3:, :, :],
                    V_1_t_ABM[2][:, 3:, :, :],
                ),
                dim=1,
            )
        )  # 8
        Variable3_1 = self.Col2_1(Variable2_1) + Variable3_0  # 9
        Variable3_2 = (
            self.Col2_2(Variable2_2) + self.Row3_1(Variable3_1) + Variable3_1
        )  # 10
        Variable3_3 = (
            self.Col2_3(Variable2_3) + self.Row3_2(Variable3_2) + Variable3_2
        )  # 11

        Variable3_4 = self.Row3_3(Variable3_3) + Variable3_3  # 10
        Variable3_5 = self.Row3_4(Variable3_4) + Variable3_4  # 11
        Variable3_6 = self.Row3_5(Variable3_5) + Variable3_5  # 12

        Variable2_4 = (
            self.Col2_4(Variable3_4) + self.Row2_3(Variable2_3) + Variable2_3
        )  # 13
        Variable2_5 = (
            self.Col2_5(Variable3_5) + self.Row2_4(Variable2_4) + Variable2_4
        )  # 14
        Variable2_6 = (
            self.Col2_6(Variable3_6) + self.Row2_5(Variable2_5) + Variable2_5
        )  # 15

        Variable1_4 = (
            self.Col1_4(Variable2_4) + self.Row1_3(Variable1_3) + Variable1_3
        )  # 16
        Variable1_5 = (
            self.Col1_5(Variable2_5) + self.Row1_4(Variable1_4) + Variable1_4
        )  # 17
        Variable1_6 = (
            self.Col1_6(Variable2_6) + self.Row1_5(Variable1_5) + Variable1_5
        )  # 18

        return self.Last_Block(Variable1_6)  # 19


class GridNet_Refine(nn.Module):
    def __init__(self):
        super(GridNet_Refine, self).__init__()

        def First(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=intInput,
                    out_channels=intOutput,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
                torch.nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=intOutput,
                    out_channels=intOutput,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
            )

        def lateral(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=intInput,
                    out_channels=intOutput,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
                torch.nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=intOutput,
                    out_channels=intOutput,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
            )

        def downsampling(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=intInput,
                    out_channels=intOutput,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                ),
                torch.nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=intOutput,
                    out_channels=intOutput,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
            )

        def upsampling(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                torch.nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=intInput,
                    out_channels=intOutput,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
                torch.nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=intOutput,
                    out_channels=intOutput,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
            )

        def Last(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=intInput,
                    out_channels=intOutput,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
                torch.nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=intOutput,
                    out_channels=intOutput,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
            )

        self.First_Block = First(3 + 32 + 4 * 32, 32)

        self.Row1_1 = lateral(32, 32)
        self.Row1_2 = lateral(32, 32)
        self.Row1_3 = lateral(32, 32)
        self.Row1_4 = lateral(32, 32)
        self.Row1_5 = lateral(32, 32)
        self.Last_Block = Last(32, 3)

        self.Row2_0 = First(4 * 64, 64)

        self.Row2_1 = lateral(64, 64)
        self.Row2_2 = lateral(64, 64)
        self.Row2_3 = lateral(64, 64)
        self.Row2_4 = lateral(64, 64)
        self.Row2_5 = lateral(64, 64)

        self.Row3_0 = First(4 * 96, 96)

        self.Row3_1 = lateral(96, 96)
        self.Row3_2 = lateral(96, 96)
        self.Row3_3 = lateral(96, 96)
        self.Row3_4 = lateral(96, 96)
        self.Row3_5 = lateral(96, 96)

        self.Col1_1 = downsampling(32, 64)
        self.Col2_1 = downsampling(64, 96)
        self.Col1_2 = downsampling(32, 64)
        self.Col2_2 = downsampling(64, 96)
        self.Col1_3 = downsampling(32, 64)
        self.Col2_3 = downsampling(64, 96)

        self.Col1_4 = upsampling(64, 32)
        self.Col2_4 = upsampling(96, 64)
        self.Col1_5 = upsampling(64, 32)
        self.Col2_5 = upsampling(96, 64)
        self.Col1_6 = upsampling(64, 32)
        self.Col2_6 = upsampling(96, 64)

    def forward(self, V_t, V_SBM_bw, V_ABM_bw, V_SBM_fw, V_ABM_fw):
        Variable1_1 = self.First_Block(
            torch.cat(
                (
                    V_t,
                    V_SBM_bw[0][:, 3:, :, :],
                    V_ABM_bw[0][:, 3:, :, :],
                    V_SBM_fw[0][:, 3:, :, :],
                    V_ABM_fw[0][:, 3:, :, :],
                ),
                dim=1,
            )
        )  # 1
        Variable1_2 = self.Row1_1(Variable1_1) + Variable1_1  # 2
        Variable1_3 = self.Row1_2(Variable1_2) + Variable1_2  # 3

        Variable2_0 = self.Row2_0(
            torch.cat(
                (
                    V_SBM_bw[1][:, 3:, :, :],
                    V_ABM_bw[1][:, 3:, :, :],
                    V_SBM_fw[1][:, 3:, :, :],
                    V_ABM_fw[1][:, 3:, :, :],
                ),
                dim=1,
            )
        )  # 4
        Variable2_1 = self.Col1_1(Variable1_1) + Variable2_0  # 5
        Variable2_2 = (
            self.Col1_2(Variable1_2) + self.Row2_1(Variable2_1) + Variable2_1
        )  # 6
        Variable2_3 = (
            self.Col1_3(Variable1_3) + self.Row2_2(Variable2_2) + Variable2_2
        )  # 7

        Variable3_0 = self.Row3_0(
            torch.cat(
                (
                    V_SBM_bw[2][:, 3:, :, :],
                    V_ABM_bw[2][:, 3:, :, :],
                    V_SBM_fw[2][:, 3:, :, :],
                    V_ABM_fw[2][:, 3:, :, :],
                ),
                dim=1,
            )
        )  # 8
        Variable3_1 = self.Col2_1(Variable2_1) + Variable3_0  # 9
        Variable3_2 = (
            self.Col2_2(Variable2_2) + self.Row3_1(Variable3_1) + Variable3_1
        )  # 10
        Variable3_3 = (
            self.Col2_3(Variable2_3) + self.Row3_2(Variable3_2) + Variable3_2
        )  # 11

        Variable3_4 = self.Row3_3(Variable3_3) + Variable3_3  # 12
        Variable3_5 = self.Row3_4(Variable3_4) + Variable3_4  # 13
        Variable3_6 = self.Row3_5(Variable3_5) + Variable3_5  # 14

        Variable2_4 = (
            self.Col2_4(Variable3_4) + self.Row2_3(Variable2_3) + Variable2_3
        )  # 15
        Variable2_5 = (
            self.Col2_5(Variable3_5) + self.Row2_4(Variable2_4) + Variable2_4
        )  # 16
        Variable2_6 = (
            self.Col2_6(Variable3_6) + self.Row2_5(Variable2_5) + Variable2_5
        )  # 17

        Variable1_4 = (
            self.Col1_4(Variable2_4) + self.Row1_3(Variable1_3) + Variable1_3
        )  # 18
        Variable1_5 = (
            self.Col1_5(Variable2_5) + self.Row1_4(Variable1_4) + Variable1_4
        )  # 19
        Variable1_6 = (
            self.Col1_6(Variable2_6) + self.Row1_5(Variable1_5) + Variable1_5
        )  # 20

        return self.Last_Block(Variable1_6)  # 21


class SynthesisNet(nn.Module):
    def __init__(self):
        super(SynthesisNet, self).__init__()

        self.ctxNet = Feature_Pyramid()

        self.FilterNet = GridNet_Filter(3 * 3 * 4)

        self.RefineNet = GridNet_Refine()

        self.Filtering = DynFilter(kernel_size=(3, 3), padding=1, DDP=False)

    def warp(self, x, flo):
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
        yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)

        grid = torch.cat((xx, yy), 1).float().to(x.device)

        vgrid = torch.autograd.Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        mask = mask.masked_fill_(mask < 0.999, 0)
        mask = mask.masked_fill_(mask > 0, 1)

        return output * mask

    def Flow_pyramid(self, flow):
        flow_pyr = []
        flow_pyr.append(flow)
        for i in range(1, 3):
            flow_pyr.append(
                F.interpolate(flow, scale_factor=0.5**i, mode="bilinear") * (0.5**i)
            )
        return flow_pyr

    def Img_pyramid(self, Img):
        img_pyr = []
        img_pyr.append(Img)
        for i in range(1, 3):
            img_pyr.append(F.interpolate(Img, scale_factor=0.5**i, mode="bilinear"))
        return img_pyr

    def forward(self, input, time_step=0.5):
        I0 = input[:, :3, :, :]  # First frame
        I1 = input[:, 3:6, :, :]  # Second frame
        SBM_t_1 = input[:, 6:8, :, :]
        SBM_Pyr_t_1 = self.Flow_pyramid(SBM_t_1)
        ABM_t_1 = input[:, 8:10, :, :]
        ABM_t_0 = input[:, 10:12, :, :]

        ABM_Pyr_t_0 = self.Flow_pyramid(ABM_t_0)
        ABM_Pyr_t_1 = self.Flow_pyramid(ABM_t_1)

        V_Pyr_0 = self.ctxNet(I0)  # Feature pyramid of first frame
        V_Pyr_1 = self.ctxNet(I1)  # Feature pyramid of second frame

        I_Pyr_0 = self.Img_pyramid(I0)
        I_Pyr_1 = self.Img_pyramid(I1)

        V_Pyr_0_t_SBM = []
        V_Pyr_1_t_SBM = []
        V_Pyr_0_t_ABM = []
        V_Pyr_1_t_ABM = []

        for i in range(3):
            V_0_t_SBM = self.warp(
                torch.cat((I_Pyr_0[i], V_Pyr_0[i]), dim=1), SBM_Pyr_t_1[i] * (-1)
            )
            V_0_t_ABM = self.warp(
                torch.cat((I_Pyr_0[i], V_Pyr_0[i]), dim=1), ABM_Pyr_t_0[i]
            )

            V_1_t_SBM = self.warp(
                torch.cat((I_Pyr_1[i], V_Pyr_1[i]), dim=1), SBM_Pyr_t_1[i]
            )
            V_1_t_ABM = self.warp(
                torch.cat((I_Pyr_1[i], V_Pyr_1[i]), dim=1), ABM_Pyr_t_1[i]
            )

            V_Pyr_0_t_SBM.append(V_0_t_SBM)
            V_Pyr_0_t_ABM.append(V_0_t_ABM)

            V_Pyr_1_t_SBM.append(V_1_t_SBM)
            V_Pyr_1_t_ABM.append(V_1_t_ABM)

        DF = F.softmax(
            self.FilterNet(V_Pyr_0_t_SBM, V_Pyr_0_t_ABM, V_Pyr_1_t_SBM, V_Pyr_1_t_ABM),
            dim=1,
        )

        Filtered_input = []
        for i in range(V_Pyr_0_t_SBM[0].size(1)):
            Filtered_input.append(
                self.Filtering(
                    torch.cat(
                        (
                            V_Pyr_0_t_SBM[0][:, i : i + 1, :, :],
                            V_Pyr_0_t_ABM[0][:, i : i + 1, :, :],
                            V_Pyr_1_t_SBM[0][:, i : i + 1, :, :],
                            V_Pyr_1_t_ABM[0][:, i : i + 1, :, :],
                        ),
                        dim=1,
                    ),
                    DF,
                )
            )

        Filtered_t = torch.cat(Filtered_input, dim=1)

        R_t = self.RefineNet(
            Filtered_t, V_Pyr_0_t_SBM, V_Pyr_0_t_ABM, V_Pyr_1_t_SBM, V_Pyr_1_t_ABM
        )

        output = Filtered_t[:, :3, :, :] + R_t

        return output


class ABME(nn.Module):
    def __init__(self):
        super(ABME, self).__init__()

        self.SBMNet = SBMENet()
        self.ABMNet = ABMRNet()
        self.SynNet = SynthesisNet()

    def forward(self, frame1, frame3):

        H = frame1.shape[2]
        W = frame1.shape[3]
        # 4K video requires GPU memory of more than 24GB. We recommend crop it into 4 regions with some margin.
        if H < 512:
            divisor = 64.0
            D_factor = 1.0
        else:
            divisor = 128.0
            D_factor = 0.5

        H_ = int(ceil(H / divisor) * divisor * D_factor)
        W_ = int(ceil(W / divisor) * divisor * D_factor)

        frame1_ = F.interpolate(frame1, (H_, W_), mode="bicubic")
        frame3_ = F.interpolate(frame3, (H_, W_), mode="bicubic")

        SBM = self.SBMNet(torch.cat((frame1_, frame3_), dim=1))[0]
        SBM_ = F.interpolate(SBM, scale_factor=4, mode="bilinear") * 20.0

        frame2_1, Mask2_1 = warp(frame1_, SBM_ * (-1), return_mask=True)
        frame2_3, Mask2_3 = warp(frame3_, SBM_, return_mask=True)

        frame2_Anchor_ = (frame2_1 + frame2_3) / 2
        frame2_Anchor = frame2_Anchor_ + 0.5 * (
            frame2_3 * (1 - Mask2_1) + frame2_1 * (1 - Mask2_3)
        )

        Z = F.l1_loss(frame2_3, frame2_1, reduction="none").mean(1, True)
        Z_ = F.interpolate(Z, scale_factor=0.25, mode="bilinear") * (-20.0)

        ABM_bw, _ = self.ABMNet(
            torch.cat((frame2_Anchor, frame1_), dim=1), SBM * (-1), Z_.exp()
        )
        ABM_fw, _ = self.ABMNet(
            torch.cat((frame2_Anchor, frame3_), dim=1), SBM, Z_.exp()
        )

        SBM_ = F.interpolate(SBM, (H, W), mode="bilinear") * 20.0
        ABM_fw = F.interpolate(ABM_fw, (H, W), mode="bilinear") * 20.0
        ABM_bw = F.interpolate(ABM_bw, (H, W), mode="bilinear") * 20.0

        SBM_[:, 0, :, :] *= W / float(W_)
        SBM_[:, 1, :, :] *= H / float(H_)
        ABM_fw[:, 0, :, :] *= W / float(W_)
        ABM_fw[:, 1, :, :] *= H / float(H_)
        ABM_bw[:, 0, :, :] *= W / float(W_)
        ABM_bw[:, 1, :, :] *= H / float(H_)

        divisor = 8.0
        H_ = int(ceil(H / divisor) * divisor)
        W_ = int(ceil(W / divisor) * divisor)

        Syn_inputs = torch.cat((frame1, frame3, SBM_, ABM_fw, ABM_bw), dim=1)

        Syn_inputs = F.interpolate(Syn_inputs, (H_, W_), mode="bilinear")
        Syn_inputs[:, 6, :, :] *= float(W_) / W
        Syn_inputs[:, 7, :, :] *= float(H_) / H
        Syn_inputs[:, 8, :, :] *= float(W_) / W
        Syn_inputs[:, 9, :, :] *= float(H_) / H
        Syn_inputs[:, 10, :, :] *= float(W_) / W
        Syn_inputs[:, 11, :, :] *= float(H_) / H

        result = self.SynNet(Syn_inputs)

        result = F.interpolate(result, (H, W), mode="bicubic")

        return result
