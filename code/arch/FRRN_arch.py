"""
networks.py (18-12-20)
https://github.com/ZongyuGuo/Inpainting_FRRN/blob/master/src/networks.py
"""

import torch
import torch.nn as nn

# from .convolutions import partialconv2d
import pytorch_lightning as pl


class FRRNet(pl.LightningModule):
    def __init__(self, block_num=16):
        super(FRRNet, self).__init__()
        self.block_num = block_num
        self.dilation_num = block_num // 2
        blocks = []
        for _ in range(self.block_num):
            blocks.append(FRRBlock())
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, mask):
        x = x.type(torch.cuda.FloatTensor)
        mask = mask.type(torch.cuda.FloatTensor)

        mid_x = []
        mid_m = []

        mask_new = mask
        for index in range(self.dilation_num):
            x, _ = self.blocks[index * 2](x, mask_new, mask)
            x, mask_new = self.blocks[index * 2 + 1](x, mask_new, mask)
            mid_x.append(x)
            mid_m.append(mask_new)

        return x, mid_x, mid_m


class FRRBlock(pl.LightningModule):
    def __init__(self):
        super(FRRBlock, self).__init__()
        self.full_conv1 = PConvLayer(
            3, 32, kernel_size=5, stride=1, padding=2, use_norm=False
        )
        self.full_conv2 = PConvLayer(
            32, 32, kernel_size=5, stride=1, padding=2, use_norm=False
        )
        self.full_conv3 = PConvLayer(
            32, 3, kernel_size=5, stride=1, padding=2, use_norm=False
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.branch_conv1 = PConvLayer(
            3, 64, kernel_size=3, stride=2, padding=1, use_norm=False
        )
        self.branch_conv2 = PConvLayer(64, 96, kernel_size=3, stride=2, padding=1)
        self.branch_conv3 = PConvLayer(96, 128, kernel_size=3, stride=2, padding=1)
        self.branch_conv4 = PConvLayer(
            128, 96, kernel_size=3, stride=1, padding=1, act="LeakyReLU"
        )
        self.branch_conv5 = PConvLayer(
            96, 64, kernel_size=3, stride=1, padding=1, act="LeakyReLU"
        )
        self.branch_conv6 = PConvLayer(
            64, 3, kernel_size=3, stride=1, padding=1, act="Tanh"
        )

    def forward(self, input, mask, mask_ori):
        x = input
        out_f, mask_f = self.full_conv1(x, mask)
        out_f, mask_f = self.full_conv2(out_f, mask_f)
        out_f, mask_f = self.full_conv3(out_f, mask_f)

        out_b, mask_b = self.branch_conv1(x, mask)
        out_b, mask_b = self.branch_conv2(out_b, mask_b)
        out_b, mask_b = self.branch_conv3(out_b, mask_b)

        out_b = self.upsample(out_b)
        mask_b = self.upsample(mask_b)
        out_b, mask_b = self.branch_conv4(out_b, mask_b)
        out_b = self.upsample(out_b)
        mask_b = self.upsample(mask_b)
        out_b, mask_b = self.branch_conv5(out_b, mask_b)
        out_b = self.upsample(out_b)
        mask_b = self.upsample(mask_b)
        out_b, mask_b = self.branch_conv6(out_b, mask_b)

        mask_new = mask_f * mask_b
        out = (out_f * mask_new + out_b * mask_new) / 2 * (1 - mask_ori) + input
        # out = (out_f * mask_new + out_b * mask_new) / 2 * (1 - mask_ori) + input * mask_ori
        return out, mask_new


class PConvLayer(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        act="ReLU",
        use_norm=True,
    ):
        super(PConvLayer, self).__init__()
        self.conv = PartialConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            return_mask=True,
        )
        self.norm = nn.InstanceNorm2d(out_channels, track_running_stats=False)
        self.use_norm = use_norm
        if act == "ReLU":
            self.act = nn.ReLU(True)
        elif act == "LeakyReLU":
            self.act = nn.LeakyReLU(0.2, True)
        elif act == "Tanh":
            self.act = nn.Tanh()

    def forward(self, x, mask):
        x, mask_update = self.conv(x, mask)
        if self.use_norm:
            x = self.norm(x)
        x = self.act(x)
        return x, mask_update
