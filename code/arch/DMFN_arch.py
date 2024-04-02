"""
block.py (18-12-20)
https://github.com/Zheng222/DMFN/blob/master/models/block.py

architecture.py (18-12-20)
https://github.com/Zheng222/DMFN/blob/master/models/architecture.py
"""

import torch.nn as nn
import torch
import pytorch_lightning as pl


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=padding,
        bias=True,
        dilation=dilation,
        groups=groups,
    )


def _norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == "bn":
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == "in":
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError(
            "normalization layer [{:s}] is not found".format(norm_type)
        )
    return layer


def _activation(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == "relu":
        layer = nn.ReLU(inplace)
    elif act_type == "lrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            "activation layer [{:s}] is not found".format(act_type)
        )
    return layer


class conv_block(pl.LightningModule):
    def __init__(
        self,
        in_nc,
        out_nc,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        padding=0,
        norm="in",
        activation="relu",
        pad_type="zero",
    ):
        super(conv_block, self).__init__()
        if pad_type == "zero":
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == "replicate":
            self.pad = nn.ReplicationPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        if norm == "in":
            self.norm = nn.InstanceNorm2d(out_nc, affine=False)
        elif norm == "bn":
            self.norm = nn.BatchNorm2d(out_nc, affine=True)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported norm type: {}".format(norm)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.Conv2d(
            in_nc, out_nc, kernel_size, stride, 0, dilation, groups, bias
        )  # padding=0

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class upconv_block(pl.LightningModule):
    def __init__(
        self,
        in_nc,
        out_nc,
        kernel_size=3,
        stride=1,
        bias=True,
        padding=0,
        pad_type="zero",
        norm="none",
        activation="relu",
    ):
        super(upconv_block, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_nc, out_nc, 4, 2, 1)
        self.act = _activation("relu")
        self.norm = _norm("in", out_nc)

        self.conv = conv_block(
            out_nc,
            out_nc,
            kernel_size,
            stride,
            bias=bias,
            padding=padding,
            pad_type=pad_type,
            norm=norm,
            activation=activation,
        )

    def forward(self, x):
        x = self.act(self.norm(self.deconv(x)))
        x = self.conv(x)
        return x


class ResBlock_new(pl.LightningModule):
    def __init__(self, nc):
        super(ResBlock_new, self).__init__()
        self.c1 = conv_layer(nc, nc // 4, 3, 1)
        self.d1 = conv_layer(nc // 4, nc // 4, 3, 1, 1)  # rate = 1
        self.d2 = conv_layer(nc // 4, nc // 4, 3, 1, 2)  # rate = 2
        self.d3 = conv_layer(nc // 4, nc // 4, 3, 1, 4)  # rate = 4
        self.d4 = conv_layer(nc // 4, nc // 4, 3, 1, 8)  # rate = 8
        self.act = _activation("relu")
        self.norm = _norm("in", nc)
        self.c2 = conv_layer(nc, nc, 3, 1)  # fusion

    def forward(self, x):
        output1 = self.act(self.norm(self.c1(x)))
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)
        d4 = self.d4(output1)

        add1 = d1 + d2
        add2 = add1 + d3
        add3 = add2 + d4
        combine = torch.cat([d1, add1, add2, add3], 1)
        output2 = self.c2(self.act(self.norm(combine)))
        output = x + self.norm(output2)
        return output



# from . import block as B


class InpaintingGenerator(pl.LightningModule):
    def __init__(self, in_nc=4, out_nc=3, nf=64, n_res=8, norm="in", activation="relu"):
        super(InpaintingGenerator, self).__init__()
        self.encoder = nn.Sequential(  # input: [4, 256, 256]
            conv_block(
                in_nc, nf, 5, stride=1, padding=2, norm="none", activation=activation
            ),  # [64, 256, 256]
            conv_block(
                nf, 2 * nf, 3, stride=2, padding=1, norm=norm, activation=activation
            ),  # [128, 128, 128]
            conv_block(
                2 * nf, 2 * nf, 3, stride=1, padding=1, norm=norm, activation=activation
            ),  # [128, 128, 128]
            conv_block(
                2 * nf, 4 * nf, 3, stride=2, padding=1, norm=norm, activation=activation
            ),  # [256, 64, 64]
        )

        blocks = []
        for _ in range(n_res):
            block = ResBlock_new(4 * nf)
            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            conv_block(
                4 * nf, 4 * nf, 3, stride=1, padding=1, norm=norm, activation=activation
            ),  # [256, 64, 64]
            upconv_block(
                4 * nf,
                2 * nf,
                kernel_size=3,
                stride=1,
                padding=1,
                norm=norm,
                activation="relu",
            ),
            # [128, 128, 128]
            upconv_block(
                2 * nf,
                nf,
                kernel_size=3,
                stride=1,
                padding=1,
                norm=norm,
                activation="relu",
            ),
            # [64, 256, 256]
            conv_block(
                nf, out_nc, 3, stride=1, padding=1, norm="none", activation="tanh"
            ),  # [3, 256, 256]
        )

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        x = self.encoder(x)
        x = self.blocks(x)
        x = self.decoder(x)
        return x
