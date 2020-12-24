"""
model.py (13-12-20)
https://github.com/Yukariin/CSA_pytorch/blob/master/model.py
"""
import torch
from torch import nn
import torch.nn.functional as F


def get_norm(name, out_channels):
    if name == 'batch':
        norm = nn.BatchNorm2d(out_channels)
    elif name == 'instance':
        norm = nn.InstanceNorm2d(out_channels)
    else:
        norm = None
    return norm


def get_act(name):
    if name == 'relu':
        activation = nn.ReLU(inplace=True)
    elif name == 'elu':
        activation == nn.ELU(inplace=True)
    elif name == 'leaky_relu':
        activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif name == 'tanh':
        activation = nn.Tanh()
    elif name == 'sigmoid':
        activation = nn.Sigmoid()
    else:
        activation = None
    return activation


class CoarseEncodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 normalization=None, activation=None):
        super().__init__()

        layers = []
        if activation:
            layers.append(get_act(activation))
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1))
        if normalization:
            layers.append(get_norm(normalization, out_channels))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class CoarseDecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 normalization=None, activation=None):
        super().__init__()

        layers = []
        if activation:
            layers.append(get_act(activation))
        layers.append(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=1))
        if normalization:
            layers.append(get_norm(normalization, out_channels))
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class CoarseNet(nn.Module):
    def __init__(self, c_img=3,
                 norm='instance', act_en='leaky_relu', act_de='relu'):
        super().__init__()

        cnum = 64

        self.en_1 = nn.Conv2d(c_img, cnum, 4, 2, padding=1)
        self.en_2 = CoarseEncodeBlock(cnum, cnum*2, 4, 2, normalization=norm, activation=act_en)
        self.en_3 = CoarseEncodeBlock(cnum*2, cnum*4, 4, 2, normalization=norm, activation=act_en)
        self.en_4 = CoarseEncodeBlock(cnum*4, cnum*8, 4, 2, normalization=norm, activation=act_en)
        self.en_5 = CoarseEncodeBlock(cnum*8, cnum*8, 4, 2, normalization=norm, activation=act_en)
        self.en_6 = CoarseEncodeBlock(cnum*8, cnum*8, 4, 2, normalization=norm, activation=act_en)
        self.en_7 = CoarseEncodeBlock(cnum*8, cnum*8, 4, 2, normalization=norm, activation=act_en)
        self.en_8 = CoarseEncodeBlock(cnum*8, cnum*8, 4, 2, activation=act_en)

        self.de_8 = CoarseDecodeBlock(cnum*8, cnum*8, 4, 2, normalization=norm, activation=act_de)
        self.de_7 = CoarseDecodeBlock(cnum*8*2, cnum*8, 4, 2, normalization=norm, activation=act_de)
        self.de_6 = CoarseDecodeBlock(cnum*8*2, cnum*8, 4, 2, normalization=norm, activation=act_de)
        self.de_5 = CoarseDecodeBlock(cnum*8*2, cnum*8, 4, 2, normalization=norm, activation=act_de)
        self.de_4 = CoarseDecodeBlock(cnum*8*2, cnum*4, 4, 2, normalization=norm, activation=act_de)
        self.de_3 = CoarseDecodeBlock(cnum*4*2, cnum*2, 4, 2, normalization=norm, activation=act_de)
        self.de_2 = CoarseDecodeBlock(cnum*2*2, cnum, 4, 2, normalization=norm, activation=act_de)
        self.de_1 = nn.Sequential(
            get_act(act_de),
            nn.ConvTranspose2d(cnum*2, c_img, 4, 2, padding=1),
            get_act('tanh'))

    def forward(self, x):
        out_1 = self.en_1(x)
        out_2 = self.en_2(out_1)
        out_3 = self.en_3(out_2)
        out_4 = self.en_4(out_3)
        out_5 = self.en_5(out_4)
        out_6 = self.en_6(out_5)
        out_7 = self.en_7(out_6)
        out_8 = self.en_8(out_7)

        dout_8 = self.de_8(out_8)
        dout_8_out_7 = torch.cat([dout_8, out_7], 1)
        dout_7 = self.de_7(dout_8_out_7)
        dout_7_out_6 = torch.cat([dout_7, out_6], 1)
        dout_6 = self.de_6(dout_7_out_6)
        dout_6_out_5 = torch.cat([dout_6, out_5], 1)
        dout_5 = self.de_5(dout_6_out_5)
        dout_5_out_4 = torch.cat([dout_5, out_4], 1)
        dout_4 = self.de_4(dout_5_out_4)
        dout_4_out_3 = torch.cat([dout_4, out_3], 1)
        dout_3 = self.de_3(dout_4_out_3)
        dout_3_out_2 = torch.cat([dout_3, out_2], 1)
        dout_2 = self.de_2(dout_3_out_2)
        dout_2_out_1 = torch.cat([dout_2, out_1], 1)
        dout_1 = self.de_1(dout_2_out_1)

        return dout_1


class RefineEncodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 normalization=None, activation=None):
        super().__init__()

        layers = []
        if activation:
            layers.append(get_act(activation))
        layers.append(
            nn.Conv2d(in_channels, in_channels, 4, 2, dilation=2, padding=3))
        if normalization:
            layers.append(get_norm(normalization, out_channels))

        if activation:
            layers.append(get_act(activation))
        layers.append(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=1))
        if normalization:
            layers.append(get_norm(normalization, out_channels))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class RefineDecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 normalization=None, activation=None):
        super().__init__()

        layers = []
        if activation:
            layers.append(get_act(activation))
        layers.append(
            nn.ConvTranspose2d(in_channels, out_channels, 3, 1, padding=1))
        if normalization:
            layers.append(get_norm(normalization, out_channels))

        if activation:
            layers.append(get_act(activation))
        layers.append(
            nn.ConvTranspose2d(out_channels, out_channels, 4, 2, padding=1))
        if normalization:
            layers.append(get_norm(normalization, out_channels))
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class RefineNet(nn.Module):
    def __init__(self, c_img=3,
                 norm='instance', act_en='leaky_relu', act_de='relu'):
        super().__init__()

        c_in = c_img + c_img
        cnum = 64

        self.en_1 = nn.Conv2d(c_in, cnum, 3, 1, padding=1)
        self.en_2 = RefineEncodeBlock(cnum, cnum*2, normalization=norm, activation=act_en)
        self.en_3 = RefineEncodeBlock(cnum*2, cnum*4, normalization=norm, activation=act_en)
        self.en_4 = RefineEncodeBlock(cnum*4, cnum*8, normalization=norm, activation=act_en)
        self.en_5 = RefineEncodeBlock(cnum*8, cnum*8, normalization=norm, activation=act_en)
        self.en_6 = RefineEncodeBlock(cnum*8, cnum*8, normalization=norm, activation=act_en)
        self.en_7 = RefineEncodeBlock(cnum*8, cnum*8, normalization=norm, activation=act_en)
        self.en_8 = RefineEncodeBlock(cnum*8, cnum*8, normalization=norm, activation=act_en)
        self.en_9 = nn.Sequential(
            get_act(act_en),
            nn.Conv2d(cnum*8, cnum*8, 4, 2, padding=1))

        self.de_9 = nn.Sequential(
            get_act(act_de),
            nn.ConvTranspose2d(cnum*8, cnum*8, 4, 2, padding=1),
            get_norm(norm, cnum*8))
        self.de_8 = RefineDecodeBlock(cnum*8*2, cnum*8, normalization=norm, activation=act_de)
        self.de_7 = RefineDecodeBlock(cnum*8*2, cnum*8, normalization=norm, activation=act_de)
        self.de_6 = RefineDecodeBlock(cnum*8*2, cnum*8, normalization=norm, activation=act_de)
        self.de_5 = RefineDecodeBlock(cnum*8*2, cnum*8, normalization=norm, activation=act_de)
        self.de_4 = RefineDecodeBlock(cnum*8*2, cnum*4, normalization=norm, activation=act_de)
        self.de_3 = RefineDecodeBlock(cnum*4*2, cnum*2, normalization=norm, activation=act_de)
        self.de_2 = RefineDecodeBlock(cnum*2*2, cnum, normalization=norm, activation=act_de)

        self.de_1 = nn.Sequential(
            get_act(act_de),
            nn.ConvTranspose2d(cnum*2, c_img, 3, 1, padding=1))


    def forward(self, x1, x2):
        x = torch.cat([x1, x2], 1)
        out_1 = self.en_1(x)
        out_2 = self.en_2(out_1)
        out_3 = self.en_3(out_2)
        out_4 = self.en_4(out_3)
        out_5 = self.en_5(out_4)
        out_6 = self.en_6(out_5)
        out_7 = self.en_7(out_6)
        out_8 = self.en_8(out_7)
        out_9 = self.en_9(out_8)

        dout_9 = self.de_9(out_9)
        dout_9_out_8 = torch.cat([dout_9, out_8], 1)
        dout_8 = self.de_8(dout_9_out_8)
        dout_8_out_7 = torch.cat([dout_8, out_7], 1)
        dout_7 = self.de_7(dout_8_out_7)
        dout_7_out_6 = torch.cat([dout_7, out_6], 1)
        dout_6 = self.de_6(dout_7_out_6)
        dout_6_out_5 = torch.cat([dout_6, out_5], 1)
        dout_5 = self.de_5(dout_6_out_5)
        dout_5_out_4 = torch.cat([dout_5, out_4], 1)
        dout_4 = self.de_4(dout_5_out_4)
        dout_4_out_3 = torch.cat([dout_4, out_3], 1)
        dout_3 = self.de_3(dout_4_out_3)
        dout_3_out_2 = torch.cat([dout_3, out_2], 1)
        dout_2 = self.de_2(dout_3_out_2)
        dout_2_out_1 = torch.cat([dout_2, out_1], 1)
        dout_1 = self.de_1(dout_2_out_1)

        return dout_1, out_4, dout_5


class CSA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        return x


class InpaintNet(nn.Module):
    def __init__(self, c_img=3, norm='instance', act_en='leaky_relu', act_de='relu'):
        super().__init__()

        self.coarse = CoarseNet(c_img=c_img, norm=norm, act_en=act_en, act_de=act_de)
        self.refine = RefineNet(c_img=c_img, norm=norm, act_en=act_en, act_de=act_de)

    def forward(self, image, mask):
        out_c = self.coarse(image)
        out_c = image * (1. - mask) + out_c * mask
        out_r, csa, csa_d = self.refine(out_c, image)
        return out_c, out_r, csa, csa_d
