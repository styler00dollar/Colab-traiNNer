"""
inpainting_network.py (19-12-20)
https://github.com/wangyx240/High-Resolution-Image-Inpainting-GAN/blob/ce917a3213216ec4b3b0e8d859b8713d5f67fedc/inpainting_network.py

network_module.py (19-12-20)
https://github.com/wangyx240/High-Resolution-Image-Inpainting-GAN/blob/ce917a3213216ec4b3b0e8d859b8713d5f67fedc/network_module.py
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter

#-----------------------------------------------
#                Normal ConvBlock
#-----------------------------------------------
class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'replicate', activation = 'none', norm = 'none', sn = False):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'elu':
            self.activation = nn.ELU(alpha=1.0,inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class TransposeConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False, scale_factor = 2):
        super(TransposeConv2dLayer, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.conv2d = Conv2dLayer(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)

    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'nearest')
        x = self.conv2d(x)
        return x


class depth_separable_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation):
        super(depth_separable_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class sc_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation):
        super(sc_conv, self).__init__()
        self.single_channel_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1
        )

    def forward(self, input):
        out = self.single_channel_conv(input)
        return out


#-----------------------------------------------
#                Gated ConvBlock
#-----------------------------------------------
class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'replicate', activation = 'elu', norm = 'none', sc=False, sn=False):
        super(GatedConv2d, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'elu':
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sc:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
            self.mask_conv2d = sc_conv(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
            #self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
            self.mask_conv2d = depth_separable_conv(in_channels, out_channels, kernel_size, stride, padding = 0,dilation=dilation)

        self.sigmoid = torch.nn.Sigmoid()
        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_channels != out_channels:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels,
        #                   kernel_size=1, stride=stride, bias=False),
        #         self.norm
        #     )

    def forward(self, x_in):
        x = self.pad(x_in)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        if self.norm:
            conv = self.norm(conv)
        if self.activation:
            conv = self.activation(conv)
        gated_mask = self.sigmoid(mask)
        x = conv * gated_mask

        #x += self.shortcut(x_in)


        return x

class TransposeGatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sc = False, scale_factor = 2):
        super(TransposeGatedConv2d, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.gated_conv2d = GatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sc)

    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'nearest')
        x = self.gated_conv2d(x)
        return x

# ----------------------------------------
#               Layer Norm
# ----------------------------------------
# class LayerNorm(nn.Module):
#     def __init__(self, num_features, eps = 1e-8, affine = True):
#         super(LayerNorm, self).__init__()
#         self.num_features = num_features
#         self.affine = affine
#         self.eps = eps
#
#         if self.affine:
#             self.gamma = Parameter(torch.Tensor(num_features).uniform_())
#             self.beta = Parameter(torch.zeros(num_features))
#
#     def forward(self, x):
#         # layer norm
#         shape = [-1] + [1] * (x.dim() - 1)                                  # for 4d input: [-1, 1, 1, 1]
#         if x.size(0) == 1:
#             # These two lines run much faster in pytorch 0.4 than the two lines listed below.
#             mean = x.view(-1).mean().view(*shape)
#             std = x.view(-1).std().view(*shape)
#         else:
#             mean = x.view(x.size(0), -1).mean(1).view(*shape)
#             std = x.view(x.size(0), -1).std(1).view(*shape)
#         x = (x - mean) / (std + self.eps)
#         # if it is learnable
#         if self.affine:
#             shape = [1, -1] + [1] * (x.dim() - 2)                          # for 4d input: [1, -1, 1, 1]
#             x = x * self.gamma.view(*shape) + self.beta.view(*shape)
#         return x
#
#-----------------------------------------------
#                  SpectralNorm
#-----------------------------------------------
def l2normalize(v, eps = 1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torchsummary import summary
from torch.nn import functional as F

#from network_module import *

# -----------------------------------------------
#                   Generator
# -----------------------------------------------
# Input: masked image + mask
# Output: filled image

class Coarse(nn.Module):
    def __init__(self, activation='elu', norm='none'):
        super(Coarse, self).__init__()
        # Initialize the padding scheme
        self.coarse1 = nn.Sequential(
            # encoder
            GatedConv2d(4, 32, 5, 2, 2, activation=activation, norm=norm, sc=True),
            GatedConv2d(32, 32, 3, 1, 1, activation=activation, norm=norm, sc=True),
            GatedConv2d(32, 64, 3, 2, 1, activation=activation, norm=norm, sc=True)
        )
        self.coarse2 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 1, activation=activation, norm=norm, sc=True),
            GatedConv2d(64, 64, 3, 1, 1, activation=activation, norm=norm, sc=True),
            GatedConv2d(64, 64, 3, 1, 1, activation=activation, norm=norm, sc=True)
        )
        self.coarse3 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 1, activation=activation, norm=norm, sc=True),
            GatedConv2d(64, 64, 3, 1, 1, activation=activation, norm=norm, sc=True),
            GatedConv2d(64, 64, 3, 1, 1, activation=activation, norm=norm, sc=True)
        )
        self.coarse4 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 2, dilation=2, activation=activation, norm=norm, sc=True),
            GatedConv2d(64, 64, 3, 1, 2, dilation=2, activation=activation, norm=norm, sc=True),
            GatedConv2d(64, 64, 3, 1, 2, dilation=2, activation=activation, norm=norm, sc=True)
        )
        self.coarse5 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 4, dilation=4, activation=activation, norm=norm, sc=True),
            GatedConv2d(64, 64, 3, 1, 4, dilation=4, activation=activation, norm=norm, sc=True),
            GatedConv2d(64, 64, 3, 1, 4, dilation=4, activation=activation, norm=norm, sc=True)
        )
        self.coarse6 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 8, dilation=8, activation=activation, norm=norm, sc=True),
            GatedConv2d(64, 64, 3, 1, 8, dilation=8, activation=activation, norm=norm, sc=True),
            GatedConv2d(64, 64, 3, 1, 8, dilation=8, activation=activation, norm=norm, sc=True)
        )
        self.coarse7 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 16, dilation=16, activation=activation, norm=norm, sc=True),
            GatedConv2d(64, 64, 3, 1, 16, dilation=16, activation=activation, norm=norm, sc=True)
        )
        self.coarse8 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 1, activation=activation, norm=norm, sc=True),
            GatedConv2d(64, 64, 3, 1, 1, activation=activation, norm=norm, sc=True),
            GatedConv2d(64, 64, 3, 1, 1, activation=activation, norm=norm, sc=True),
        )
        # decoder
        self.coarse9 = nn.Sequential(
            TransposeGatedConv2d(64, 64, 3, 1, 1, activation=activation, norm=norm, sc=True),
            TransposeGatedConv2d(64, 32, 3, 1, 1, activation=activation, norm=norm, sc=True),
            GatedConv2d(32, 3, 3, 1, 1, activation='none', norm=norm, sc=True),
            nn.Tanh()
        )

    def forward(self, first_in):
        first_out = self.coarse1(first_in)
        first_out = self.coarse2(first_out) + first_out
        first_out = self.coarse3(first_out) + first_out
        first_out = self.coarse4(first_out) + first_out
        first_out = self.coarse5(first_out) + first_out
        first_out = self.coarse6(first_out) + first_out
        first_out = self.coarse7(first_out) + first_out
        first_out = self.coarse8(first_out) + first_out
        first_out = self.coarse9(first_out)
        first_out = torch.clamp(first_out, 0, 1)
        return first_out

class GatedGenerator(nn.Module):
    def __init__(self, activation='elu', norm='none'):
        super(GatedGenerator, self).__init__()

        ########################################## Coarse Network ##################################################
        self.coarse = Coarse()

        ########################################## Refinement Network #########################################################
        self.refinement1 = nn.Sequential(
            GatedConv2d(3, 32, 5, 2, 2, activation=activation, norm=norm),                  #[B,32,256,256]
            GatedConv2d(32, 32, 3, 1, 1, activation=activation, norm=norm),
        )
        self.refinement2 = nn.Sequential(
            # encoder
            GatedConv2d(32, 64, 3, 2, 1, activation=activation, norm=norm),
            GatedConv2d(64, 64, 3, 1, 1, activation=activation, norm=norm)
        )
        self.refinement3 = nn.Sequential(
            GatedConv2d(64, 128, 3, 2, 1, activation=activation, norm=norm)
        )
        self.refinement4 = nn.Sequential(
            GatedConv2d(128, 128, 3, 1, 1, activation=activation, norm=norm),
            GatedConv2d(128, 128, 3, 1, 1, activation=activation, norm=norm),
        )
        self.refinement5 = nn.Sequential(
            GatedConv2d(128, 128, 3, 1, 2, dilation = 2, activation=activation, norm=norm),
            GatedConv2d(128, 128, 3, 1, 4, dilation = 4, activation=activation, norm=norm)
        )
        self.refinement6 = nn.Sequential(
            GatedConv2d(128, 128, 3, 1, 8, dilation = 8, activation=activation, norm=norm),
            GatedConv2d(128, 128, 3, 1, 16, dilation = 16, activation=activation, norm=norm),
        )
        self.refinement7 = nn.Sequential(
            GatedConv2d(256, 128, 3, 1, 1, activation=activation, norm=norm),
            TransposeGatedConv2d(128, 64, 3, 1, 1, activation=activation, norm=norm),
            GatedConv2d(64, 64, 3, 1, 1, activation=activation, norm=norm)
        )
        self.refinement8 = nn.Sequential(
            TransposeGatedConv2d(128, 64, 3, 1, 1, activation=activation, norm=norm),
            GatedConv2d(64, 32, 3, 1, 1, activation=activation, norm=norm)
        )
        self.refinement9 = nn.Sequential(
            TransposeGatedConv2d(64, 32, 3, 1, 1, activation=activation, norm=norm),
            GatedConv2d(32, 3, 3, 1, 1, activation='none', norm=norm),
            nn.Tanh()
        )
        self.conv_pl3 = nn.Sequential(
            GatedConv2d(128, 128, 3, 1, 1, activation=activation, norm=norm)
        )
        self.conv_pl2 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 1, activation=activation, norm=norm),
            GatedConv2d(64, 64, 3, 1, 2, dilation=2, activation=activation, norm=norm)
        )
        self.conv_pl1 = nn.Sequential(
            GatedConv2d(32, 32, 3, 1, 1, activation=activation, norm=norm),
            GatedConv2d(32, 32, 3, 1, 2, dilation=2, activation=activation, norm=norm)

        )

    def forward(self, img, mask):
        img_256 = F.interpolate(img, size=[256, 256], mode='bilinear')
        mask_256 = F.interpolate(mask, size=[256, 256], mode='nearest')
        first_masked_img = img_256 * (1 - mask_256) + mask_256
        first_in = torch.cat((first_masked_img, mask_256), 1)  # in: [B, 4, H, W]
        first_out = self.coarse(first_in)  # out: [B, 3, H, W]
        first_out = F.interpolate(first_out, size=[512,512], mode='bilinear')

        # Refinement
        second_in = img * (1 - mask) + first_out * mask

        pl1 = self.refinement1(second_in)                   #out: [B, 32, 256, 256]
        pl2 = self.refinement2(pl1)                         #out: [B, 64, 128, 128]
        second_out = self.refinement3(pl2)                  #out: [B, 128, 64, 64]
        second_out = self.refinement4(second_out) + second_out                #out: [B, 128, 64, 64]
        second_out = self.refinement5(second_out) + second_out
        pl3 = self.refinement6(second_out) +second_out           #out: [B, 128, 64, 64]
        #Calculate Attention
        patch_fb = self.cal_patch(32, mask, 512)
        att = self.compute_attention(pl3, patch_fb)

        second_out = torch.cat((pl3, self.conv_pl3(self.attention_transfer(pl3, att))), 1) #out: [B, 256, 64, 64]
        second_out = self.refinement7(second_out)                                                 #out: [B, 64, 128, 128]
        second_out = torch.cat((second_out, self.conv_pl2(self.attention_transfer(pl2, att))), 1) #out: [B, 128, 128, 128]
        second_out = self.refinement8(second_out)                                                 #out: [B, 32, 256, 256]
        second_out = torch.cat((second_out, self.conv_pl1(self.attention_transfer(pl1, att))), 1) #out: [B, 64, 256, 256]
        second_out = self.refinement9(second_out) # out: [B, 3, H, W]
        second_out = torch.clamp(second_out, 0, 1)
        return first_out, second_out

    def cal_patch(self, patch_num, mask, raw_size):
        pool = nn.MaxPool2d(raw_size // patch_num)  # patch_num=32
        patch_fb = pool(mask)  # out: [B, 1, 32, 32]
        return patch_fb

    def compute_attention(self, feature, patch_fb):  # in: [B, C:128, 64, 64]
        b = feature.shape[0]
        feature = F.interpolate(feature, scale_factor=0.5, mode='bilinear')  # in: [B, C:128, 32, 32]
        p_fb = torch.reshape(patch_fb, [b, 32 * 32, 1])
        p_matrix = torch.bmm(p_fb, (1 - p_fb).permute([0, 2, 1]))
        f = feature.permute([0, 2, 3, 1]).reshape([b, 32 * 32, 128])
        c = self.cosine_Matrix(f, f) * p_matrix
        s = F.softmax(c, dim=2) * p_matrix
        return s

    def attention_transfer(self, feature, attention):  # feature: [B, C, H, W]
        b_num, c, h, w = feature.shape
        f = self.extract_image_patches(feature, 32)
        f = torch.reshape(f, [b_num, f.shape[1] * f.shape[2], -1])
        f = torch.bmm(attention, f)
        f = torch.reshape(f, [b_num, 32, 32, h // 32, w // 32, c])
        f = f.permute([0, 5, 1, 3, 2, 4])
        f = torch.reshape(f, [b_num, c, h, w])
        return f

    def extract_image_patches(self, img, patch_num):
        b, c, h, w = img.shape
        img = torch.reshape(img, [b, c, patch_num, h//patch_num, patch_num, w//patch_num])
        img = img.permute([0, 2, 4, 3, 5, 1])
        return img

    def cosine_Matrix(self, _matrixA, _matrixB):
        _matrixA_matrixB = torch.bmm(_matrixA, _matrixB.permute([0, 2, 1]))
        _matrixA_norm = torch.sqrt((_matrixA * _matrixA).sum(axis=2)).unsqueeze(dim=2)
        _matrixB_norm = torch.sqrt((_matrixB * _matrixB).sum(axis=2)).unsqueeze(dim=2)
        return _matrixA_matrixB / torch.bmm(_matrixA_norm, _matrixB_norm.permute([0, 2, 1]))
