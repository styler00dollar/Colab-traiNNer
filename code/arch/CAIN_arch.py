import math
import numpy as np

import torch
import torch.nn as nn

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
def sub_mean(x):
    mean = x.mean(2, keepdim=True).mean(3, keepdim=True)
    x -= mean
    return x, mean


class ConvNorm(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, stride=1, norm=False):
        super(ConvNorm, self).__init__()

        reflection_padding = kernel_size // 2
        #self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        # because of tensorrt
        self.reflection_pad = torch.nn.ZeroPad2d(reflection_padding)
        self.conv = nn.Conv2d(in_feat, out_feat, stride=stride, kernel_size=kernel_size, bias=True)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        return out





class meanShift(nn.Module):
    def __init__(self, rgbRange, rgbMean, sign, nChannel=3):
        super(meanShift, self).__init__()
        if nChannel == 1:
            l = rgbMean[0] * rgbRange * float(sign)

            self.shifter =  nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
            self.shifter.weight.data = torch.eye(1).view(1, 1, 1, 1)
            self.shifter.bias.data = torch.Tensor([l])
        elif nChannel == 3:  
            r = rgbMean[0] * rgbRange * float(sign)
            g = rgbMean[1] * rgbRange * float(sign)
            b = rgbMean[2] * rgbRange * float(sign)

            self.shifter =  nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)
            self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
            self.shifter.bias.data = torch.Tensor([r, g, b])
        else:
            r = rgbMean[0] * rgbRange * float(sign)
            g = rgbMean[1] * rgbRange * float(sign)
            b = rgbMean[2] * rgbRange * float(sign)
            self.shifter =  nn.Conv2d(6, 6, kernel_size=1, stride=1, padding=0)
            self.shifter.weight.data = torch.eye(6).view(6, 6, 1, 1)
            self.shifter.bias.data = torch.Tensor([r, g, b, r, g, b])

        # Freeze the meanShift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)

        return x


""" CONV - (BN) - RELU - CONV - (BN) """
class ResBlock(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size=3, reduction=False, bias=False, # 'reduction' is just for placeholder
                 norm=False, act=nn.ReLU(True), downscale=False):
        super(ResBlock, self).__init__()

        self.body = nn.Sequential(
            ConvNorm(in_feat, out_feat, kernel_size=kernel_size, stride=2 if downscale else 1),
            act,
            ConvNorm(out_feat, out_feat, kernel_size=kernel_size, stride=1)
        )
        

    def forward(self, x):
        res = x
        out = self.body(x)
        out += res

        return out 


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, reduction, bias=False,
            norm=False, act=nn.ReLU(True), downscale=False, return_ca=False):
        super(RCAB, self).__init__()

        self.body = nn.Sequential(
            ConvNorm(in_feat, out_feat, kernel_size, stride=1, norm=norm),
            act,
            ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
            CALayer(out_feat, reduction)
        )

    def forward(self, x):
        res = x
        out = self.body(x)
        out += res
        return out


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, Block, n_resblocks, n_feat, kernel_size, reduction, act, norm=False):
        super(ResidualGroup, self).__init__()
        modules_body = [Block(n_feat, n_feat, kernel_size, reduction, bias=False, norm=norm, act=act)
            for _ in range(n_resblocks)]
        modules_body.append(ConvNorm(n_feat, n_feat, kernel_size, stride=1, norm=norm))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class Interpolation(nn.Module):
    def __init__(self, n_resgroups, n_resblocks, n_feats, 
                 reduction=16, act=nn.LeakyReLU(0.2, True), norm=False):
        super(Interpolation, self).__init__()

        self.headConv = nn.Conv2d(n_feats*2, n_feats,stride=1,padding=1,bias=False,groups=1,kernel_size=3)
        modules_body = [
            ResidualGroup(
                RCAB,
                n_resblocks=12,
                n_feat=n_feats,
                kernel_size=3,
                reduction=reduction, 
                act=act, 
                norm=norm)
            for _ in range(2)]
        self.body = nn.Sequential(*modules_body)

        self.tailConv = nn.Conv2d(n_feats, n_feats,stride=1,padding=1,bias=False,groups=1,kernel_size=3)

    def forward(self, x0, x1):
        # Build input tensor
        x = torch.cat([x0, x1], dim=1)
        x = self.headConv(x)

        res = self.body(x)
        res += x

        out = self.tailConv(res)
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels=3, depth=3):
        super(Encoder, self).__init__()
        self.shuffler = torch.nn.PixelUnshuffle(2**depth)
        relu = nn.LeakyReLU(0.2, True)
        self.interpolate = Interpolation(5, 12, in_channels * (4**depth), act=relu)

    def forward(self, x1, x2):
        feats1 = self.shuffler(x1)
        feats2 = self.shuffler(x2)
        feats = self.interpolate(feats1, feats2)
        return feats


class Decoder(nn.Module):
    def __init__(self, depth=3):
        super(Decoder, self).__init__()
        self.shuffler = torch.nn.PixelShuffle(2**depth)

    def forward(self, feats):
        out = self.shuffler(feats)
        return out


class CAIN(nn.Module):
    def __init__(self, depth=3):
        super(CAIN, self).__init__()
        self.encoder = Encoder(in_channels=3, depth=depth)
        self.decoder = Decoder(depth=depth)

    def forward(self, x1, x2):
        x1, m1 = sub_mean(x1)
        x2, m2 = sub_mean(x2)
        out = self.decoder(self.encoder(x1, x2))
        mi = (m1 + m2) / 2
        out += mi
        return out
