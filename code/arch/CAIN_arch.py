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

import yaml
with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

# CONV
if cfg['network_G']['conv'] == 'doconv':
  from .conv.doconv import *

if cfg['network_G']['conv'] == 'gated':
  from .conv.gatedconv import *

if cfg['network_G']['conv'] == 'TBC':
  from .conv.TBC import *

if cfg['network_G']['conv'] == 'dynamic':
  from .conv.dynamicconv import *
  nof_kernels_param = cfg['network_G']['nof_kernels']
  reduce_param = cfg['network_G']['reduce']

if cfg['network_G']['conv'] == 'MBConv':
  from .conv.MBConv import MBConv

if cfg['network_G']['conv'] == 'Involution':
  from .conv.Involution import Involution

if cfg['network_G']['conv'] == 'CondConv':
  from .conv.CondConv import CondConv

if cfg['network_G']['conv'] == 'fft':
  from .lama_arch import FourierUnit

if cfg['network_G']['conv'] == 'WSConv':
  from nfnets import WSConv2d, WSConvTranspose2d, ScaledStdConv2d

# ATTENTION
if cfg['network_G']['attention'] == 'OutlookAttention':
  from .attention.OutlookAttention import OutlookAttention

if cfg['network_G']['attention'] == 'A2Atttention':
  from .attention.A2Atttention import DoubleAttention

if cfg['network_G']['attention'] == 'CBAM':
  from .attention.CBAM import CBAMBlock

if cfg['network_G']['attention'] == 'CoTAttention':
  from .attention.CoTAttention import CoTAttention

if cfg['network_G']['attention'] == 'CoordAttention':
  from .attention.CoordAttention import CoordAtt

if cfg['network_G']['attention'] == 'ECAAttention':
  from .attention.ECAAttention import ECAAttention

if cfg['network_G']['attention'] == 'HaloAttention':
  from .attention.HaloAttention import HaloAttention

if cfg['network_G']['attention'] == 'ParNetAttention':
  from .attention.ParNetAttention import ParNetAttention

if cfg['network_G']['attention'] == 'TripletAttention':
  from .attention.TripletAttention import TripletAttention

if cfg['network_G']['attention'] == 'SKAttention':
  from .attention.SKAttention import SKAttention

if cfg['network_G']['attention'] == 'SGE':
  from .attention.SGE import SpatialGroupEnhance

if cfg['network_G']['attention'] == 'SEAttention':
  from .attention.SEAttention import SEAttention

if cfg['network_G']['attention'] == 'PolarizedSelfAttention':
  from .attention.PolarizedSelfAttention import SequentialPolarizedSelfAttention

# https://github.com/fangwei123456/PixelUnshuffle-pytorch/blob/master/PixelUnshuffle/__init__.py
def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor*downscale_factor, 0, y, x] = 1

    # GatedConv2dWithActivation, doconv, TBC and dynamic does not support kernel as input, using normal conv2d because of this
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)

class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor
    def forward(self, input):
        '''
        input: batchSize * c * k*w * k*h
        kdownscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''

        return pixel_unshuffle(input, self.downscale_factor)


class ConvNorm(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, stride=1, norm=False):
        super(ConvNorm, self).__init__()

        reflection_padding = kernel_size // 2
        #self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        # because of tensorrt
        self.reflection_pad = torch.nn.ZeroPad2d(reflection_padding)

        if cfg['network_G']['conv'] == 'doconv':
          self.conv = DOConv2d(in_feat, out_feat, stride=stride, kernel_size=kernel_size, bias=True)
        elif cfg['network_G']['conv'] == 'gated':
          self.conv = GatedConv2dWithActivation(in_feat, out_feat, stride=stride, kernel_size=kernel_size, bias=True)
        elif cfg['network_G']['conv'] == 'TBC':
          self.conv = TiedBlockConv2d(in_feat, out_feat, stride=stride, kernel_size=kernel_size, bias=True)
        elif cfg['network_G']['conv'] == 'dynamic':
          self.conv = DynamicConvolution(nof_kernels_param, reduce_param, in_channels=in_feat, out_channels=out_feat, stride=stride, kernel_size=kernel_size, bias=True)
        elif cfg['network_G']['conv'] == 'CondConv':
          self.conv = CondConv(in_planes=in_feat,out_planes=out_feat,kernel_size=kernel_size,stride=1,padding=1,bias=False)
        elif cfg['network_G']['conv'] == 'MBConv':
          self.conv =  MBConv(in_feat, out_feat, 1, 1, True)
        elif cfg['network_G']['conv'] == 'fft':
          self.conv = FourierUnit(in_feat, out_feat)
        elif cfg['network_G']['conv'] == 'WSConv':
          self.conv = WSConv2d(in_feat, out_feat, stride=stride, kernel_size=kernel_size, bias=True)
        elif cfg['network_G']['conv'] == 'conv2d' or cfg['network_G']['conv'] == 'Involution':
          self.conv = nn.Conv2d(in_feat, out_feat, stride=stride, kernel_size=kernel_size, bias=True)

    def forward(self, x):
        # do not pad if conv does not need it (because of shape errors)
        if cfg['network_G']['conv'] != 'CondConv' and cfg['network_G']['conv'] != 'MBConv' and cfg['network_G']['conv'] != 'fft':
          x = self.reflection_pad(x)
        out = self.conv(x)
        return out





class meanShift(nn.Module):
    def __init__(self, rgbRange, rgbMean, sign, nChannel=3):
        super(meanShift, self).__init__()
        if nChannel == 1:
            l = rgbMean[0] * rgbRange * float(sign)

            if cfg['network_G']['conv'] == 'doconv':
              self.shifter =  DOConv2d(1, 1, kernel_size=1, stride=1, padding=0)
            elif cfg['network_G']['conv'] == 'gated':
              self.shifter =  GatedConv2dWithActivation(1, 1, kernel_size=1, stride=1, padding=0)   
            elif cfg['network_G']['conv'] == 'TBC':     
              self.shifter =  TiedBlockConv2d(1, 1, kernel_size=1, stride=1, padding=0)   
            elif cfg['network_G']['conv'] == 'dynamic':
              self.shifter =  DynamicConvolution(nof_kernels_param, reduce_param, in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)   
            elif cfg['network_G']['conv'] == 'MBConv':
              self.shifter =  MBConv(1, 1, 1, 2, True)
            elif cfg['network_G']['conv'] == 'Involution':
              self.shifter =  Involution(in_channel=1, kernel_size=1, stride=1)
            elif cfg['network_G']['conv'] == 'CondConv':
              self.shifter = CondConv(in_planes=1,out_planes=1,kernel_size=1,stride=1,padding=0,bias=False)
            elif cfg['network_G']['conv'] == 'fft':
              self.shifter = FourierUnit(in_channels=1, out_channels=1, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho')
            elif cfg['network_G']['conv'] == 'WSConv':
              self.conv = WSConv2d(1, 1, kernel_size=1, stride=1, padding=0)
            elif cfg['network_G']['conv'] == 'conv2d':
              self.shifter =  nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)

            self.shifter.weight.data = torch.eye(1).view(1, 1, 1, 1)
            self.shifter.bias.data = torch.Tensor([l])
        elif nChannel == 3:  
            r = rgbMean[0] * rgbRange * float(sign)
            g = rgbMean[1] * rgbRange * float(sign)
            b = rgbMean[2] * rgbRange * float(sign)

            if cfg['network_G']['conv'] == 'doconv':
              self.shifter = DOConv2d(3, 3, kernel_size=1, stride=1, padding=0)
            elif cfg['network_G']['conv'] == 'gated':
              self.shifter = GatedConv2dWithActivation(3, 3, kernel_size=1, stride=1, padding=0)
            elif cfg['network_G']['conv'] == 'TBC':   
              self.shifter = TiedBlockConv2d(3, 3, kernel_size=1, stride=1, padding=0)
            elif cfg['network_G']['conv'] == 'dynamic':
              self.shifter = DynamicConvolution(nof_kernels_param, reduce_param, in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
            elif cfg['network_G']['conv'] == 'MBConv':
              self.shifter = MBConv(3, 3, 1, 2, True)
            elif cfg['network_G']['conv'] == 'Involution':
              self.shifter = Involution(in_channel=3, kernel_size=1, stride=1)
            elif cfg['network_G']['conv'] == 'CondConv':
              self.shifter = CondConv(in_planes=3,out_planes=3,kernel_size=1,stride=1,padding=0,bias=False)
            elif cfg['network_G']['conv'] == 'fft':
              self.shifter = FourierUnit(in_channels=3, out_channels=3, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho')
            elif cfg['network_G']['conv'] == 'WSConv':
              self.conv = WSConv2d(3, 3, kernel_size=1, stride=1, padding=0)
            elif cfg['network_G']['conv'] == 'conv2d':
              self.shifter = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)

            self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
            self.shifter.bias.data = torch.Tensor([r, g, b])
        else:
            r = rgbMean[0] * rgbRange * float(sign)
            g = rgbMean[1] * rgbRange * float(sign)
            b = rgbMean[2] * rgbRange * float(sign)

            if cfg['network_G']['conv'] == 'doconv':
              self.shifter = DOConv2d(6, 6, kernel_size=1, stride=1, padding=0)
            elif cfg['network_G']['conv'] == 'gated':
              self.shifter = GatedConv2dWithActivation(6, 6, kernel_size=1, stride=1, padding=0)
            elif cfg['network_G']['conv'] == 'TBC':  
              self.shifter = TiedBlockConv2d(6, 6, kernel_size=1, stride=1, padding=0)
            elif cfg['network_G']['conv'] == 'dynamic':
              self.shifter = DynamicConvolution(nof_kernels_param, reduce_param, in_channels=6, out_channels=6, kernel_size=1, stride=1, padding=0)  
            elif cfg['network_G']['conv'] == 'MBConv':
              self.shifter = MBConv(6, 6, 1, 2, True)
            elif cfg['network_G']['conv'] == 'Involution':
              self.shifter = Involution(in_channel=6, kernel_size=1, stride=1)
            elif cfg['network_G']['conv'] == 'CondConv':
              self.shifter = CondConv(in_planes=6,out_planes=6,kernel_size=1,stride=1,padding=0,bias=False)
            elif cfg['network_G']['conv'] == 'fft':
              self.shifter = FourierUnit(in_channels=6, out_channels=6, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho')
            elif cfg['network_G']['conv'] == 'WSConv':
              self.conv = WSConv2d(6, 6, kernel_size=1, stride=1, padding=0)
            elif cfg['network_G']['conv'] == 'conv2d':
              self.shifter = nn.Conv2d(6, 6, kernel_size=1, stride=1, padding=0)

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

        if cfg['network_G']['conv'] == 'doconv':
          self.conv_du = nn.Sequential(
              DOConv2d(channel, channel // reduction, 1, padding=0, bias=False),
              nn.ReLU(inplace=True),
              DOConv2d(channel // reduction, channel, 1, padding=0, bias=False),
              nn.Sigmoid()
          )

        elif cfg['network_G']['conv'] == 'TBC':  
          self.conv_du = nn.Sequential(
              TiedBlockConv2d(channel, channel // reduction, 1, padding=0, bias=False),
              nn.ReLU(inplace=True),
              TiedBlockConv2d(channel // reduction, channel, 1, padding=0, bias=False),
              nn.Sigmoid()
          )

        elif cfg['network_G']['conv'] == 'dynamic':
          self.conv_du = nn.Sequential(
              DynamicConvolution(nof_kernels_param, reduce_param, in_channels=channel, out_channels= (channel // reduction), kernel_size=1, padding=0, bias=False),
              nn.ReLU(inplace=True),
              DynamicConvolution(nof_kernels_param, reduce_param, in_channels=(channel // reduction), out_channels=channel, kernel_size=1, padding=0, bias=False),
              nn.Sigmoid()
          )

        elif cfg['network_G']['conv'] == 'CondConv':
          self.conv_du = nn.Sequential(
              CondConv(in_planes=channel,out_planes=channel // reduction,kernel_size=1,stride=1,padding=0,bias=False),
              nn.ReLU(inplace=True),
              CondConv(in_planes=channel // reduction,out_planes=channel,kernel_size=1,stride=1,padding=0,bias=False),
              nn.Sigmoid()
          )

        elif cfg['network_G']['conv'] == 'WSConv':
          self.conv_du = nn.Sequential(
              WSConv2d(channel, channel // reduction, 1, padding=0, bias=False),
              nn.ReLU(inplace=True),
              WSConv2d(channel // reduction, channel, 1, padding=0, bias=False),
              nn.Sigmoid()
          )
                        
        # shape error if gated, MBConv, Involution or fft is used here
        elif cfg['network_G']['conv'] == 'conv2d' or cfg['network_G']['conv'] == 'gated' or cfg['network_G']['conv'] == 'MBConv' or cfg['network_G']['conv'] == 'Involution' or cfg['network_G']['conv'] == 'fft':
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

        if cfg['network_G']['attention'] == 'CA':
          self.body = nn.Sequential(
              ConvNorm(in_feat, out_feat, kernel_size, stride=1, norm=norm),
              act,
              ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
              CALayer(out_feat, reduction)
          )
        elif cfg['network_G']['attention'] == 'OutlookAttention':
          self.body = nn.Sequential(
              ConvNorm(in_feat, out_feat, kernel_size, stride=1, norm=norm),
              act,
              ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
              OutlookAttention(160)
          )

        elif cfg['network_G']['attention'] == 'A2Atttention':
          self.body = nn.Sequential(
              ConvNorm(in_feat, out_feat, kernel_size, stride=1, norm=norm),
              act,
              ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
              DoubleAttention(out_feat,128,128,True)
          )

        elif cfg['network_G']['attention'] == 'CBAM':
          self.body = nn.Sequential(
              ConvNorm(in_feat, out_feat, kernel_size, stride=1, norm=norm),
              act,
              ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
              CBAMBlock(out_feat,reduction=16,kernel_size=3)
          )

        elif cfg['network_G']['attention'] == 'CoTAttention':
          self.body = nn.Sequential(
              ConvNorm(in_feat, out_feat, kernel_size, stride=1, norm=norm),
              act,
              ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
              CoTAttention(out_feat,kernel_size=3)
          )

        elif cfg['network_G']['attention'] == 'CoordAttention':
          self.body = nn.Sequential(
              ConvNorm(in_feat, out_feat, kernel_size, stride=1, norm=norm),
              act,
              ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
              CoordAtt(out_feat,out_feat)
          )

        elif cfg['network_G']['attention'] == 'ECAAttention':
          self.body = nn.Sequential(
              ConvNorm(in_feat, out_feat, kernel_size, stride=1, norm=norm),
              act,
              ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
              ECAAttention(kernel_size=3)
          )

        elif cfg['network_G']['attention'] == 'HaloAttention':
          self.body = nn.Sequential(
              ConvNorm(in_feat, out_feat, kernel_size, stride=1, norm=norm),
              act,
              ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
              HaloAttention(dim=out_feat,block_size=2,halo_size=1)
          )

        elif cfg['network_G']['attention'] == 'ParNetAttention':
          self.body = nn.Sequential(
              ConvNorm(in_feat, out_feat, kernel_size, stride=1, norm=norm),
              act,
              ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
              ParNetAttention(channel=out_feat)
          )

        elif cfg['network_G']['attention'] == 'TripletAttention':
          self.body = nn.Sequential(
              ConvNorm(in_feat, out_feat, kernel_size, stride=1, norm=norm),
              act,
              ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
              TripletAttention()
          )

        elif cfg['network_G']['attention'] == 'SKAttention':
          self.body = nn.Sequential(
              ConvNorm(in_feat, out_feat, kernel_size, stride=1, norm=norm),
              act,
              ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
              SKAttention(channel=out_feat,reduction=8)
          )

        elif cfg['network_G']['attention'] == 'SGE':
          self.body = nn.Sequential(
              ConvNorm(in_feat, out_feat, kernel_size, stride=1, norm=norm),
              act,
              ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
              SpatialGroupEnhance(groups=8)
          )

        elif cfg['network_G']['attention'] == 'SEAttention':
          self.body = nn.Sequential(
              ConvNorm(in_feat, out_feat, kernel_size, stride=1, norm=norm),
              act,
              ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
              SEAttention(channel=out_feat,reduction=8)
          )

        elif cfg['network_G']['attention'] == 'PolarizedSelfAttention':
          self.body = nn.Sequential(
              ConvNorm(in_feat, out_feat, kernel_size, stride=1, norm=norm),
              act,
              ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
              SequentialPolarizedSelfAttention(channel=out_feat)
          )

        elif cfg['network_G']['attention'] == 'S2Attention':
          self.body = nn.Sequential(
              ConvNorm(in_feat, out_feat, kernel_size, stride=1, norm=norm),
              act,
              ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
              S2Attention(out_feat)
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

        if cfg['network_G']['conv'] == 'doconv':
          self.headConv = DOConv2d(n_feats*2, n_feats,stride=1,padding=1,bias=False,groups=1,kernel_size=3)
        elif cfg['network_G']['conv'] == 'gated':
          self.headConv = GatedConv2dWithActivation(n_feats*2, n_feats,stride=1,padding=1,bias=False,groups=1,kernel_size=3)
        elif cfg['network_G']['conv'] == 'TBC':  
          self.headConv = TiedBlockConv2d(n_feats*2, n_feats,stride=1,padding=1,bias=False,groups=1,kernel_size=3)
        elif cfg['network_G']['conv'] == 'dynamic':
          self.headConv = DynamicConvolution(nof_kernels_param, reduce_param, in_channels=n_feats*2, out_channels=n_feats,stride=1,padding=1,bias=False,groups=1,kernel_size=3)
        elif cfg['network_G']['conv'] == 'MBConv':
          self.headConv = MBConv(n_feats*2, n_feats,1,1,True)
        elif cfg['network_G']['conv'] == 'fft':
          self.headConv = FourierUnit(in_channels=n_feats*2, out_channels=n_feats, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
            spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho')
        elif cfg['network_G']['conv'] == 'WSConv':
          self.headConv = WSConv2d(n_feats*2, n_feats,stride=1,padding=1,bias=False,groups=1,kernel_size=3)
        # Involution does have fixed in/output dimension, CondConv results in shape error
        elif cfg['network_G']['conv'] == 'conv2d' or cfg['network_G']['conv'] == 'Involution' or cfg['network_G']['conv'] == 'CondConv':
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
            for _ in range(cfg['network_G']['RG'])]
        self.body = nn.Sequential(*modules_body)

        if cfg['network_G']['conv'] == 'doconv':
          self.tailConv = DOConv2d(n_feats, n_feats,stride=1,padding=1,bias=False,groups=1,kernel_size=3)
        elif cfg['network_G']['conv'] == 'gated':
          self.tailConv = GatedConv2dWithActivation(n_feats, n_feats,stride=1,padding=1,bias=False,groups=1,kernel_size=3) 
        elif cfg['network_G']['conv'] == 'TBC':  
          self.tailConv = TiedBlockConv2d(n_feats, n_feats,stride=1,padding=1,bias=False,groups=1,kernel_size=3) 
        elif cfg['network_G']['conv'] == 'dynamic':
          self.tailConv = DynamicConvolution(nof_kernels_param, reduce_param, in_channels=n_feats, out_channels=n_feats,stride=1,padding=1,bias=False,groups=1,kernel_size=3) 
        elif cfg['network_G']['conv'] == 'MBConv':
          self.tailConv = MBConv(n_feats, n_feats,1,1,True)
        elif cfg['network_G']['conv'] == 'Involution':
          self.tailConv =  Involution(in_channel=n_feats, kernel_size=3, stride=1)
        elif cfg['network_G']['conv'] == 'CondConv':
          self.tailConv = CondConv(in_planes=n_feats,out_planes=n_feats,kernel_size=1,stride=1,padding=0,bias=False)
        elif cfg['network_G']['conv'] == 'fft':
          self.tailConv = FourierUnit(in_channels=n_feats, out_channels=n_feats, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
            spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho')
        elif cfg['network_G']['conv'] == 'WSConv':
          self.tailConv = WSConv2d(n_feats, n_feats,stride=1,padding=1,bias=False,groups=1,kernel_size=3)
        elif cfg['network_G']['conv'] == 'conv2d':
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
        #self.shuffler = torch.nn.PixelUnshuffle(2**depth)
        # custom unshuffle
        self.shuffler = PixelUnshuffle(2**depth)
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
