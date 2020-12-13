"""
networks.py (13-12-20)
https://github.com/geekyutao/RN/blob/a3cf1fccc08f22fcf4b336503a8853748720fd67/networks.py

rn.py (13-12-20)
https://github.com/geekyutao/RN/blob/a3cf1fccc08f22fcf4b336503a8853748720fd67/rn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RN_binarylabel(nn.Module):
    def __init__(self, feature_channels):
        super(RN_binarylabel, self).__init__()
        self.bn_norm = nn.BatchNorm2d(feature_channels, affine=False, track_running_stats=False)

    def forward(self, x, label):
        '''
        input:  x: (B,C,M,N), features
                label: (B,1,M,N), 1 for foreground regions, 0 for background regions
        output: _x: (B,C,M,N)
        '''
        label = label.detach()

        rn_foreground_region = self.rn(x * label, label)

        rn_background_region = self.rn(x * (1 - label), 1 - label)

        return rn_foreground_region + rn_background_region

    def rn(self, region, mask):
        '''
        input:  region: (B,C,M,N), 0 for surroundings
                mask: (B,1,M,N), 1 for target region, 0 for surroundings
        output: rn_region: (B,C,M,N)
        '''
        shape = region.size()

        sum = torch.sum(region, dim=[0,2,3])  # (B, C) -> (C)
        Sr = torch.sum(mask, dim=[0,2,3])    # (B, 1) -> (1)
        Sr[Sr==0] = 1
        mu = (sum / Sr)     # (B, C) -> (C)

        return self.bn_norm(region + (1 - mask) * mu[None,:,None,None]) * \
        (torch.sqrt(Sr / (shape[0] * shape[2] * shape[3])))[None,:,None,None]

class RN_B(nn.Module):
    def __init__(self, feature_channels):
        super(RN_B, self).__init__()
        '''
        input: tensor(features) x: (B,C,M,N)
               condition Mask: (B,1,H,W): 0 for background, 1 for foreground
        return: tensor RN_B(x): (N,C,M,N)
        ---------------------------------------
        args:
            feature_channels: C
        '''
        # RN
        self.rn = RN_binarylabel(feature_channels)    # need no external parameters

        # gamma and beta
        self.foreground_gamma = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)
        self.foreground_beta = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)
        self.background_gamma = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)
        self.background_beta = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)

    def forward(self, x, mask):
        # mask = F.adaptive_max_pool2d(mask, output_size=x.size()[2:])
        mask = F.interpolate(mask, size=x.size()[2:], mode='nearest')   # after down-sampling, there can be all-zero mask

        rn_x = self.rn(x, mask)

        rn_x_foreground = (rn_x * mask) * (1 + self.foreground_gamma[None,:,None,None]) + self.foreground_beta[None,:,None,None]
        rn_x_background = (rn_x * (1 - mask)) * (1 + self.background_gamma[None,:,None,None]) + self.background_beta[None,:,None,None]

        return rn_x_foreground + rn_x_background

class SelfAware_Affine(nn.Module):
    def __init__(self, kernel_size=7):
        super(SelfAware_Affine, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.gamma_conv = nn.Conv2d(1, 1, kernel_size, padding=padding)
        self.beta_conv = nn.Conv2d(1, 1, kernel_size, padding=padding)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)

        x = self.conv1(x)
        importance_map = self.sigmoid(x)

        gamma = self.gamma_conv(importance_map)
        beta = self.beta_conv(importance_map)

        return importance_map, gamma, beta

class RN_L(nn.Module):
    def __init__(self, feature_channels, threshold=0.8):
        super(RN_L, self).__init__()
        '''
        input: tensor(features) x: (B,C,M,N)
        return: tensor RN_L(x): (B,C,M,N)
        ---------------------------------------
        args:
            feature_channels: C
        '''
        # SelfAware_Affine
        self.sa = SelfAware_Affine()
        self.threshold = threshold

        # RN
        self.rn = RN_binarylabel(feature_channels)    # need no external parameters


    def forward(self, x):

        sa_map, gamma, beta = self.sa(x)     # (B,1,M,N)

        # m = sa_map.detach()
        if x.is_cuda:
            mask = torch.zeros_like(sa_map).cuda()
        else:
            mask = torch.zeros_like(sa_map)
        mask[sa_map.detach() >= self.threshold] = 1

        rn_x = self.rn(x, mask.expand(x.size()))

        rn_x = rn_x * (1 + gamma) + beta

        return rn_x




import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import *
import torch.nn.functional as F

#from rn import RN_B, RN_L


class G_Net(nn.Module):
    def __init__(self, input_channels=3, residual_blocks=8, threshold=0.8):
        super(G_Net, self).__init__()

        # Encoder
        self.encoder_prePad = nn.ReflectionPad2d(3)
        self.encoder_conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=7, padding=0)
        self.encoder_in1 = RN_B(feature_channels=64)
        self.encoder_relu1 = nn.ReLU(True)
        self.encoder_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.encoder_in2 = RN_B(feature_channels=128)
        self.encoder_relu2 = nn.ReLU(True)
        self.encoder_conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.encoder_in3 = RN_B(feature_channels=256)
        self.encoder_relu3 = nn.ReLU(True)


        # Middle
        blocks = []
        for _ in range(residual_blocks):
            # block = ResnetBlock(256, 2, use_spectral_norm=False)
            block = saRN_ResnetBlock(256, dilation=2, threshold=threshold, use_spectral_norm=False)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)


        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128*4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            RN_L(128),
            nn.ReLU(True),

            nn.Conv2d(128, 64*4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            RN_L(64),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=input_channels, kernel_size=7, padding=0)

        )


    def encoder(self, x, mask):
        x = self.encoder_prePad(x)

        x = self.encoder_conv1(x)
        x = self.encoder_in1(x, mask)
        x = self.encoder_relu2(x)

        x = self.encoder_conv2(x)
        x = self.encoder_in2(x, mask)
        x = self.encoder_relu2(x)

        x = self.encoder_conv3(x)
        x = self.encoder_in3(x, mask)
        x = self.encoder_relu3(x)
        return x

    def forward(self, x, mask):
        #gt = x
        #x = (x * (1 - mask).float()) + mask
        # input mask: 1 for hole, 0 for valid
        x = self.encoder(x, mask)

        x = self.middle(x)

        x = self.decoder(x)

        x = (torch.tanh(x) + 1) / 2
        # x = x*mask+gt*(1-mask)
        return x


# original D
class D_Net(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True):
        super(D_Net, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]





class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=256, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out

class saRN_ResnetBlock(nn.Module):
    def __init__(self, dim, dilation, threshold, use_spectral_norm=True):
        super(saRN_ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=256, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            RN_L(feature_channels=256, threshold=threshold),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(dim, track_running_stats=False),
            RN_L(feature_channels=dim, threshold=threshold),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        # skimage.io.imsave('block.png', out[0].detach().permute(1,2,0).cpu().numpy()[:,:,0])

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
