"""
network.py (13-12-20)
https://github.com/styler00dollar/Colab-Global-and-Local-Inpainting/blob/master/model/network.py

Attention.py (13-12-20)
https://github.com/styler00dollar/Colab-Global-and-Local-Inpainting/blob/master/model/Attention.py

tools.py (13-12-20)
https://github.com/styler00dollar/Colab-Global-and-Local-Inpainting/blob/master/utils/tools.py
"""
import os
import torch
import yaml
import numpy as np
#from PIL import Image

import torch.nn.functional as F
import cv2

# Change the values of tensor x from range [0, 1] to [-1, 1]
def normalize(x):
    return x.mul_(2).add_(-1)

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


def local_patch(x, bbox_list):
    assert len(x.size()) == 4
    patches = []
    for i, bbox in enumerate(bbox_list):
        t, l, h, w = bbox
        patches.append(x[i, :, t:t + h, l:l + w])
    return torch.stack(patches, dim=0)


def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x

def deprocess(img):
    img = img.add_(1).div_(2)
    return img


# get configs
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


import torch
import torch.nn as nn
import torch.nn.functional as F
#from utils.tools import *

# Contextual attention implementation is borrowed from IJCAI 2019 : "MUSICAL: Multi-Scale Image Contextual Attention Learning for Inpainting".
# Original implementation causes bad results for Pytorch 1.2+.
class GlobalLocalAttention(nn.Module):
    def __init__(self, in_dim, patch_size=3, propagate_size=3, stride=1):
        super(GlobalLocalAttention, self).__init__()
        self.patch_size = patch_size
        self.propagate_size = propagate_size
        self.stride = stride
        self.prop_kernels = None
        self.in_dim = in_dim
        self.feature_attention = GlobalAttention(in_dim)
        self.patch_attention = GlobalAttentionPatch(in_dim)

    def forward(self, foreground, mask, background="same"):
        ###assume the masked area has value 1
        bz, nc, w, h = foreground.size()
        if background == "same":
            background = foreground.clone()
        mask = F.interpolate(mask, size=(h, w), mode='nearest')
        background = background * (1 - mask)
        foreground = self.feature_attention(foreground, background, mask)
        background = F.pad(background,
                           [self.patch_size // 2, self.patch_size // 2, self.patch_size // 2, self.patch_size // 2])
        conv_kernels_all = background.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size,
                                                                                     self.stride).contiguous().view(bz,
                                                                                                                    nc,
                                                                                                                    -1,
                                                                                                                    self.patch_size,
                                                                                                                    self.patch_size)

        mask_resized = mask.repeat(1, self.in_dim, 1, 1)
        mask_resized = F.pad(mask_resized,
                             [self.patch_size // 2, self.patch_size // 2, self.patch_size // 2, self.patch_size // 2])
        mask_kernels_all = mask_resized.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size,
                                                                                       self.stride).contiguous().view(
            bz,
            nc,
            -1,
            self.patch_size,
            self.patch_size)
        conv_kernels_all = conv_kernels_all.transpose(2, 1)
        mask_kernels_all = mask_kernels_all.transpose(2, 1)
        output_tensor = []
        for i in range(bz):
            feature_map = foreground[i:i + 1]

            # form convolutional kernels
            conv_kernels = conv_kernels_all[i] + 0.0000001
            mask_kernels = mask_kernels_all[i]
            conv_kernels = self.patch_attention(conv_kernels, conv_kernels, mask_kernels)
            norm_factor = torch.sum(conv_kernels ** 2, [1, 2, 3], keepdim=True) ** 0.5
            conv_kernels = conv_kernels / norm_factor

            conv_result = F.conv2d(feature_map, conv_kernels, padding=self.patch_size // 2)
            # print(conv_result.shape)
            if self.propagate_size != 1:
                if self.prop_kernels is None:
                    self.prop_kernels = torch.ones([conv_result.size(1), 1, self.propagate_size, self.propagate_size])
                    self.prop_kernels.requires_grad = False
                    self.prop_kernels = self.prop_kernels.cuda()
                conv_result = F.conv2d(conv_result, self.prop_kernels, stride=1, padding=1, groups=conv_result.size(1))
            mm = (torch.mean(mask_kernels_all[i], dim=[1,2,3], keepdim=True)==0.0).to(torch.float32)
            mm = mm.permute(1,0,2,3).cuda()
            conv_result = conv_result * mm
            attention_scores = F.softmax(conv_result, dim=1)
            attention_scores = attention_scores * mm

            ##propagate the scores
            recovered_foreground = F.conv_transpose2d(attention_scores, conv_kernels, stride=1,
                                                      padding=self.patch_size // 2)
            output_tensor.append(recovered_foreground)
        return torch.cat(output_tensor, dim=0)


class GlobalAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(GlobalAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)  #
        self.rate = 1
        self.gamma = torch.tensor([1.0], requires_grad=True).cuda()

    def forward(self, a, b, c):
        m_batchsize, C, width, height = a.size()  # B, C, H, W
        down_rate = int(c.size(2)//width)
        c = F.interpolate(c, scale_factor=1./down_rate*self.rate, mode='nearest')
        proj_query = self.query_conv(a).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B, C, N -> B N C
        proj_key = self.key_conv(b).view(m_batchsize, -1, width * height)  # B, C, N
        feature_similarity = torch.bmm(proj_query, proj_key)  # B, N, N

        mask = c.view(m_batchsize, -1, width * height)  # B, C, N
        mask = mask.repeat(1, height * width, 1).permute(0, 2, 1)  # B, 1, H, W -> B, C, H, W // B

        feature_pruning = feature_similarity * mask
        attention = self.softmax(feature_pruning)  # B, N, C
        feature_pruning = torch.bmm(self.value_conv(a).view(m_batchsize, -1, width * height),
                                    attention.permute(0, 2, 1))  # -. B, C, N
        out = feature_pruning.view(m_batchsize, C, width, height)  # B, C, H, W
        out = a * c + self.gamma *  (1.0 - c) * out
        return out


class GlobalAttentionPatch(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(GlobalAttentionPatch, self).__init__()
        self.chanel_in = in_dim

        self.query_channel = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_channel = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_channel = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.softmax_channel = nn.Softmax(dim=-1)
        self.gamma = torch.tensor([1.0], requires_grad=True).cuda()

    def forward(self, x, y, m):
        '''
        Something
        '''
        feature_size = list(x.size())
        # Channel attention
        query_channel = self.query_channel(x).view(feature_size[0], -1, feature_size[2] * feature_size[3])
        key_channel = self.key_channel(y).view(feature_size[0], -1, feature_size[2] * feature_size[3]).permute(0,
                                                                                                               2,
                                                                                                               1)
        channel_correlation = torch.bmm(query_channel, key_channel)
        m_r = m.view(feature_size[0], -1, feature_size[2]*feature_size[3])
        channel_correlation = torch.bmm(channel_correlation, m_r)
        energy_channel = self.softmax_channel(channel_correlation)
        value_channel = self.value_channel(x).view(feature_size[0], -1, feature_size[2] * feature_size[3])
        attented_channel = (energy_channel * value_channel).view(feature_size[0], feature_size[1],
                                                                         feature_size[2],
                                                                         feature_size[3])
        out = x * m + self.gamma * (1.0 - m) * attented_channel
        return out


if __name__ == '__main__':
    x = torch.rand(3, 128, 64, 64, requires_grad=True).float().cuda()
    y = torch.rand(3, 1, 256, 256, requires_grad=False).float().cuda()
    y[y > 0.5] = 1
    y[y <= 0.5] = 0
    net = GlobalLocalAttention(128).cuda()
    out = net(x, y)
    print(out.shape)



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn
#from model.Attention import GlobalLocalAttention, GlobalAttention


class Generator(nn.Module):
    def __init__(self, input_dim=5, ngf=32, use_cuda=True, device_ids=[0]):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.cnum = ngf
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.coarse_generator = CoarseGenerator(self.input_dim, self.cnum, self.use_cuda, self.device_ids)
        self.fine_generator = FineGenerator(self.input_dim, self.cnum, self.use_cuda, self.device_ids)

    def forward(self, x, mask):
        x_stage1 = self.coarse_generator(x, mask)
        x_stage2 = self.fine_generator(x, x_stage1, mask)
        #return x_stage1, x_stage2
        return x_stage2


class CoarseGenerator(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(CoarseGenerator, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        # 3 x 256 x 256
        self.conv1 = gen_conv(input_dim, cnum, 5, 1, 2)
        self.conv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        # cnum*2 x 128 x 128
        self.conv3 = gen_conv(cnum, cnum * 2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cnum * 2, cnum * 2, 3, 2, 1)
        # cnum*4 x 64 x 64
        self.conv5 = gen_conv(cnum * 2, cnum * 4, 3, 1, 1)
        self.conv6 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.conv7_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 16, rate=16)
        # attention branch
        # 3 x 256 x 256
        self.pmconv1 = gen_conv(input_dim, cnum, 5, 1, 2)
        self.pmconv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        # cnum*2 x 128 x 128
        self.pmconv3 = gen_conv(cnum, cnum * 2, 3, 1, 1)
        self.pmconv4_downsample = gen_conv(cnum * 2, cnum * 4, 3, 2, 1)
        # cnum*4 x 64 x 64
        self.pmconv5 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.pmconv6 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1, activation='relu')
        self.contextul_attention = GlobalAttention(in_dim=128)
        self.pmconv9 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.pmconv10 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.allconv11 = gen_conv(cnum * 8, cnum * 4, 3, 1, 1)
        self.allconv12 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.allconv13 = gen_conv(cnum * 4, cnum * 2, 3, 1, 1)
        self.allconv14 = gen_conv(cnum * 2, cnum * 2, 3, 1, 1)
        self.allconv15 = gen_conv(cnum * 2, cnum, 3, 1, 1)
        self.allconv16 = gen_conv(cnum, cnum // 2, 3, 1, 1)
        self.allconv17 = gen_conv(cnum // 2, 3, 3, 1, 1, activation='none')

    def forward(self, xin, mask):
        # For indicating the boundaries of images
        ones = torch.ones(xin.size(0), 1, xin.size(2), xin.size(3))
        if self.use_cuda:
            ones = ones.cuda()
            mask = mask.cuda()
        # conv branch
        xnow = torch.cat([xin, ones, mask], dim=1)
        x = self.conv1(xnow)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x_hallu = x
        # attention branch
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        x = self.contextul_attention(x, x, mask)
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], dim=1)
        # merge two branches
        x = self.allconv11(x)
        x = self.allconv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv13(x)
        x = self.allconv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv15(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        x_stage1 = torch.clamp(x, -1., 1.)

        return x_stage1



class FineGenerator(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(FineGenerator, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        # 3 x 256 x 256
        self.conv1 = gen_conv(input_dim, cnum, 5, 1, 2)
        self.conv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        # cnum*2 x 128 x 128
        self.conv3 = gen_conv(cnum, cnum * 2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cnum * 2, cnum * 2, 3, 2, 1)
        # cnum*4 x 64 x 64

        self.conv5 = gen_conv(cnum * 2, cnum * 4, 3, 1, 1)
        self.conv6 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)

        self.conv7_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 16, rate=16)

        # attention branch
        # 3 x 256 x 256
        self.pmconv1 = gen_conv(input_dim, cnum, 5, 1, 2)
        self.pmconv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        # cnum*2 x 128 x 128
        self.pmconv3 = gen_conv(cnum, cnum * 2, 3, 1, 1)
        self.pmconv4_downsample = gen_conv(cnum * 2, cnum * 4, 3, 2, 1)
        # cnum*4 x 64 x 64
        self.pmconv5 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.pmconv6 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1, activation='relu')
        self.contextul_attention = GlobalLocalAttention(in_dim=128)
        self.pmconv9 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.pmconv10 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)

        self.allconv11 = gen_conv(cnum * 8, cnum * 4, 3, 1, 1)
        self.allconv12 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)

        self.allconv13 = gen_conv(cnum * 4, cnum * 2, 3, 1, 1)
        self.allconv14 = gen_conv(cnum * 2, cnum * 2, 3, 1, 1)
        self.allconv15 = gen_conv(cnum * 2, cnum, 3, 1, 1)
        self.allconv16 = gen_conv(cnum, cnum // 2, 3, 1, 1)

        self.allconv17 = gen_conv(cnum // 2, 3, 3, 1, 1, activation='none')

    def forward(self, xin, x_stage1, mask):
        x1_inpaint = x_stage1 * mask + xin * (1. - mask)
        # For indicating the boundaries of images
        ones = torch.ones(xin.size(0), 1, xin.size(2), xin.size(3))
        if self.use_cuda:
            ones = ones.cuda()
            mask = mask.cuda()
        # conv branch
        xnow = torch.cat([x1_inpaint, ones, mask], dim=1)
        x = self.conv1(xnow)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x_hallu = x
        # attention branch
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        x = self.contextul_attention(x,mask)
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], dim=1)
        # merge two branches
        x = self.allconv11(x)
        x = self.allconv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv13(x)
        x = self.allconv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv15(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        x_stage2 = torch.clamp(x, -1., 1.)

        return x_stage2

class GlobalDis(nn.Module):
    def __init__(self, config, use_cuda=True, device_ids=None):
        super(GlobalDis, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ndf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        # self.linear = nn.Linear(self.cnum * 4 * 8 * 8, 1)
        # self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.dis_conv_module(x)
        # x = self.dropout(x)
        # x = x.view(x.size()[0], -1)
        # x = self.linear(x)

        return x

class DisConvModule(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(DisConvModule, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.conv1 = dis_conv(input_dim, cnum, 5, 2, 2)
        self.conv2 = dis_conv(cnum, cnum * 2, 5, 2, 2)
        self.conv3 = dis_conv(cnum * 2, cnum * 4, 5, 2, 2)
        self.conv4 = dis_conv(cnum * 4, cnum * 4, 5, 2, 2)
        self.conv5 = dis_conv(cnum * 4, cnum * 4, 5, 2, 2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)


        return x


def gen_conv(input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1,
             activation='elu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation)


def dis_conv(input_dim, output_dim, kernel_size=5, stride=2, padding=0, rate=1,
             activation='lrelu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation, weight_norm='sn')


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, weight_norm='none', norm='none',
                 activation='lrelu', pad_type='zeros', transpose=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zeros':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(weight_norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim,
                                           kernel_size, stride,
                                           padding=conv_padding,
                                           output_padding=conv_padding,
                                           dilation=dilation,
                                           bias=self.use_bias)

        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                  padding=conv_padding, dilation=dilation,
                                  bias=self.use_bias, padding_mode=pad_type)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, x):
        if self.pad:
            x = self.conv(self.pad(x))
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
