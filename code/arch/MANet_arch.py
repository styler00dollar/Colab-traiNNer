"""
net.py (11-4-20)
https://github.com/wangning-001/MANet/blob/2d6483c4d4db83396f6cf649e2458353fa5bfb78/net.py

ContextualAttention.py (11-4-20)
https://github.com/wangning-001/MANet/blob/2d6483c4d4db83396f6cf649e2458353fa5bfb78/ContextualAttention.py

partialconv2d.py (12-4-20)
https://github.com/wangning-001/MANet/blob/2d6483c4d4db83396f6cf649e2458353fa5bfb78/partialconv2d.py
"""

###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################

import torch
import torch.nn.functional as F
from torch import nn


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        # whether the mask is multi-channel or not
        if "multi_channel" in kwargs:
            self.multi_channel = kwargs["multi_channel"]
            kwargs.pop("multi_channel")
        else:
            self.multi_channel = False

        self.return_mask = True

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
            )
        else:
            self.weight_maskUpdater = torch.ones(
                1, 1, self.kernel_size[0], self.kernel_size[1]
            )

        self.slide_winsize = (
            self.weight_maskUpdater.shape[1]
            * self.weight_maskUpdater.shape[2]
            * self.weight_maskUpdater.shape[3]
        )

        self.last_size = (None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask=None):
        if mask is not None or self.last_size != (
            input.data.shape[2],
            input.data.shape[3],
        ):
            self.last_size = (input.data.shape[2], input.data.shape[3])

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(
                            input.data.shape[0],
                            input.data.shape[1],
                            input.data.shape[2],
                            input.data.shape[3],
                        ).to(input)
                    else:
                        mask = torch.ones(
                            1, 1, input.data.shape[2], input.data.shape[3]
                        ).to(input)

                self.update_mask = F.conv2d(
                    mask,
                    self.weight_maskUpdater,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=1,
                )

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        if (
            self.update_mask.type() != input.type()
            or self.mask_ratio.type() != input.type()
        ):
            self.update_mask.to(input)
            self.mask_ratio.to(input)

        raw_out = super(PartialConv2d, self).forward(
            torch.mul(input, mask) if mask is not None else input
        )

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


# from partialconv2d import PartialConv2d
# from .conv.partialconv import PartialConv2d
import torch.nn as nn
from torchvision import models

# import ContextualAttention
PartialConv = PartialConv2d

import torch.nn as nn


class SEModule(nn.Module):
    def __init__(self, num_channel, squeeze_ratio=1.0):
        super(SEModule, self).__init__()
        self.sequeeze_mod = nn.AdaptiveAvgPool2d(1)
        self.num_channel = num_channel

        blocks = [
            nn.Linear(num_channel, int(num_channel * squeeze_ratio)),
            nn.ReLU(),
            nn.Linear(int(num_channel * squeeze_ratio), num_channel),
            nn.Sigmoid(),
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        ori = x
        x = self.sequeeze_mod(x)
        x = x.view(x.size(0), 1, self.num_channel)
        x = self.blocks(x)
        x = x.view(x.size(0), self.num_channel, 1, 1)
        x = ori * x
        return x


class ContextualAttentionModule(nn.Module):
    def __init__(self, patch_size=3, propagate_size=3, stride=1):
        super(ContextualAttentionModule, self).__init__()
        self.patch_size = patch_size
        self.propagate_size = propagate_size
        self.stride = stride
        self.prop_kernels = None

    def forward(self, foreground, masks):
        ###assume the masked area has value 1
        bz, nc, w, h = foreground.size()
        if masks.size(3) != foreground.size(3):
            masks = F.interpolate(masks, foreground.size()[2:])
        background = foreground.clone()
        background = background * masks
        background = F.pad(
            background,
            [
                self.patch_size // 2,
                self.patch_size // 2,
                self.patch_size // 2,
                self.patch_size // 2,
            ],
        )
        conv_kernels_all = (
            background.unfold(2, self.patch_size, self.stride)
            .unfold(3, self.patch_size, self.stride)
            .contiguous()
            .view(bz, nc, -1, self.patch_size, self.patch_size)
        )
        conv_kernels_all = conv_kernels_all.transpose(2, 1)
        output_tensor = []
        for i in range(bz):
            mask = masks[i : i + 1]
            feature_map = foreground[i : i + 1]
            # form convolutional kernels
            conv_kernels = conv_kernels_all[i] + 0.0000001
            norm_factor = torch.sum(conv_kernels**2, [1, 2, 3], keepdim=True) ** 0.5
            conv_kernels = conv_kernels / norm_factor

            conv_result = F.conv2d(
                feature_map, conv_kernels, padding=self.patch_size // 2
            )
            if self.propagate_size != 1:
                if self.prop_kernels is None:
                    self.prop_kernels = torch.ones(
                        [
                            conv_result.size(1),
                            1,
                            self.propagate_size,
                            self.propagate_size,
                        ]
                    )
                    self.prop_kernels.requires_grad = False
                    self.prop_kernels = self.prop_kernels.cuda()
                conv_result = F.conv2d(
                    conv_result,
                    self.prop_kernels,
                    stride=1,
                    padding=1,
                    groups=conv_result.size(1),
                )
            attention_scores = F.softmax(conv_result, dim=1)
            ##propagate the scores
            recovered_foreground = F.conv_transpose2d(
                attention_scores, conv_kernels, stride=1, padding=self.patch_size // 2
            )
            # average the recovered value, at the same time make non-masked area 0
            recovered_foreground = (recovered_foreground * (1 - mask)) / (
                self.patch_size**2
            )
            # recover the image
            final_output = recovered_foreground + feature_map * mask
            output_tensor.append(final_output)
        return torch.cat(output_tensor, dim=0)


class PixelContextualAttention(nn.Module):
    def __init__(
        self, inchannel, patch_size_list=[1], propagate_size_list=[3], stride_list=[1]
    ):
        assert isinstance(
            patch_size_list, list
        ), "patch_size should be a list containing scales, or you should use Contextual Attention to initialize your module"
        assert len(patch_size_list) == len(propagate_size_list) and len(
            propagate_size_list
        ) == len(stride_list), "the input_lists should have same lengths"
        super(PixelContextualAttention, self).__init__()
        for i in range(len(patch_size_list)):
            name = "CA_{:d}".format(i)
            setattr(
                self,
                name,
                ContextualAttentionModule(
                    patch_size_list[i], propagate_size_list[i], stride_list[i]
                ),
            )
        self.num_of_modules = len(patch_size_list)
        self.SqueezeExc = SEModule(inchannel * 2)
        self.combiner = nn.Conv2d(inchannel * 2, inchannel, kernel_size=1)

    def forward(self, foreground, mask):
        outputs = [foreground]
        for i in range(self.num_of_modules):
            name = "CA_{:d}".format(i)
            CA_module = getattr(self, name)
            outputs.append(CA_module(foreground, mask))
        outputs = torch.cat(outputs, dim=1)
        outputs = self.SqueezeExc(outputs)
        outputs = self.combiner(outputs)
        return outputs


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, "enc_{:d}".format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, "enc_{:d}".format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


class PCBActiv(nn.Module):
    def __init__(
        self, in_ch, out_ch, bn=True, sample="none-3", activ="relu", conv_bias=True
    ):
        super().__init__()
        if sample == "down-5":
            self.conv = PartialConv(
                in_ch, out_ch, 5, 2, 2, bias=conv_bias, multi_channel=True
            )
        elif sample == "down-7":
            self.conv = PartialConv(
                in_ch, out_ch, 7, 2, 3, bias=conv_bias, multi_channel=True
            )
        elif sample == "down-3":
            self.conv = PartialConv(
                in_ch, out_ch, 3, 2, 1, bias=conv_bias, multi_channel=True
            )
        else:
            self.conv = PartialConv(
                in_ch, out_ch, 3, 1, 1, bias=conv_bias, multi_channel=True
            )

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == "relu":
            self.activation = nn.ReLU()
        elif activ == "leaky":
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, "bn"):
            h = self.bn(h)
        if hasattr(self, "activation"):
            h = self.activation(h)
        return h, h_mask


class PConvUNet(nn.Module):
    def __init__(self, layer_size=8, input_channels=3, upsampling_mode="nearest"):
        # def __init__(self, layer_size=8, input_channels=3, upsampling_mode='bilinear'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample="down-7")
        self.enc_2 = PCBActiv(64, 128, sample="down-5")
        self.enc_3 = PCBActiv(128, 256, sample="down-5")
        self.enc_4 = PCBActiv(256, 512, sample="down-3")
        for i in range(4, self.layer_size):
            name = "enc_{:d}".format(i + 1)
            setattr(self, name, PCBActiv(512, 512, sample="down-3"))

        for i in range(4, self.layer_size):
            name = "dec_{:d}".format(i + 1)
            setattr(self, name, PCBActiv(512 + 512, 512, activ="leaky"))
        self.dec_4 = PCBActiv(512 + 256, 256, activ="leaky")
        self.dec_3 = PCBActiv(256 + 128, 128, activ="leaky")
        self.dec_2 = PCBActiv(128 + 64, 64, activ="leaky")
        self.dec_1 = PCBActiv(
            64 + input_channels, input_channels, bn=False, activ=None, conv_bias=True
        )
        self.CA_1 = PixelContextualAttention(256)
        self.CA_2 = PixelContextualAttention(128)

    def forward(self, input, input_mask):
        input_mask = torch.cat([input_mask, input_mask, input_mask], dim=1)

        input = input.type(torch.cuda.FloatTensor)
        input_mask = input_mask.type(torch.cuda.FloatTensor)

        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N

        h_dict["h_0"], h_mask_dict["h_0"] = input, input_mask

        h_key_prev = "h_0"
        for i in range(1, self.layer_size + 1):
            l_key = "enc_{:d}".format(i)
            h_key = "h_{:d}".format(i)
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
                h_dict[h_key_prev], h_mask_dict[h_key_prev]
            )
            h_key_prev = h_key

        h_key = "h_{:d}".format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]

        for i in range(self.layer_size, 0, -1):
            enc_h_key = "h_{:d}".format(i - 1)
            dec_l_key = "dec_{:d}".format(i)

            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
            h_mask = F.interpolate(h_mask, scale_factor=2, mode="nearest")

            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)
            if i == 4:
                h = self.CA_1(h, input_mask[:, 0:1, :, :])
            if i == 3:
                h = self.CA_2(h, input_mask[:, 0:1, :, :])
        return h  # , h_mask
