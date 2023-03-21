"""
model.py (18-12-20)
https://github.com/jingyuanli001/PRVS-Image-Inpainting/blob/master/model.py

PRVSNet.py (18-12-20)
https://github.com/jingyuanli001/PRVS-Image-Inpainting/blob/master/modules/PRVSNet.py

partialconv2d.py (18-12-20) # using their partconv2d to avoid dimension errors
https://github.com/jingyuanli001/PRVS-Image-Inpainting/blob/master/modules/partialconv2d.py

PConvLayer.py (18-12-20)
https://github.com/jingyuanli001/PRVS-Image-Inpainting/blob/master/modules/PConvLayer.py

VSRLayer.py (18-12-20)
https://github.com/jingyuanli001/PRVS-Image-Inpainting/blob/master/modules/VSRLayer.py

Attention.py (18-12-20)
https://github.com/jingyuanli001/PRVS-Image-Inpainting/blob/master/modules/Attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models

# from .convolutions import partialconv2d
import pytorch_lightning as pl

###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################


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


PartialConv = PartialConv2d


class AttentionModule(pl.LightningModule):
    def __init__(self, patch_size=3, propagate_size=3, stride=1):
        super(AttentionModule, self).__init__()
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


class Bottleneck(pl.LightningModule):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class EdgeGenerator(pl.LightningModule):
    def __init__(self, in_channels_feature, kernel_s=3, add_last_edge=True):
        super(EdgeGenerator, self).__init__()

        self.p_conv = PartialConv2d(
            in_channels_feature + 1,
            64,
            kernel_size=kernel_s,
            stride=1,
            padding=kernel_s // 2,
            multi_channel=True,
            bias=False,
        )

        self.edge_resolver = Bottleneck(64, 16)
        self.out_layer = nn.Conv2d(64, 1, 1, bias=False)

    def forward(self, in_x, mask):
        x, mask_updated = self.p_conv(in_x, mask)
        x = self.edge_resolver(x)
        edge_out = self.out_layer(x)
        return edge_out, mask_updated


class VSRLayer(pl.LightningModule):
    def __init__(
        self,
        in_channel,
        out_channel,
        stride=2,
        kernel_size=3,
        batch_norm=True,
        activation="ReLU",
        deconv=False,
    ):
        super(VSRLayer, self).__init__()
        self.edge_generator = EdgeGenerator(in_channel, kernel_s=kernel_size)
        self.feat_rec = PartialConv(
            in_channel + 1,
            out_channel,
            stride=stride,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            multi_channel=True,
        )
        if deconv:
            self.deconv = nn.ConvTranspose2d(out_channel, out_channel, 4, 2, 1)
        else:
            self.deconv = lambda x: x

        if batch_norm:
            self.batchnorm = nn.BatchNorm2d(out_channel)
        else:
            self.batchnorm = lambda x: x

        self.stride = stride

        if activation == "ReLU":
            self.activation = nn.ReLU(True)
        elif activation == "Leaky":
            self.activation = nn.LeakyReLU(0.2, True)
        else:
            self.activation = lambda x: x

    def forward(self, feat_in, mask_in, edge_in):
        edge_in = F.interpolate(edge_in, size=feat_in.size()[2:])
        edge_updated, mask_updated = self.edge_generator(
            torch.cat([feat_in, edge_in], dim=1),
            torch.cat([mask_in, mask_in[:, :1, :, :]], dim=1),
        )
        edge_reconstructed = edge_in * mask_in[:, :1, :, :] + edge_updated * (
            mask_updated[:, :1, :, :] - mask_in[:, :1, :, :]
        )
        feat_out, feat_mask = self.feat_rec(
            torch.cat([feat_in, edge_reconstructed], dim=1),
            torch.cat([mask_in, mask_updated[:, :1, :, :]], dim=1),
        )
        feat_out = self.deconv(feat_out)
        feat_out = self.batchnorm(feat_out)
        feat_out = self.activation(feat_out)
        mask_updated = F.interpolate(mask_updated, size=feat_out.size()[2:])
        feat_mask = F.interpolate(feat_mask, size=feat_out.size()[2:])
        return feat_out, feat_mask * mask_updated[:, 0:1, :, :], edge_reconstructed


class PConvLayer(pl.LightningModule):
    def __init__(
        self,
        in_ch,
        out_ch,
        bn=True,
        sample="none-3",
        activ="relu",
        conv_bias=False,
        deconv=False,
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
        if deconv:
            self.deconv = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1, bias=conv_bias)
        else:
            self.deconv = None
        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == "relu":
            self.activation = nn.ReLU()
        elif activ == "leaky":
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if self.deconv is not None:
            h = self.deconv(h)
        if hasattr(self, "bn"):
            h = self.bn(h)
        if hasattr(self, "activation"):
            h = self.activation(h)
        h_mask = F.interpolate(h_mask, size=h.size()[2:])
        return h, h_mask


class VGG16FeatureExtractor(pl.LightningModule):
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


class Bottleneck(pl.LightningModule):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)

        return out


class PRVSNet(pl.LightningModule):
    def __init__(self, layer_size=8, input_channels=3, att=False):
        super().__init__()
        self.layer_size = layer_size
        self.enc_1 = VSRLayer(3, 64, kernel_size=7)
        self.enc_2 = VSRLayer(64, 128, kernel_size=5)
        self.enc_3 = PConvLayer(128, 256, sample="down-5")
        self.enc_4 = PConvLayer(256, 512, sample="down-3")
        for i in range(4, self.layer_size):
            name = "enc_{:d}".format(i + 1)
            setattr(self, name, PConvLayer(512, 512, sample="down-3"))
        self.deconv = nn.ConvTranspose2d(512, 512, 4, 2, 1)
        for i in range(4, self.layer_size):
            name = "dec_{:d}".format(i + 1)
            setattr(self, name, PConvLayer(512 + 512, 512, activ="leaky", deconv=True))
        self.dec_4 = PConvLayer(512 + 256, 256, activ="leaky", deconv=True)
        if att:
            self.att = Attention.AttentionModule()
        else:
            self.att = lambda x: x
        self.dec_3 = PConvLayer(256 + 128, 128, activ="leaky", deconv=True)
        self.dec_2 = VSRLayer(128 + 64, 64, stride=1, activation="leaky", deconv=True)
        self.dec_1 = VSRLayer(
            64 + input_channels, 64, stride=1, activation=None, batch_norm=False
        )
        self.resolver = Bottleneck(64, 16)
        self.output = nn.Conv2d(128, 3, 1)

    def forward(self, input, input_mask, input_edge):
        input = input.type(torch.cuda.FloatTensor)
        input_mask = input_mask.type(torch.cuda.FloatTensor)
        input_edge = input_edge.type(torch.cuda.FloatTensor)

        input = input * input_mask[:, 0:1, :, :]
        input_edge = input_edge * input_mask[:, 0:1, :, :]
        input_mask = torch.cat([input_mask] * 3, dim=1)
        input_mask = input_mask[:, :3, :, :]

        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N
        h_edge_list = []
        h_dict["h_0"], h_mask_dict["h_0"] = input, input_mask
        edge = input_edge

        h_key_prev = "h_0"
        for i in range(1, self.layer_size + 1):
            l_key = "enc_{:d}".format(i)
            h_key = "h_{:d}".format(i)
            if i not in [1, 2]:
                h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
                    h_dict[h_key_prev], h_mask_dict[h_key_prev]
                )
            else:
                h_dict[h_key], h_mask_dict[h_key], edge = getattr(self, l_key)(
                    h_dict[h_key_prev], h_mask_dict[h_key_prev], edge
                )
                h_edge_list.append(edge)
            h_key_prev = h_key

        h_key = "h_{:d}".format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]
        h = self.deconv(h)
        h_mask = F.interpolate(h_mask, scale_factor=2)

        for i in range(self.layer_size, 0, -1):
            enc_h_key = "h_{:d}".format(i - 1)
            dec_l_key = "dec_{:d}".format(i)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)
            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            if i not in [2, 1]:
                h, h_mask = getattr(self, dec_l_key)(h, h_mask)
            else:
                edge = h_edge_list[i - 1]
                h, h_mask, edge = getattr(self, dec_l_key)(h, h_mask, edge)
                h_edge_list.append(edge)
            if i == 4:
                h = self.att(h)
        h_out = self.resolver(h)
        h_out = torch.cat([h_out, h], dim=1)
        h_out = self.output(h_out)
        return h_out, h_mask, h_edge_list[-2], h_edge_list[-1]
