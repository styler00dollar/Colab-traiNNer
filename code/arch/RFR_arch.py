"""
RFRNet.py (18-12-20)
https://github.com/jingyuanli001/RFR-Inpainting/blob/master/modules/RFRNet.py

partialconv2d.py (18-12-20) # using their partconv2d to avoid dimension errors
https://github.com/jingyuanli001/RFR-Inpainting/blob/master/modules/partialconv2d.py

Attention.py (18-12-20)
https://github.com/jingyuanli001/RFR-Inpainting/blob/master/modules/Attention.py
"""

from torch import nn
import torch
import torch.nn.functional as F

# from models.modules.architectures.convolutions.deformconv2d import DeformConv2d
import pytorch_lightning as pl


class KnowledgeConsistentAttention(pl.LightningModule):
    def __init__(self, patch_size=3, propagate_size=3, stride=1):
        super(KnowledgeConsistentAttention, self).__init__()
        self.patch_size = patch_size
        self.propagate_size = propagate_size
        self.stride = stride
        self.prop_kernels = None
        self.att_scores_prev = None
        self.masks_prev = None
        self.ratio = nn.Parameter(torch.ones(1))

    def forward(self, foreground, masks):
        bz, nc, h, w = foreground.size()
        if masks.size(3) != foreground.size(3):
            masks = F.interpolate(masks, foreground.size()[2:])
        background = foreground.clone()
        background = background
        conv_kernels_all = background.view(bz, nc, w * h, 1, 1)
        conv_kernels_all = conv_kernels_all.permute(0, 2, 1, 3, 4)
        output_tensor = []
        att_score = []
        for i in range(bz):
            feature_map = foreground[i : i + 1]
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
                conv_result = F.avg_pool2d(conv_result, 3, 1, padding=1) * 9
            attention_scores = F.softmax(conv_result, dim=1)
            if self.att_scores_prev is not None:
                attention_scores = (
                    self.att_scores_prev[i : i + 1] * self.masks_prev[i : i + 1]
                    + attention_scores * (torch.abs(self.ratio) + 1e-7)
                ) / (self.masks_prev[i : i + 1] + (torch.abs(self.ratio) + 1e-7))
            att_score.append(attention_scores)
            feature_map = F.conv_transpose2d(
                attention_scores, conv_kernels, stride=1, padding=self.patch_size // 2
            )
            final_output = feature_map
            output_tensor.append(final_output)
        self.att_scores_prev = torch.cat(att_score, dim=0).view(bz, h * w, h, w)
        self.masks_prev = masks.view(bz, 1, h, w)
        return torch.cat(output_tensor, dim=0)


class AttentionModule(pl.LightningModule):
    def __init__(
        self, inchannel, patch_size_list=[1], propagate_size_list=[3], stride_list=[1]
    ):
        assert isinstance(
            patch_size_list, list
        ), "patch_size should be a list containing scales, or you should use Contextual Attention to initialize your module"
        assert len(patch_size_list) == len(propagate_size_list) and len(
            propagate_size_list
        ) == len(stride_list), "the input_lists should have same lengths"
        super(AttentionModule, self).__init__()

        self.att = KnowledgeConsistentAttention(
            patch_size_list[0], propagate_size_list[0], stride_list[0]
        )
        self.num_of_modules = len(patch_size_list)
        self.combiner = nn.Conv2d(inchannel * 2, inchannel, kernel_size=1)

    def forward(self, foreground, mask):
        outputs = self.att(foreground, mask)
        outputs = torch.cat([outputs, foreground], dim=1)
        outputs = self.combiner(outputs)
        return outputs


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


class RFRModule(pl.LightningModule):
    def __init__(self, layer_size=6, in_channel=64):
        super(RFRModule, self).__init__()
        self.freeze_enc_bn = False
        self.layer_size = layer_size
        for i in range(3):
            name = "enc_{:d}".format(i + 1)
            out_channel = in_channel * 2
            block = [
                nn.Conv2d(in_channel, out_channel, 3, 2, 1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
            ]
            in_channel = out_channel
            setattr(self, name, nn.Sequential(*block))

        for i in range(3, 6):
            name = "enc_{:d}".format(i + 1)
            block = [
                nn.Conv2d(in_channel, out_channel, 3, 1, 2, dilation=2, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
            ]
            setattr(self, name, nn.Sequential(*block))
        self.att = AttentionModule(512)
        for i in range(5, 3, -1):
            name = "dec_{:d}".format(i)
            block = [
                nn.Conv2d(
                    in_channel + in_channel, in_channel, 3, 1, 2, dilation=2, bias=False
                ),
                nn.BatchNorm2d(in_channel),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            setattr(self, name, nn.Sequential(*block))

        block = [
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.dec_3 = nn.Sequential(*block)

        block = [
            nn.ConvTranspose2d(768, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.dec_2 = nn.Sequential(*block)

        block = [
            nn.ConvTranspose2d(384, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.dec_1 = nn.Sequential(*block)

    def forward(self, input, mask):
        h_dict = {}  # for the output of enc_N

        h_dict["h_0"] = input

        h_key_prev = "h_0"
        for i in range(1, self.layer_size + 1):
            l_key = "enc_{:d}".format(i)
            h_key = "h_{:d}".format(i)
            h_dict[h_key] = getattr(self, l_key)(h_dict[h_key_prev])
            h_key_prev = h_key

        h = h_dict[h_key]
        for i in range(self.layer_size - 1, 0, -1):
            enc_h_key = "h_{:d}".format(i)
            dec_l_key = "dec_{:d}".format(i)
            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h = getattr(self, dec_l_key)(h)
            if i == 3:
                h = self.att(h, mask)
        return h


class RFRNet(pl.LightningModule):
    def __init__(self, conv_type):
        super(RFRNet, self).__init__()

        self.conv_type = conv_type
        if self.conv_type == "partial":
            self.conv1 = PartialConv2d(3, 64, 7, 2, 3, multi_channel=True, bias=False)
            self.conv2 = PartialConv2d(64, 64, 7, 1, 3, multi_channel=True, bias=False)
            self.conv21 = PartialConv2d(64, 64, 7, 1, 3, multi_channel=True, bias=False)
            self.conv22 = PartialConv2d(64, 64, 7, 1, 3, multi_channel=True, bias=False)
            self.tail1 = PartialConv2d(67, 32, 3, 1, 1, multi_channel=True, bias=False)
            # original code uses conv2d
            self.out = nn.Conv2d(64, 3, 3, 1, 1, bias=False)
        elif self.conv_type == "deform":
            self.conv1 = DeformConv2d(3, 64, 7, 2, 3)
            self.conv2 = DeformConv2d(64, 64, 7, 1, 3)
            self.conv21 = DeformConv2d(64, 64, 7, 1, 3)
            self.conv22 = DeformConv2d(64, 64, 7, 1, 3)
            self.tail1 = DeformConv2d(67, 32, 3, 1, 1)
            # original code uses conv2d
            self.out = nn.Conv2d(64, 3, 3, 1, 1, bias=False)
        else:
            print("conv_type not found")

        self.bn1 = nn.BatchNorm2d(64)
        self.bn20 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.RFRModule = RFRModule()
        self.Tconv = nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.tail2 = Bottleneck(32, 8)

    def forward(self, in_image, mask):
        # in_image = torch.cat((in_image, mask), dim=1)
        mask = torch.cat([mask, mask, mask], 1)
        if self.conv_type == "partial":
            x1, m1 = self.conv1(
                in_image.type(torch.cuda.FloatTensor), mask.type(torch.cuda.FloatTensor)
            )
        elif self.conv_type == "deform":
            x1 = self.conv1(in_image)
            m1 = self.conv1(mask)

        x1 = F.relu(self.bn1(x1), inplace=True)

        if self.conv_type == "partial":
            x1, m1 = self.conv2(x1, m1)
        elif self.conv_type == "deform":
            x1 = self.conv2(x1)
            m1 = self.conv2(m1)

        x1 = F.relu(self.bn20(x1), inplace=True)
        x2 = x1
        x2, m2 = x1, m1
        n, c, h, w = x2.size()
        feature_group = [x2.view(n, c, 1, h, w)]
        mask_group = [m2.view(n, c, 1, h, w)]
        self.RFRModule.att.att.att_scores_prev = None
        self.RFRModule.att.att.masks_prev = None

        for i in range(6):
            if self.conv_type == "partial":
                x2, m2 = self.conv21(x2, m2)
                x2, m2 = self.conv22(x2, m2)
            elif self.conv_type == "deform":
                x2 = self.conv21(x2)
                m2 = self.conv21(m2)
                x2 = self.conv22(x2)
                m2 = self.conv22(m2)

            x2 = F.leaky_relu(self.bn2(x2), inplace=True)
            x2 = self.RFRModule(x2, m2[:, 0:1, :, :])
            x2 = x2 * m2
            feature_group.append(x2.view(n, c, 1, h, w))
            mask_group.append(m2.view(n, c, 1, h, w))
        x3 = torch.cat(feature_group, dim=2)
        m3 = torch.cat(mask_group, dim=2)
        amp_vec = m3.mean(dim=2)
        x3 = (x3 * m3).mean(dim=2) / (amp_vec + 1e-7)
        x3 = x3.view(n, c, h, w)
        m3 = m3[:, :, -1, :, :]
        x4 = self.Tconv(x3)
        x4 = F.leaky_relu(self.bn3(x4), inplace=True)
        m4 = F.interpolate(m3, scale_factor=2)
        x5 = torch.cat([in_image, x4], dim=1)
        m5 = torch.cat([mask, m4], dim=1)

        if self.conv_type == "partial":
            x5, _ = self.tail1(x5, m5)
        elif self.conv_type == "deform":
            x5 = self.tail1(x5)

        x5 = F.leaky_relu(x5, inplace=True)
        x6 = self.tail2(x5)
        x6 = torch.cat([x5, x6], dim=1)
        output = self.out(x6)
        return output
