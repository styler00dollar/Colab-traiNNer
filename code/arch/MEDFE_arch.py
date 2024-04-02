"""
Encoder.py (25-12-20)
https://github.com/KumapowerLIU/Rethinking-Inpainting-MEDFE/blob/9adf8898a142784976bb3e162a9fd864c224e01e/models/Encoder.py

Decoder.py (25-12-20)
https://github.com/KumapowerLIU/Rethinking-Inpainting-MEDFE/blob/9adf8898a142784976bb3e162a9fd864c224e01e/models/Decoder.py

networks.py (25-12-20)
https://github.com/KumapowerLIU/Rethinking-Inpainting-MEDFE/blob/9adf8898a142784976bb3e162a9fd864c224e01e/models/networks.py

MEDFE.py (25-12-20)
https://github.com/KumapowerLIU/Rethinking-Inpainting-MEDFE/blob/master/models/MEDFE.py

PCconv.py (25-12-20)
https://github.com/KumapowerLIU/Rethinking-Inpainting-MEDFE/blob/dd838b01d9786dc2c67de5d71869e5a60da28eb9/models/PCconv.py

Selfpatch.py (25-12-20)
https://github.com/KumapowerLIU/Rethinking-Inpainting-MEDFE/blob/dd838b01d9786dc2c67de5d71869e5a60da28eb9/util/Selfpatch.py

util.py (25-12-20)
https://github.com/KumapowerLIU/Rethinking-Inpainting-MEDFE/blob/dd838b01d9786dc2c67de5d71869e5a60da28eb9/util/util.py

InnerCos.py (25-12-20)
https://github.com/KumapowerLIU/Rethinking-Inpainting-MEDFE/blob/c7156eab4a9890888fa86e641cd685e21b78c31e/models/InnerCos.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.autograd import Variable
import collections
import inspect
import re
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class InnerCos(pl.LightningModule):
    def __init__(self):
        super(InnerCos, self).__init__()
        self.criterion = nn.L1Loss()
        self.target = None
        self.down_model = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=1, stride=1, padding=0), nn.Tanh()
        )

    def set_target(self, targetde, targetst):
        self.targetst = F.interpolate(targetst, size=(32, 32), mode="bilinear")
        self.targetde = F.interpolate(targetde, size=(32, 32), mode="bilinear")

    def get_target(self):
        return self.target

    def forward(self, in_data):
        loss_co = in_data[1]
        self.ST = self.down_model(loss_co[0])
        self.DE = self.down_model(loss_co[1])
        # self.loss = self.criterion(self.ST, self.targetst)+self.criterion(self.DE, self.targetde)
        self.output = in_data[0]
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

    def __repr__(self):
        return self.__class__.__name__


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name="network"):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def binary_mask(in_mask, threshold):
    assert in_mask.dim() == 2, "mask must be 2 dimensions"

    output = torch.ByteTensor(in_mask.size())
    output = (output > threshold).float().mul_(1)

    return output


def gussin(v):
    outk = []
    v = v
    for i in range(32):
        for k in range(32):
            out = []
            for x in range(32):
                row = []
                for y in range(32):
                    cord_x = i
                    cord_y = k
                    dis_x = np.abs(x - cord_x)
                    dis_y = np.abs(y - cord_y)
                    dis_add = -(dis_x * dis_x + dis_y * dis_y)
                    dis_add = dis_add / (2 * v * v)
                    dis_add = math.exp(dis_add) / (2 * math.pi * v * v)

                    row.append(dis_add)
                out.append(row)

            outk.append(out)

    out = np.array(outk)
    f = out.sum(-1).sum(-1)
    q = []
    for i in range(1024):
        g = out[i] / f[i]
        q.append(g)
    out = np.array(q)
    return torch.from_numpy(out)


def cal_feat_mask(inMask, conv_layers, threshold):
    assert inMask.dim() == 4, "mask must be 4 dimensions"
    assert inMask.size(0) == 1, "the first dimension must be 1 for mask"
    inMask = inMask.float()
    convs = []
    inMask = Variable(inMask, requires_grad=False)
    for id_net in range(conv_layers):
        conv = nn.Conv2d(1, 1, 4, 2, 1, bias=False)
        conv.weight.data.fill_(1 / 16)
        convs.append(conv)
    lnet = nn.Sequential(*convs)
    if inMask.is_cuda:
        lnet = lnet.cuda()
    output = lnet(inMask)
    output = (output > threshold).float().mul_(1)

    return output


def cal_mask_given_mask_thred(img, mask, patch_size, stride, mask_thred):
    assert img.dim() == 3, "img has to be 3 dimenison!"
    assert mask.dim() == 2, "mask has to be 2 dimenison!"
    dim = img.dim()
    # math.floor 是向下取整
    _, H, W = img.size(dim - 3), img.size(dim - 2), img.size(dim - 1)
    nH = int(math.floor((H - patch_size) / stride + 1))
    nW = int(math.floor((W - patch_size) / stride + 1))
    N = nH * nW

    flag = torch.zeros(N).long()
    offsets_tmp_vec = torch.zeros(N).long()
    # 返回的是一个list类型的数据

    nonmask_point_idx_all = torch.zeros(N).long()

    tmp_non_mask_idx = 0

    mask_point_idx_all = torch.zeros(N).long()

    tmp_mask_idx = 0
    # 所有的像素点都浏览一遍
    for i in range(N):
        h = int(math.floor(i / nW))
        w = int(math.floor(i % nW))
        # print(h, w)
        # 截取一个个1×1的小方片
        mask_tmp = mask[
            h * stride : h * stride + patch_size, w * stride : w * stride + patch_size
        ]

        if torch.sum(mask_tmp) < mask_thred:
            nonmask_point_idx_all[tmp_non_mask_idx] = i
            tmp_non_mask_idx += 1
        else:
            mask_point_idx_all[tmp_mask_idx] = i
            tmp_mask_idx += 1
            flag[i] = 1
            offsets_tmp_vec[i] = -1
    # print(flag)  #checked
    # print(offsets_tmp_vec) # checked

    non_mask_num = tmp_non_mask_idx
    mask_num = tmp_mask_idx

    nonmask_point_idx = nonmask_point_idx_all.narrow(0, 0, non_mask_num)
    mask_point_idx = mask_point_idx_all.narrow(0, 0, mask_num)

    # get flatten_offsets
    flatten_offsets_all = torch.LongTensor(N).zero_()
    for i in range(N):
        offset_value = torch.sum(offsets_tmp_vec[0 : i + 1])
        if flag[i] == 1:
            offset_value = offset_value + 1
        # print(i+offset_value)
        flatten_offsets_all[i + offset_value] = -offset_value

    flatten_offsets = flatten_offsets_all.narrow(0, 0, non_mask_num)

    # print('flatten_offsets')
    # print(flatten_offsets)   # checked

    # print('nonmask_point_idx')
    # print(nonmask_point_idx)  #checked

    return flag, nonmask_point_idx, flatten_offsets, mask_point_idx


# sp_x: LongTensor
# sp_y: LongTensor
def cal_sps_for_Advanced_Indexing(h, w):
    sp_y = torch.arange(0, w).long()
    sp_y = torch.cat([sp_y] * h)

    lst = []
    for i in range(h):
        lst.extend([i] * w)
    sp_x = torch.from_numpy(np.array(lst))
    return sp_x, sp_y


"""
def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
"""


def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [
        e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)
    ]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print(
        "\n".join(
            [
                "%s %s"
                % (
                    method.ljust(spacing),
                    processFunc(str(getattr(object, method).__doc__)),
                )
                for method in methodList
            ]
        )
    )


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r"\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", line)
        if m:
            return m.group(1)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print("shape,", x.shape)
    if val:
        x = x.flatten()
        print(
            "mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f"
            % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x))
        )


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Selfpatch(object):
    def buildAutoencoder(
        self, target_img, target_img_2, target_img_3, patch_size=1, stride=1
    ):
        nDim = 3
        assert target_img.dim() == nDim, "target image must be of dimension 3."
        target_img.size(0)

        self.Tensor = (
            torch.cuda.FloatTensor if torch.cuda.is_available else torch.Tensor
        )

        patches_features = self._extract_patches(target_img, patch_size, stride)
        patches_features_f = self._extract_patches(target_img_3, patch_size, stride)

        patches_on = self._extract_patches(target_img_2, 1, stride)

        return patches_features_f, patches_features, patches_on

    def build(self, target_img, patch_size=5, stride=1):
        nDim = 3
        assert target_img.dim() == nDim, "target image must be of dimension 3."
        target_img.size(0)

        self.Tensor = (
            torch.cuda.FloatTensor if torch.cuda.is_available else torch.Tensor
        )

        patches_features = self._extract_patches(target_img, patch_size, stride)

        return patches_features

    def _build(
        self,
        patch_size,
        stride,
        C,
        target_patches,
        npatches,
        normalize,
        interpolate,
        type,
    ):
        # for each patch, divide by its L2 norm.
        if type == 1:
            enc_patches = target_patches.clone()
            for i in range(npatches):
                enc_patches[i] = enc_patches[i] * (1 / (enc_patches[i].norm(2) + 1e-8))

            conv_enc = nn.Conv2d(
                npatches,
                npatches,
                kernel_size=1,
                stride=stride,
                bias=False,
                groups=npatches,
            )
            conv_enc.weight.data = enc_patches
            return conv_enc

            # normalize is not needed, it doesn't change the result!
            if normalize:
                raise NotImplementedError

            if interpolate:
                raise NotImplementedError
        else:
            conv_dec = nn.ConvTranspose2d(
                npatches, C, kernel_size=patch_size, stride=stride, bias=False
            )
            conv_dec.weight.data = target_patches
            return conv_dec

    def _extract_patches(self, img, patch_size, stride):
        n_dim = 3
        assert img.dim() == n_dim, "image must be of dimension 3."
        kH, kW = patch_size, patch_size
        dH, dW = stride, stride
        input_windows = img.unfold(1, kH, dH).unfold(2, kW, dW)
        i_1, i_2, i_3, i_4, i_5 = (
            input_windows.size(0),
            input_windows.size(1),
            input_windows.size(2),
            input_windows.size(3),
            input_windows.size(4),
        )
        input_windows = (
            input_windows.permute(1, 2, 0, 3, 4)
            .contiguous()
            .view(i_2 * i_3, i_1, i_4, i_5)
        )
        patches_all = input_windows
        return patches_all


# SE MODEL
class SELayer(pl.LightningModule):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(
                channel, channel // reduction, kernel_size=1, stride=1, padding=0
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channel // reduction, channel, kernel_size=1, stride=1, padding=0
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.fc(y)
        return x * y.expand_as(x)


class Convnorm(pl.LightningModule):
    def __init__(self, in_ch, out_ch, sample="none-3", activ="leaky"):
        super().__init__()
        self.bn = nn.InstanceNorm2d(out_ch, affine=True)

        if sample == "down-3":
            self.conv = nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False)
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, 3, 1)
        if activ == "leaky":
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input):
        out = input
        out = self.conv(out)
        out = self.bn(out)
        if hasattr(self, "activation"):
            out = self.activation(out[0])
        return out


class PCBActiv(pl.LightningModule):
    def __init__(
        self,
        in_ch,
        out_ch,
        bn=True,
        sample="none-3",
        activ="leaky",
        conv_bias=False,
        innorm=False,
        inner=False,
        outer=False,
    ):
        super().__init__()
        if sample == "same-5":
            self.conv = PartialConv(in_ch, out_ch, 5, 1, 2, bias=conv_bias)
        elif sample == "same-7":
            self.conv = PartialConv(in_ch, out_ch, 7, 1, 3, bias=conv_bias)
        elif sample == "down-3":
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.InstanceNorm2d(out_ch, affine=True)
        if activ == "relu":
            self.activation = nn.ReLU()
        elif activ == "leaky":
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.innorm = innorm
        self.inner = inner
        self.outer = outer

    def forward(self, input):
        out = input
        if self.inner:
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])
            out = self.conv(out)
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])

        elif self.innorm:
            out = self.conv(out)
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])
        elif self.outer:
            out = self.conv(out)
            out[0] = self.bn(out[0])
        else:
            out = self.conv(out)
            out[0] = self.bn(out[0])
            if hasattr(self, "activation"):
                out[0] = self.activation(out[0])
        return out


class ConvDown(pl.LightningModule):
    def __init__(
        self,
        in_c,
        out_c,
        kernel,
        stride,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        layers=1,
        activ=True,
    ):
        super().__init__()
        nf_mult = 1
        nums = out_c / 64
        sequence = []

        for i in range(1, layers + 1):
            nf_mult_prev = nf_mult
            if nums == 8:
                if in_c == 512:
                    pass
                else:
                    nf_mult = 2

            else:
                nf_mult = min(2**i, 8)
            if kernel != 1:
                if activ is False and layers == 1:
                    sequence += [
                        nn.Conv2d(
                            nf_mult_prev * in_c,
                            nf_mult * in_c,
                            kernel_size=kernel,
                            stride=stride,
                            padding=padding,
                            bias=bias,
                        ),
                        nn.InstanceNorm2d(nf_mult * in_c),
                    ]
                else:
                    sequence += [
                        nn.Conv2d(
                            nf_mult_prev * in_c,
                            nf_mult * in_c,
                            kernel_size=kernel,
                            stride=stride,
                            padding=padding,
                            bias=bias,
                        ),
                        nn.InstanceNorm2d(nf_mult * in_c),
                        nn.LeakyReLU(0.2, True),
                    ]

            else:
                sequence += [
                    nn.Conv2d(
                        in_c,
                        out_c,
                        kernel_size=kernel,
                        stride=stride,
                        padding=padding,
                        bias=bias,
                    ),
                    nn.InstanceNorm2d(out_c),
                    nn.LeakyReLU(0.2, True),
                ]

            if activ is False:
                if i + 1 == layers:
                    if layers == 2:
                        sequence += [
                            nn.Conv2d(
                                nf_mult * in_c,
                                nf_mult * in_c,
                                kernel_size=kernel,
                                stride=stride,
                                padding=padding,
                                bias=bias,
                            ),
                            nn.InstanceNorm2d(nf_mult * in_c),
                        ]
                    else:
                        sequence += [
                            nn.Conv2d(
                                nf_mult_prev * in_c,
                                nf_mult * in_c,
                                kernel_size=kernel,
                                stride=stride,
                                padding=padding,
                                bias=bias,
                            ),
                            nn.InstanceNorm2d(nf_mult * in_c),
                        ]
                    break

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class ConvUp(pl.LightningModule):
    def __init__(
        self, in_c, out_c, kernel, stride, padding=0, dilation=1, groups=1, bias=False
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_c, out_c, kernel, stride, padding, dilation, groups, bias
        )
        self.bn = nn.InstanceNorm2d(out_c)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, size):
        out = F.interpolate(input=input, size=size, mode="bilinear")
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class BASE(pl.LightningModule):
    def __init__(self, inner_nc):
        super(BASE, self).__init__()
        se = SELayer(inner_nc, 16)
        model = [se]
        gus = gussin(1.5).cuda()
        self.gus = torch.unsqueeze(gus, 1).double()
        self.model = nn.Sequential(*model)
        self.down = nn.Sequential(
            nn.Conv2d(1024, 512, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        Nonparm = Selfpatch()
        out_32 = self.model(x)
        b, c, h, w = out_32.size()
        gus = self.gus.float()
        gus_out = out_32[0].expand(h * w, c, h, w)
        gus_out = gus * gus_out
        gus_out = torch.sum(gus_out, -1)
        gus_out = torch.sum(gus_out, -1)
        gus_out = gus_out.contiguous().view(b, c, h, w)
        csa2_in = F.sigmoid(out_32)
        csa2_f = torch.nn.functional.pad(csa2_in, (1, 1, 1, 1))
        csa2_ff = torch.nn.functional.pad(out_32, (1, 1, 1, 1))
        csa2_fff, csa2_f, csa2_conv = Nonparm.buildAutoencoder(
            csa2_f[0], csa2_in[0], csa2_ff[0], 3, 1
        )
        csa2_conv = csa2_conv.expand_as(csa2_f)
        csa_a = csa2_conv * csa2_f
        csa_a = torch.mean(csa_a, 1)
        a_c, a_h, a_w = csa_a.size()
        csa_a = csa_a.contiguous().view(a_c, -1)
        csa_a = F.softmax(csa_a, dim=1)
        csa_a = csa_a.contiguous().view(a_c, 1, a_h, a_h)
        out = csa_a * csa2_fff
        out = torch.sum(out, -1)
        out = torch.sum(out, -1)
        out_csa = out.contiguous().view(b, c, h, w)
        out_32 = torch.cat([gus_out, out_csa], 1)
        out_32 = self.down(out_32)
        return out_32


class PartialConv(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        self.input_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.mask_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            False,
        )

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, inputt):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        input = inputt[0]
        mask = inputt[1].float().cuda()

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes.bool(), 1.0)
        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes.bool(), 0.0)
        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes.bool(), 0.0)
        out = []
        out.append(output)
        out.append(new_mask)
        return out


class PCconv(pl.LightningModule):
    def __init__(self):
        super(PCconv, self).__init__()
        self.down_128 = ConvDown(64, 128, 4, 2, padding=1, layers=2)
        self.down_64 = ConvDown(128, 256, 4, 2, padding=1)
        self.down_32 = ConvDown(256, 256, 1, 1)
        self.down_16 = ConvDown(512, 512, 4, 2, padding=1, activ=False)
        self.down_8 = ConvDown(512, 512, 4, 2, padding=1, layers=2, activ=False)
        self.down_4 = ConvDown(512, 512, 4, 2, padding=1, layers=3, activ=False)
        self.down = ConvDown(768, 256, 1, 1)
        self.fuse = ConvDown(512, 512, 1, 1)
        self.up = ConvUp(512, 256, 1, 1)
        self.up_128 = ConvUp(512, 64, 1, 1)
        self.up_64 = ConvUp(512, 128, 1, 1)
        self.up_32 = ConvUp(512, 256, 1, 1)
        self.base = BASE(512)
        seuqence_3 = []
        seuqence_5 = []
        seuqence_7 = []
        for i in range(5):
            seuqence_3 += [PCBActiv(256, 256, innorm=True)]
            seuqence_5 += [PCBActiv(256, 256, sample="same-5", innorm=True)]
            seuqence_7 += [PCBActiv(256, 256, sample="same-7", innorm=True)]

        self.cov_3 = nn.Sequential(*seuqence_3)
        self.cov_5 = nn.Sequential(*seuqence_5)
        self.cov_7 = nn.Sequential(*seuqence_7)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, mask):
        mask = cal_feat_mask(mask, 3, 1)
        # input[2]:256 32 32
        b, c, h, w = input[2].size()
        mask_1 = torch.add(torch.neg(mask.float()), 1)
        mask_1 = mask_1.expand(b, c, h, w)

        x_1 = self.activation(input[0])
        x_2 = self.activation(input[1])
        x_3 = self.activation(input[2])
        x_4 = self.activation(input[3])
        x_5 = self.activation(input[4])
        x_6 = self.activation(input[5])
        # Change the shape of each layer and intergrate low-level/high-level features
        x_1 = self.down_128(x_1)
        x_2 = self.down_64(x_2)
        x_3 = self.down_32(x_3)
        x_4 = self.up(x_4, (32, 32))
        x_5 = self.up(x_5, (32, 32))
        x_6 = self.up(x_6, (32, 32))

        # The first three layers are Texture/detail
        # The last three layers are Structure
        x_DE = torch.cat([x_1, x_2, x_3], 1)
        x_ST = torch.cat([x_4, x_5, x_6], 1)

        x_ST = self.down(x_ST)
        x_DE = self.down(x_DE)
        x_ST = [x_ST, mask_1]
        x_DE = [x_DE, mask_1]

        # Multi Scale PConv fill the Details
        x_DE_3 = self.cov_3(x_DE)
        x_DE_5 = self.cov_5(x_DE)
        x_DE_7 = self.cov_7(x_DE)
        x_DE_fuse = torch.cat([x_DE_3[0], x_DE_5[0], x_DE_7[0]], 1)
        x_DE_fi = self.down(x_DE_fuse)

        # Multi Scale PConv fill the Structure
        x_ST_3 = self.cov_3(x_ST)
        x_ST_5 = self.cov_5(x_ST)
        x_ST_7 = self.cov_7(x_ST)
        x_ST_fuse = torch.cat([x_ST_3[0], x_ST_5[0], x_ST_7[0]], 1)
        x_ST_fi = self.down(x_ST_fuse)

        x_cat = torch.cat([x_ST_fi, x_DE_fi], 1)
        x_cat_fuse = self.fuse(x_cat)

        # Feature equalizations
        x_final = self.base(x_cat_fuse)

        # Add back to the input
        x_ST = x_final
        x_DE = x_final
        x_1 = self.up_128(x_DE, (128, 128)) + input[0]
        x_2 = self.up_64(x_DE, (64, 64)) + input[1]
        x_3 = self.up_32(x_DE, (32, 32)) + input[2]
        x_4 = self.down_16(x_ST) + input[3]
        x_5 = self.down_8(x_ST) + input[4]
        x_6 = self.down_4(x_ST) + input[5]

        out = [x_1, x_2, x_3, x_4, x_5, x_6]
        loss = [x_ST_fi, x_DE_fi]
        out_final = [out, loss]
        return out_final




# Define the resnet block
class ResnetBlock(pl.LightningModule):
    def __init__(self, dim, dilation=1):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=3,
                padding=0,
                dilation=dilation,
                bias=False,
            ),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=3,
                padding=0,
                dilation=1,
                bias=False,
            ),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# define the Encoder unit
class UnetSkipConnectionEBlock(pl.LightningModule):
    def __init__(
        self,
        outer_nc,
        inner_nc,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        super(UnetSkipConnectionEBlock, self).__init__()
        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1)

        downrelu = nn.LeakyReLU(0.2, True)

        downnorm = norm_layer(inner_nc, affine=True)
        if outermost:
            down = [downconv]
            model = down
        elif innermost:
            down = [downrelu, downconv]
            model = down
        else:
            down = [downrelu, downconv, downnorm]
            if use_dropout:
                model = down + [nn.Dropout(0.5)]
            else:
                model = down
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Encoder(pl.LightningModule):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        res_num=4,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        super(Encoder, self).__init__()

        # construct unet structure
        Encoder_1 = UnetSkipConnectionEBlock(
            input_nc,
            ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            outermost=True,
        )
        Encoder_2 = UnetSkipConnectionEBlock(
            ngf, ngf * 2, norm_layer=norm_layer, use_dropout=use_dropout
        )
        Encoder_3 = UnetSkipConnectionEBlock(
            ngf * 2, ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout
        )
        Encoder_4 = UnetSkipConnectionEBlock(
            ngf * 4, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout
        )
        Encoder_5 = UnetSkipConnectionEBlock(
            ngf * 8, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout
        )
        Encoder_6 = UnetSkipConnectionEBlock(
            ngf * 8,
            ngf * 8,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            innermost=True,
        )

        blocks = []
        for _ in range(res_num):
            block = ResnetBlock(ngf * 8, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.Encoder_1 = Encoder_1
        self.Encoder_2 = Encoder_2
        self.Encoder_3 = Encoder_3
        self.Encoder_4 = Encoder_4
        self.Encoder_5 = Encoder_5
        self.Encoder_6 = Encoder_6

    def forward(self, input):
        y_1 = self.Encoder_1(input)
        y_2 = self.Encoder_2(y_1)
        y_3 = self.Encoder_3(y_2)
        y_4 = self.Encoder_4(y_3)
        y_5 = self.Encoder_5(y_4)
        y_6 = self.Encoder_6(y_5)
        y_7 = self.middle(y_6)

        return y_1, y_2, y_3, y_4, y_5, y_7


import torch.nn as nn


class UnetSkipConnectionDBlock(pl.LightningModule):
    def __init__(
        self,
        inner_nc,
        outer_nc,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        super(UnetSkipConnectionDBlock, self).__init__()
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)
        upconv = nn.ConvTranspose2d(
            inner_nc, outer_nc, kernel_size=4, stride=2, padding=1
        )
        up = [uprelu, upconv, upnorm]

        if outermost:
            up = [uprelu, upconv, nn.Tanh()]
            model = up
        elif innermost:
            up = [uprelu, upconv, upnorm]
            model = up
        else:
            up = [uprelu, upconv, upnorm]
            model = up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Decoder(pl.LightningModule):
    def __init__(
        self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False
    ):
        super(Decoder, self).__init__()

        # construct unet structure
        Decoder_1 = UnetSkipConnectionDBlock(
            ngf * 8,
            ngf * 8,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            innermost=True,
        )
        Decoder_2 = UnetSkipConnectionDBlock(
            ngf * 16, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout
        )
        Decoder_3 = UnetSkipConnectionDBlock(
            ngf * 16, ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout
        )
        Decoder_4 = UnetSkipConnectionDBlock(
            ngf * 8, ngf * 2, norm_layer=norm_layer, use_dropout=use_dropout
        )
        Decoder_5 = UnetSkipConnectionDBlock(
            ngf * 4, ngf, norm_layer=norm_layer, use_dropout=use_dropout
        )
        Decoder_6 = UnetSkipConnectionDBlock(
            ngf * 2,
            output_nc,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            outermost=True,
        )

        self.Decoder_1 = Decoder_1
        self.Decoder_2 = Decoder_2
        self.Decoder_3 = Decoder_3
        self.Decoder_4 = Decoder_4
        self.Decoder_5 = Decoder_5
        self.Decoder_6 = Decoder_6

    def forward(self, input_1, input_2, input_3, input_4, input_5, input_6):
        y_1 = self.Decoder_1(input_6)
        y_2 = self.Decoder_2(torch.cat([y_1, input_5], 1))
        y_3 = self.Decoder_3(torch.cat([y_2, input_4], 1))
        y_4 = self.Decoder_4(torch.cat([y_3, input_3], 1))
        y_5 = self.Decoder_5(torch.cat([y_4, input_2], 1))
        y_6 = self.Decoder_6(torch.cat([y_5, input_1], 1))
        out = y_6

        return out


class PCblock(pl.LightningModule):
    def __init__(self, stde_list):
        super(PCblock, self).__init__()
        self.pc_block = PCconv()
        innerloss = InnerCos()
        stde_list.append(innerloss)
        loss = [innerloss]
        self.loss = nn.Sequential(*loss)

    def forward(self, input, mask):
        out = self.pc_block(input, mask)
        out = self.loss(out)
        return out


class MEDFEGenerator(pl.LightningModule):
    def __init__(
        self,
        input_nc=4,
        output_nc=3,
        ngf=64,
        norm="batch",
        use_dropout=False,
        stde_list=[],
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        self.netEN = Encoder(
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
        )
        self.netDE = Decoder(
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
        )
        self.netMEDFE = PCblock(stde_list)

    def mask_process(self, mask):
        mask = mask[0][0]
        mask = torch.unsqueeze(mask, 0)
        mask = torch.unsqueeze(mask, 1)
        mask = mask.byte()
        return mask

    def forward(self, images, masks):
        # masks =torch.cat([masks,masks,masks],1)

        fake_p_1, fake_p_2, fake_p_3, fake_p_4, fake_p_5, fake_p_6 = self.netEN(
            torch.cat([images, masks], 1)
        )
        x_out = self.netMEDFE(
            [fake_p_1, fake_p_2, fake_p_3, fake_p_4, fake_p_5, fake_p_6], masks
        )
        self.fake_out = self.netDE(
            x_out[0], x_out[1], x_out[2], x_out[3], x_out[4], x_out[5]
        )

        return self.fake_out
