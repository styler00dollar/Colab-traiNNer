# https://github.com/vanbou/MFRAN/blob/main/src/model/mfran.py
# https://github.com/vanbou/MFRAN/blob/main/src/model/common.py


import torch
import torch.nn as nn

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class MeanShift(nn.Conv2d):
    def __init__(
        self,
        rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040),
        rgb_std=(1.0, 1.0, 1.0),
        sign=-1,
    ):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == "relu":
                    m.append(nn.ReLU(True))
                elif act == "prelu":
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == "relu":
                m.append(nn.ReLU(True))
            elif act == "prelu":
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class CrossChannelAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(CrossChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)  # bs,c,1,1
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y = self.conv(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        return x * y.expand_as(x)


#  Dual-enhanced channel attention
class DECA(nn.Module):
    def __init__(self, n_feats=64, group=8):
        super(DECA, self).__init__()
        self.G = group
        self.channel = n_feats
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(n_feats // (2 * group), n_feats // (2 * group))
        self.cweight = Parameter(torch.zeros(1, n_feats // (2 * group), 1, 1))
        self.cbias = Parameter(torch.ones(1, n_feats // (2 * group), 1, 1))
        self.sweight = Parameter(torch.zeros(1, n_feats // (2 * group), 1, 1))
        self.sbias = Parameter(torch.ones(1, n_feats // (2 * group), 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.eca = CrossChannelAttention()

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)

        # channel_split
        x_path, y_path = x.chunk(2, dim=1)

        # intel-channel attention
        x_channel = self.avg_pool(x_path)
        x_channel = self.cweight * x_channel + self.cweight
        x_channel = x_path * self.sigmoid(x_channel)

        # cross-channel attention
        y_channel = self.eca(y_path)

        # concatenate along channel axis
        out = torch.cat([x_channel, y_channel], dim=1)
        out = out.contiguous().view(b, -1, h, w)

        # channel shuffle
        features = self.channel_shuffle(out, 2)
        return features


# Dilated Residual Attention Block
class DRAB(nn.Module):
    def __init__(
        self, in_feat, out_feat, kernel_size=3, stride=1, padding=1, dilation=1
    ):
        super().__init__()

        self.pad = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(
            in_feat, out_feat, kernel_size, stride, padding=0, bias=True
        )
        self.convAtros = nn.Conv2d(
            in_feat,
            out_feat,
            kernel_size,
            stride,
            padding=1,
            bias=True,
            dilation=dilation,
        )
        self.seca = DECA(in_feat)

    def forward(self, x):
        res = x
        out = self.pad(x)
        out = self.conv(out)
        out = F.relu_(out)
        out = self.convAtros(out)
        out = self.seca(out)
        out += res
        return out


# Fractal Residual Block
class FRB(nn.Module):
    def __init__(self, path, in_feat, out_feat):
        super().__init__()
        self.n_columns = path
        self.columns = nn.ModuleList([nn.ModuleList() for _ in range(path)])
        self.max_depth = 2 ** (path - 1)
        dist = self.max_depth
        self.count = np.zeros([self.max_depth], dtype=int)
        dila = 1
        for col in self.columns:
            for i in range(self.max_depth):
                if (i + 1) % dist == 0:
                    first_block = i + 1 == dist  # first block in this column
                    if first_block:
                        cur_C_in = in_feat
                    else:
                        cur_C_in = out_feat
                    # if i >3:
                    module = DRAB(cur_C_in, out_feat, dilation=dila, padding=dila)
                    dila += 1
                    if dila > 3:
                        dila = 1
                    self.count[i] += 1
                else:
                    module = None
                col.append(module)
            dist //= 2
        joins = [2, 3, 2, 4]

        self.joinTimes = len(joins)
        self.weights = []
        for i in joins:
            self.weights.append(nn.Parameter((torch.ones(i)).cuda()))

        self.joinConv = nn.ModuleList()
        for i in joins:
            self.joinConv.append(default_conv(i * out_feat, out_feat, 1, bias=False))

    # feature fusion with mean
    def mean_join(self, outs, cnt=0):
        outs = torch.stack(outs)  # [n_cols, B, C, H, W]
        outs = outs.mean(dim=0)  # no drop
        return outs

    # feature fusion with 1x1 conv
    def conv_join(self, outs, cnt=0):
        outs = torch.cat(outs, dim=1)  # [n_cols, B, C, H, W]
        return self.joinConv[cnt](outs)

    def forward(self, x):
        outs = [x] * self.n_columns
        cnt = 0
        for i in range(self.max_depth):
            st = self.n_columns - self.count[i]
            cur_outs = []  # outs of current depth

            for c in range(st, self.n_columns):
                cur_in = outs[c]  # current input
                cur_module = self.columns[c][i]  # current module
                cur_outs.append(cur_module(cur_in))

            if len(cur_outs) == 1:
                joined = cur_outs[0]
            else:
                joined = self.conv_join(cur_outs, cnt)
                # joined = self.mean_join(cur_outs, cnt)
                # joined = self.join(cur_outs, cnt)
                cnt = (cnt + 1) % self.joinTimes
            for c in range(st, self.n_columns):
                outs[c] = joined

        outs[-1] += x

        return outs[-1]


class MFRAN(nn.Module):
    def __init__(
        self,
        n_feats=64,
        kernel_size=3,
        scale=2,
        n_blocks=16,
        div=2,
        rgb_range=255,
        n_colors=3,
        path=4,
        conv=default_conv,
    ):
        super().__init__()

        self.n_blocks = n_blocks
        self.div = div

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = nn.ModuleList()
        for i in range(n_blocks):
            modules_body.append(FRB(path, n_feats, n_feats))

        # define tail module
        modules_tail = [
            nn.Conv2d(
                n_feats, n_feats, kernel_size, padding=(kernel_size - 1) // 2, stride=1
            ),
            Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(
                n_feats,
                n_colors,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=1,
            ),
        ]

        self.adjust = nn.Sequential(
            nn.Conv2d(
                (self.n_blocks // self.div + 1) * n_feats,
                n_feats,
                1,
                padding=0,
                stride=1,
            )
        )
        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = x

        sf = []
        for i in range(self.n_blocks // 2):
            x = self.body[i](x)
            sf.append(x)
        sf.append(res)
        res2 = x
        x = torch.cat(sf, 1)
        x = self.adjust(x)
        mid = x
        x = mid

        df = []
        for i in range(self.n_blocks // 2):
            x = self.body[i](x)
            df.append(x)
        df.append(res2)
        res = torch.cat(df, 1)
        x = self.adjust(res)
        mid2 = x
        mid2 += mid

        x = self.tail(mid2)
        x = self.add_mean(x)
        return x
