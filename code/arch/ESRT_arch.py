"""
25-Nov-21
https://github.com/louisfghbvc/Efficient-Transformer-for-Single-Image-Super-Resolution/blob/main/models/models.py
https://github.com/louisfghbvc/Efficient-Transformer-for-Single-Image-Super-Resolution/blob/main/models/et.py
https://github.com/louisfghbvc/Efficient-Transformer-for-Single-Image-Super-Resolution/blob/main/models/hpb.py
https://github.com/louisfghbvc/Efficient-Transformer-for-Single-Image-Super-Resolution/blob/main/models/arfb.py
https://github.com/louisfghbvc/Efficient-Transformer-for-Single-Image-Super-Resolution/blob/main/models/comm.py
https://github.com/louisfghbvc/Efficient-Transformer-for-Single-Image-Super-Resolution/blob/main/models/hfm.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F


# High Filter Module
class HFM(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        
        self.k = k

        self.net = nn.Sequential(
            nn.AvgPool2d(kernel_size = self.k, stride = self.k),
            nn.Upsample(scale_factor = self.k, mode = 'nearest'),
        )

    def forward(self, tL):
        assert tL.shape[2] % self.k == 0, 'h, w must divisible by k'
        return tL - self.net(tL)


def defaultConv(inChannels, outChannels, kernelSize, bias=True):
    return nn.Conv2d(
        inChannels, outChannels, kernelSize,
        padding=(kernelSize//2), bias=bias)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualUnit(nn.Module):
    def __init__(self, inChannel, outChannel, reScale, kernelSize=1, bias=True):
        super().__init__()

        self.reduction = defaultConv(
            inChannel, outChannel//2, kernelSize, bias)
        self.expansion = defaultConv(
            outChannel//2, inChannel, kernelSize, bias)
        self.lamRes = reScale[0]
        self.lamX = reScale[1]

    def forward(self, x):
        res = self.reduction(x)
        res = self.lamRes * self.expansion(res)
        x = self.lamX * x + res

        return x


class ARFB(nn.Module):
    def __init__(self, inChannel, outChannel, reScale):
        super().__init__()
        self.RU1 = ResidualUnit(inChannel, outChannel, reScale)
        self.RU2 = ResidualUnit(inChannel, outChannel, reScale)
        self.conv1 = defaultConv(2*inChannel, 2*outChannel, kernelSize=1)
        self.conv3 = defaultConv(2*inChannel, outChannel, kernelSize=3)
        self.lamRes = reScale[0]
        self.lamX = reScale[1]

    def forward(self, x):

        x_ru1 = self.RU1(x)
        x_ru2 = self.RU2(x_ru1)
        x_ru = torch.cat((x_ru1, x_ru2), 1)
        x_ru = self.conv1(x_ru)
        x_ru = self.conv3(x_ru)
        x_ru = self.lamRes * x_ru
        x = x*self.lamX + x_ru
        return x


class HPB(nn.Module):
    def __init__(self, inChannel, outChannel, reScale):
        super().__init__()
        self.hfm = HFM()
        self.arfb1 = ARFB(inChannel, outChannel, reScale)
        self.arfb2 = ARFB(inChannel, outChannel, reScale)
        self.arfb3 = ARFB(inChannel, outChannel, reScale)
        self.arfbShare = ARFB(inChannel, outChannel, reScale)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.se = SELayer(inChannel)
        self.conv1 = defaultConv(2*inChannel, outChannel, kernelSize=1 )
    def forward(self,x):
        ori = x
        x = self.arfb1(x)
        x = self.hfm(x)
        x = self.arfb2(x)
        x_share = F.interpolate(x,scale_factor=0.5)
        for _ in range(5):
            x_share = self.arfbShare(x_share)
        x_share = self.upsample(x_share)

        x = torch.cat((x_share,x),1)
        x = self.conv1(x)
        x = self.se(x)
        x = self.arfb3(x)
        x = ori+x
        return x
        

class Config():
    lamRes = torch.nn.Parameter(torch.ones(1))
    lamX = torch.nn.Parameter(torch.ones(1))


# LayerNorm
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# TODO: not sure numbers of layer in mlp
class FeedForward(nn.Module):
    def __init__(self, dim, hiddenDim, dropOut=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hiddenDim),
            nn.GELU(),
            nn.Dropout(dropOut),
            nn.Linear(hiddenDim, dim),
            nn.Dropout(dropOut)
        )

    def forward(self, x):
        return self.net(x)

# Efficient Multi-Head Attention
class EMHA(nn.Module):
    def __init__(self, inChannels, splitFactors=4, heads=8):
        super().__init__()
        dimHead = inChannels // (2*heads)

        self.heads = heads
        self.splitFactors = splitFactors
        self.scale = dimHead ** -0.5

        self.reduction = nn.Conv1d(
            in_channels=inChannels, out_channels=inChannels//2, kernel_size=1)
        self.attend = nn.Softmax(dim=-1)
        self.toQKV = nn.Linear(
            inChannels // 2, inChannels // 2 * 3, bias=False)
        self.expansion = nn.Conv1d(
            in_channels=inChannels//2, out_channels=inChannels, kernel_size=1)

    def forward(self, x):
        x = self.reduction(x)
        x = x.transpose(-1, -2)

        qkv = self.toQKV(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        qs, ks, vs = map(lambda t: t.chunk(
            self.splitFactors, dim=2), [q, k, v])

        pool = []
        for qi, ki, vi in zip(qs, ks, vs):
            tmp = torch.matmul(qi, ki.transpose(-1, -2)) * self.scale
            attn = self.attend(tmp)
            out = torch.matmul(attn, vi)
            out = rearrange(out, 'b h n d -> b n (h d)')
            pool.append(out)

        out = torch.cat(tuple(pool), dim=1)
        out = out.transpose(-1, -2)
        out = self.expansion(out)
        return out

class EfficientTransformer(nn.Module):
    def __init__(self, inChannels, mlpDim=256, k=3, splitFactors=4, heads=8, dropOut=0.):
        super().__init__()

        self.k = k
        self.unFold = nn.Unfold(kernel_size=(k, k), padding=1)
        self.norm1 = nn.LayerNorm(inChannels*k*k)
        self.emha = EMHA(inChannels=inChannels*k*k,
                         splitFactors=splitFactors, heads=heads)
        self.norm2 = nn.LayerNorm(inChannels*k*k)
        self.mlp = FeedForward(inChannels*k*k, mlpDim, dropOut=dropOut)

    def forward(self, x):
        _, _, h, w = x.shape
        # b c h w -> b (kkc) (hw)
        x = self.unFold(x)
        x = x.transpose(-2, -1)
        x = self.norm1(x)
        x = x.transpose(-2, -1)
        x = self.emha(x) + x
        x = x.transpose(-2, -1)
        x = self.norm2(x)
        x = self.mlp(x) + x
        x = x.transpose(-2, -1)
        return F.fold(x, output_size=(h, w), kernel_size=(self.k, self.k), padding=(1, 1))


class BackBoneBlock(nn.Module):
    def __init__(self, num, fm, **args):
        super().__init__()
        self.arr = nn.ModuleList([])
        for _ in range(num):
            self.arr.append(fm(**args))

    def forward(self, x):
        for block in self.arr:
            x = block(x)
        return x


class ESRT(nn.Module):
    def __init__(self, hiddenDim=32, mlpDim=128, scaleFactor=2):
        super().__init__()
        self.conv3 = nn.Conv2d(3, hiddenDim,
                               kernel_size=3, padding=1)
        
        lamRes = torch.nn.Parameter(torch.ones(1))
        lamX = torch.nn.Parameter(torch.ones(1))
        self.adaptiveWeight = (lamRes, lamX)

        self.path1 = nn.Sequential(
            BackBoneBlock(3, HPB, inChannel=hiddenDim,
                          outChannel=hiddenDim, reScale=self.adaptiveWeight),
            BackBoneBlock(1, EfficientTransformer,
                          mlpDim=mlpDim, inChannels=hiddenDim),
            nn.Conv2d(hiddenDim, hiddenDim, kernel_size=3, padding=1),
            nn.PixelShuffle(scaleFactor),
            nn.Conv2d(hiddenDim // (scaleFactor**2),
                      3, kernel_size=3, padding=1),
        )

        self.path2 = nn.Sequential(
            nn.PixelShuffle(scaleFactor),
            nn.Conv2d(hiddenDim // (scaleFactor**2),
                      3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.conv3(x)
        x1, x2 = self.path1(x), self.path2(x)
        return x1 + x2

