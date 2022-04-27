"""
lightweight_gan.py (31-jul-21)
https://github.com/lucidrains/lightweight-gan/blob/b7c34d587d029177ddc641f42b2604506352dfb2/lightweight_gan/lightweight_gan.py
"""
import os

# import multiprocessing
from random import random
import math
from math import log2, floor
import torch
from torch.cuda.amp import autocast, GradScaler
from torch import nn, einsum
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from functools import partial
import kornia
from einops import rearrange


def exists(val):
    return val is not None


def is_power_of_two(val):
    return log2(val).is_integer()


def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException


def gradient_accumulate_contexts(gradient_accumulate_every, is_ddp, ddps):
    if is_ddp:
        num_no_syncs = gradient_accumulate_every - 1
        head = [combine_contexts(map(lambda ddp: ddp.no_sync, ddps))] * num_no_syncs
        tail = [null_context]
        contexts = head + tail
    else:
        contexts = [null_context] * gradient_accumulate_every

    for context in contexts:
        with context():
            yield


def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)


def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (
        torch.sin(val * omega) / so
    ).unsqueeze(1) * high
    return res


def safe_div(n, d):
    try:
        res = n / d
    except ZeroDivisionError:
        prefix = "" if int(n >= 0) else "-"
        res = float(f"{prefix}inf")
    return res


# helper classes


class NanException(Exception):
    pass


class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else=lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob

    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)


ChanNorm = partial(nn.InstanceNorm2d, affine=True)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = ChanNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class SumBranches(nn.Module):
    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        return sum(map(lambda fn: fn(x), self.branches))


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer("f", f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return kornia.filters.filter2d(x, f, normalized=True)


# attention


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding=0, stride=1, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                dim_in,
                dim_in,
                kernel_size=kernel_size,
                padding=padding,
                groups=dim_in,
                stride=stride,
                bias=bias,
            ),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        return self.net(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.nonlin = nn.GELU()
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, 3, padding=1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]
        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim=1))
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> (b h) (x y) c", h=h), (q, k, v)
        )

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        q = q * self.scale

        context = einsum("b n d, b n e -> b d e", k, v)
        out = einsum("b n d, b d e -> b n e", q, context)
        out = rearrange(out, "(b h) (x y) d -> b (h d) x y", h=h, x=x, y=y)

        out = self.nonlin(out)
        return self.to_out(out)


# modifiable global variables

norm_class = nn.BatchNorm2d


def upsample(scale_factor=2):
    return nn.Upsample(scale_factor=scale_factor)


# squeeze excitation classes

# global context network
# https://arxiv.org/abs/2012.13375
# similar to squeeze-excite, but with a simplified attention pooling and a subsequent layer norm


class GlobalContext(nn.Module):
    def __init__(self, *, chan_in, chan_out):
        super().__init__()
        self.to_k = nn.Conv2d(chan_in, 1, 1)
        chan_intermediate = max(3, chan_out // 2)

        self.net = nn.Sequential(
            nn.Conv2d(chan_in, chan_intermediate, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(chan_intermediate, chan_out, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        context = self.to_k(x)
        context = context.flatten(2).softmax(dim=-1)
        out = einsum("b i n, b c n -> b c i", context, x.flatten(2))
        out = out.unsqueeze(-1)
        return self.net(out)


# frequency channel attention
# https://arxiv.org/abs/2012.11879


def get_1d_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i + 0.5) / L) / math.sqrt(L)
    return result * (1 if freq == 0 else math.sqrt(2))


def get_dct_weights(width, channel, fidx_u, fidx_v):
    dct_weights = torch.zeros(1, channel, width, width)
    c_part = channel // len(fidx_u)

    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for x in range(width):
            for y in range(width):
                coor_value = get_1d_dct(x, u_x, width) * get_1d_dct(y, v_y, width)
                dct_weights[:, i * c_part : (i + 1) * c_part, x, y] = coor_value

    return dct_weights


class FCANet(nn.Module):
    def __init__(self, *, chan_in, chan_out, reduction=4, width):
        super().__init__()

        freq_w, freq_h = ([0] * 8), list(
            range(8)
        )  # in paper, it seems 16 frequencies was ideal
        dct_weights = get_dct_weights(
            width, chan_in, [*freq_w, *freq_h], [*freq_h, *freq_w]
        )
        self.register_buffer("dct_weights", dct_weights)

        chan_intermediate = max(3, chan_out // reduction)

        self.net = nn.Sequential(
            nn.Conv2d(chan_in, chan_intermediate, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(chan_intermediate, chan_out, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = reduce(
            x * self.dct_weights, "b c (h h1) (w w1) -> b c h1 w1", "sum", h1=1, w1=1
        )
        return self.net(x)


# generative adversarial network
import pytorch_lightning as pl


class Generator(pl.LightningModule):
    def __init__(
        self,
        *,
        image_size=512,
        latent_dim=256,
        fmap_max=512,
        fmap_inverse_coef=12,
        transparent=False,
        greyscale=False,
        attn_res_layers=[],
        freq_chan_attn=False,
    ):
        super().__init__()
        resolution = log2(image_size)
        assert is_power_of_two(image_size), "image size must be a power of 2"

        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3

        # fmap_max = default(fmap_max, latent_dim)

        self.initial_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim * 2, 4),
            norm_class(latent_dim * 2),
            nn.GLU(dim=1),
        )

        num_layers = int(resolution) - 2
        features = list(
            map(lambda n: (n, 2 ** (fmap_inverse_coef - n)), range(2, num_layers + 2))
        )
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))
        features = list(map(lambda n: 3 if n[0] >= 8 else n[1], features))
        features = [latent_dim, *features]

        in_out_features = list(zip(features[:-1], features[1:]))

        self.res_layers = range(2, num_layers + 2)
        self.layers = nn.ModuleList([])

        self.res_to_feature_map = dict(zip(self.res_layers, in_out_features))

        self.sle_map = ((3, 7), (4, 8), (5, 9), (6, 10))
        self.sle_map = list(
            filter(lambda t: t[0] <= resolution and t[1] <= resolution, self.sle_map)
        )
        self.sle_map = dict(self.sle_map)

        self.num_layers_spatial_res = 1

        for (res, (chan_in, chan_out)) in zip(self.res_layers, in_out_features):
            image_width = 2**res

            attn = None
            if image_width in attn_res_layers:
                attn = PreNorm(chan_in, LinearAttention(chan_in))

            sle = None
            if res in self.sle_map:
                residual_layer = self.sle_map[res]
                sle_chan_out = self.res_to_feature_map[residual_layer - 1][-1]

                if freq_chan_attn:
                    sle = FCANet(
                        chan_in=chan_out, chan_out=sle_chan_out, width=2 ** (res + 1)
                    )
                else:
                    sle = GlobalContext(chan_in=chan_out, chan_out=sle_chan_out)

            layer = nn.ModuleList(
                [
                    nn.Sequential(
                        upsample(),
                        Blur(),
                        nn.Conv2d(chan_in, chan_out * 2, 3, padding=1),
                        norm_class(chan_out * 2),
                        nn.GLU(dim=1),
                    ),
                    sle,
                    attn,
                ]
            )
            self.layers.append(layer)

        self.image_size = image_size
        self.cat_conv = nn.ModuleList([])
        self.first_conv = nn.ModuleList([])
        self.concat_conv = nn.ModuleList([])

        # conv layers to process input image for cat operation
        # input is 4 channel image
        if self.image_size == 512:
            self.first_conv.append(torch.nn.Conv2d(4, 32, kernel_size=1, stride=2))
        elif self.image_size == 1024:
            self.first_conv.append(torch.nn.Conv2d(4, 32, kernel_size=1, stride=4))

        # creating conv list
        for f in [32, 64, 128, 256]:
            first_conv = nn.Conv2d(f, f * 2, kernel_size=3, padding=1, stride=2)
            self.first_conv.append(first_conv)

        # conv after loop conv
        self.first_conv.append(torch.nn.Conv2d(512, 512, kernel_size=1, stride=2))
        self.first_conv.append(torch.nn.Conv2d(512, 256, kernel_size=1, stride=2))
        self.first_conv.append(torch.nn.Conv2d(256, 256, kernel_size=1, stride=2))
        self.first_conv.append(torch.nn.Conv2d(256, 256, kernel_size=1, stride=2))
        self.first_conv.append(torch.nn.Conv2d(256, 256, kernel_size=1, stride=2))

        # merge conv for random
        self.random_conv = torch.nn.Conv2d(512, 256, kernel_size=1)

        # concat conv at the end
        for f in [512, 512, 256, 128, 64, 32]:
            concat_conv = nn.Conv2d(f * 2, f, kernel_size=3, padding=1)
            self.concat_conv.append(concat_conv)

        if self.image_size == 1024:
            self.trasnpose_conv = torch.nn.ConvTranspose2d(
                3, 3, 3, stride=2, padding=1, output_padding=1
            )

        # the final conv just does [1, 3, 1024, 1024] -> [1, 3, 1024, 1024] or [1, 3, 512, 512] -> [1, 3, 512, 512]
        self.out_conv = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        # original has random input of [1, 256] and that gets into [1, 256, 1, 1]
        y = torch.rand(x.shape[0], 256, 1, 1).to(self.device)
        # y = rearrange(x, 'b c -> b c () ()')

        # conv the image to make concat later
        first = []

        for i, f in enumerate(self.first_conv):
            result = f(x)
            first.append(result)
            x = result

        # concat with random array and conv to original dimension
        x = torch.cat([x, y], dim=1)
        x = self.random_conv(x)

        x = self.initial_conv(x)
        x = F.normalize(x, dim=1)
        residuals = dict()

        count = 0
        for (res, (up, sle, attn)) in zip(self.res_layers, self.layers):

            if exists(attn):
                x = attn(x) + x

            x = up(x)

            if exists(sle):
                out_res = self.sle_map[res]
                residual = sle(x)
                residuals[out_res] = residual

            next_res = res + 1
            if next_res in residuals:
                x = x * residuals[next_res]

            # concat with conv picture and conv
            # stopping once final picture is reached
            if (
                not (x.shape[1] == 3 and x.shape[2] == 512 and x.shape[3] == 512)
                and self.image_size == 512
            ):
                x = torch.cat([x, first[len(first) - count - 5]], dim=1)
                x = self.concat_conv[count](x)
                count += 1

            elif (
                not (x.shape[1] == 3 and x.shape[2] == 512 and x.shape[3] == 512)
                and self.image_size == 1024
            ):
                x = torch.cat([x, first[len(first) - count - 5]], dim=1)
                x = self.concat_conv[count](x)
                count += 1
            elif (
                x.shape[1] == 3 and x.shape[2] == 512 and x.shape[3] == 512
            ) and self.image_size == 1024:
                # rgb image size 512 -> 1024 with transposeconv
                x = self.trasnpose_conv(x)
                break

        return self.out_conv(x)


class SimpleFontGenerator512(pl.LightningModule):
    def __init__(
        self,
        *,
        image_size=512,
        latent_dim=256,
        fmap_max=512,
        fmap_inverse_coef=12,
        transparent=False,
        greyscale=False,
        attn_res_layers=[],
        freq_chan_attn=False,
    ):
        super().__init__()
        resolution = log2(image_size)
        assert is_power_of_two(image_size), "image size must be a power of 2"

        if transparent:
            self.init_channel = 4
        elif greyscale:
            self.init_channel = 1
        else:
            self.init_channel = 3

        # fmap_max = default(fmap_max, latent_dim)

        self.initial_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim * 2, 4),
            norm_class(latent_dim * 2),
            nn.GLU(dim=1),
        )

        num_layers = int(resolution) - 2
        features = list(
            map(lambda n: (n, 2 ** (fmap_inverse_coef - n)), range(2, num_layers + 2))
        )
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))
        features = list(map(lambda n: 3 if n[0] >= 8 else n[1], features))
        features = [latent_dim, *features]

        # in_out_features = list(zip(features[:-1], features[1:]))
        in_out_features = [
            (256, 512),
            (512, 512),
            (512, 256),
            (256, 128),
            (128, 64),
            (64, 32),
            (32, self.init_channel),
        ]

        self.res_layers = range(2, num_layers + 2)
        self.layers = nn.ModuleList([])

        self.res_to_feature_map = dict(zip(self.res_layers, in_out_features))

        self.sle_map = ((3, 7), (4, 8), (5, 9), (6, 10))
        self.sle_map = list(
            filter(lambda t: t[0] <= resolution and t[1] <= resolution, self.sle_map)
        )
        self.sle_map = dict(self.sle_map)

        self.num_layers_spatial_res = 1

        for (res, (chan_in, chan_out)) in zip(self.res_layers, in_out_features):
            # hotfix, output should have same amount of output channels
            # if chan_out == 3:
            #  chan_out = self.init_channel

            image_width = 2**res

            attn = None
            if image_width in attn_res_layers:
                attn = PreNorm(chan_in, LinearAttention(chan_in))

            sle = None
            if res in self.sle_map:
                residual_layer = self.sle_map[res]
                sle_chan_out = self.res_to_feature_map[residual_layer - 1][-1]

                if freq_chan_attn:
                    sle = FCANet(
                        chan_in=chan_out, chan_out=sle_chan_out, width=2 ** (res + 1)
                    )
                else:
                    sle = GlobalContext(chan_in=chan_out, chan_out=sle_chan_out)

            layer = nn.ModuleList(
                [
                    nn.Sequential(
                        upsample(),
                        Blur(),
                        nn.Conv2d(chan_in, chan_out * 2, 3, padding=1),
                        norm_class(chan_out * 2),
                        nn.GLU(dim=1),
                    ),
                    sle,
                    attn,
                ]
            )
            self.layers.append(layer)

        # torch.Size([1, 128, 64, 64])
        self.image_size = image_size
        self.cat_conv = nn.ModuleList([])
        self.first_conv = nn.ModuleList([])
        self.concat_conv = nn.ModuleList([])

        # creating conv list
        for f in [128, 256]:
            # for f in [128,256]:
            first_conv = nn.Conv2d(f, f * 2, kernel_size=3, padding=1, stride=2)
            self.first_conv.append(first_conv)

        # conv after loop conv
        self.first_conv.append(torch.nn.Conv2d(512, 512, kernel_size=1, stride=2))
        self.first_conv.append(torch.nn.Conv2d(512, 256, kernel_size=1, stride=2))
        self.first_conv.append(torch.nn.Conv2d(256, 256, kernel_size=1, stride=2))
        self.first_conv.append(torch.nn.Conv2d(256, 256, kernel_size=1, stride=2))
        self.first_conv.append(torch.nn.Conv2d(256, 256, kernel_size=1, stride=2))

        # merge conv for random
        self.random_conv = torch.nn.Conv2d(512, 256, kernel_size=1)

        # concat conv at the end
        for f in [512, 512, 256, 128, 64, 32]:
            concat_conv = nn.Conv2d(f * 2, f, kernel_size=3, padding=1)
            self.concat_conv.append(concat_conv)

        # if self.image_size == 1024:
        #  self.trasnpose_conv = torch.nn.ConvTranspose2d(3, 3, 3, stride=2, padding=1, output_padding=1)

        # the final conv just does [1, 3, 1024, 1024] -> [1, 3, 1024, 1024] or [1, 3, 512, 512] -> [1, 3, 512, 512]
        self.out_conv = nn.Conv2d(self.init_channel, self.init_channel, 3, padding=1)

        # font conv (up)
        # torch.Size([1, 128, 64, 64])
        # input is 64px image
        self.font_conv = nn.ModuleList([])
        # self.font_conv.append(nn.Conv2d(3, 128, 3, padding = 1))
        self.init_conv = nn.Conv2d(self.init_channel, 128, 3, padding=1)

        for i in [128, 64]:
            # vlt kernel_size fix
            self.font_conv.append(
                nn.ConvTranspose2d(
                    i, int(i / 2), stride=2, kernel_size=3, padding=1, output_padding=1
                )
            )

        self.font_conv.append(nn.Conv2d(32, self.init_channel, 3, padding=1))

        # todo, probably better way of doing up
        # todo: shape check
        # for i in [128, 64]:
        #  if freq_chan_attn:
        #      sle = FCANet(
        #          chan_in = chan_out,
        #          chan_out = sle_chan_out,
        #          width = 2 ** (res + 1)
        #      )
        #  else:
        #      sle = GlobalContext(
        #          chan_in = chan_out,
        #          chan_out = sle_chan_out
        #      )
        #  attn = PreNorm(chan_in, LinearAttention(chan_in))
        #  nn.ModuleList([
        #          nn.Sequential(
        #              upsample(),
        #              Blur(),
        #              nn.Conv2d(i, int(i/2) * 2, 3, padding = 1),
        #              norm_class(int(i/2) * 2),
        #              nn.GLU(dim = 1)
        #          ),
        #          sle,
        #          attn
        #      ])

        # self.m2 = nn.Conv2d(32, 3, 3, padding = 1)

    def forward(self, x):
        # original has random input of [1, 256] and that gets into [1, 256, 1, 1]

        y = torch.rand(x.shape[0], 256, 1, 1).to(self.device)
        # y = rearrange(x, 'b c -> b c () ()')

        # conv the image to make concat later
        first = []
        second = []

        x = self.init_conv(x)
        z = zz = x  # creating a copy

        # font conv up
        for i, f in enumerate(self.font_conv):
            result = f(x)
            second.append(result)
            x = result

        # font conv down
        for i, f in enumerate(self.first_conv):
            result = f(z)
            first.append(result)
            z = result

        # concat results into one list
        # first = first + second
        # [::-1] means to mirror the list
        first = second[::-1] + list(zz.unsqueeze(0)) + first

        # concat with random array and conv to original dimension
        x = torch.cat([z, y], dim=1)
        x = self.random_conv(x)

        x = self.initial_conv(x)
        x = F.normalize(x, dim=1)
        residuals = dict()

        # x = z

        count = 0
        for (res, (up, sle, attn)) in zip(self.res_layers, self.layers):
            if exists(attn):
                x = attn(x) + x
            x = up(x)  # 32 -> 3 channels in the end
            if exists(sle):
                out_res = self.sle_map[res]
                residual = sle(x)
                residuals[out_res] = residual
            next_res = res + 1
            if next_res in residuals:
                x = x * residuals[next_res]

            # concat with conv picture and conv
            # stopping once final picture is reached
            if (
                not (
                    x.shape[1] == self.init_channel
                    and x.shape[2] == 512
                    and x.shape[3] == 512
                )
                and self.image_size == 512
            ):
                x = torch.cat([x, first[len(first) - count - 5]], dim=1)
                x = self.concat_conv[count](x)
                count += 1

            elif (
                not (
                    x.shape[1] == self.init_channel
                    and x.shape[2] == 512
                    and x.shape[3] == 512
                )
                and self.image_size == 1024
            ):
                x = torch.cat([x, first[len(first) - count - 5]], dim=1)
                x = self.concat_conv[count](x)
                count += 1
            elif (
                x.shape[1] == self.init_channel
                and x.shape[2] == 512
                and x.shape[3] == 512
            ) and self.image_size == 1024:
                # rgb image size 512 -> 1024 with transposeconv
                x = self.trasnpose_conv(x)
                break

        return self.out_conv(x)


class SimpleFontGenerator256(pl.LightningModule):
    def __init__(
        self,
        *,
        image_size=256,
        latent_dim=256,
        fmap_max=512,
        fmap_inverse_coef=12,
        transparent=False,
        greyscale=False,
        attn_res_layers=[],
        freq_chan_attn=False,
    ):
        super().__init__()
        resolution = log2(image_size)
        assert is_power_of_two(image_size), "image size must be a power of 2"

        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3

        # fmap_max = default(fmap_max, latent_dim)

        self.initial_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim * 2, 4),
            norm_class(latent_dim * 2),
            nn.GLU(dim=1),
        )

        num_layers = int(resolution) - 2
        features = list(
            map(lambda n: (n, 2 ** (fmap_inverse_coef - n)), range(2, num_layers + 2))
        )
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))
        features = list(map(lambda n: 3 if n[0] >= 8 else n[1], features))
        features = [latent_dim, *features]

        in_out_features = list(zip(features[:-1], features[1:]))
        # in_out_features = [(256, 512), (512, 256), (256, 128), (128, 64), (64, 32)]

        self.res_layers = range(2, num_layers + 2)  # range(2, 8)
        # self.res_layers = range(2, 7)

        self.layers = nn.ModuleList([])

        self.res_to_feature_map = dict(zip(self.res_layers, in_out_features))

        # 512, 128 / 256, 64 / 256, 64 / 128, 32
        self.sle_map = ((3, 7), (4, 8), (5, 9), (6, 10))
        self.sle_map = list(
            filter(lambda t: t[0] <= resolution and t[1] <= resolution, self.sle_map)
        )
        self.sle_map = dict(self.sle_map)

        self.num_layers_spatial_res = 1

        for (res, (chan_in, chan_out)) in zip(self.res_layers, in_out_features):

            image_width = 2**res

            attn = None
            if image_width in attn_res_layers:
                attn = PreNorm(chan_in, LinearAttention(chan_in))

            sle = None
            if res in self.sle_map:
                residual_layer = self.sle_map[res]
                sle_chan_out = self.res_to_feature_map[residual_layer - 1][-1]

                if freq_chan_attn:
                    sle = FCANet(
                        chan_in=chan_out, chan_out=sle_chan_out, width=2 ** (res + 1)
                    )
                else:
                    sle = GlobalContext(chan_in=chan_out, chan_out=sle_chan_out)

            layer = nn.ModuleList(
                [
                    nn.Sequential(
                        upsample(),
                        Blur(),
                        nn.Conv2d(chan_in, chan_out * 2, 3, padding=1),
                        norm_class(chan_out * 2),
                        nn.GLU(dim=1),
                    ),
                    sle,
                    attn,
                ]
            )
            self.layers.append(layer)

        # torch.Size([1, 128, 64, 64])
        self.image_size = image_size
        self.cat_conv = nn.ModuleList([])
        self.first_conv = nn.ModuleList([])
        self.concat_conv = nn.ModuleList([])

        # creating conv list
        for f in [256]:
            # for f in [128,256]:
            first_conv = nn.Conv2d(f, f * 2, kernel_size=3, padding=1, stride=2)
            self.first_conv.append(first_conv)

        # conv after loop conv
        self.first_conv.append(torch.nn.Conv2d(512, 512, kernel_size=1, stride=2))
        self.first_conv.append(torch.nn.Conv2d(512, 256, kernel_size=1, stride=2))
        self.first_conv.append(torch.nn.Conv2d(256, 256, kernel_size=1, stride=2))
        self.first_conv.append(torch.nn.Conv2d(256, 256, kernel_size=1, stride=2))

        # merge conv for random
        self.random_conv = torch.nn.Conv2d(512, 256, kernel_size=1)

        # concat conv at the end
        for f in [512, 512, 256, 128, 64, 32]:
            concat_conv = nn.Conv2d(f * 2, f, kernel_size=3, padding=1)
            self.concat_conv.append(concat_conv)

        # final conv [1, 32, 256, 256] -> [1, 3, 256, 256]

        # font conv (up)
        # torch.Size([1, 128, 32, 32])
        # input is 32px image
        self.font_conv = nn.ModuleList([])
        self.init_conv = nn.Conv2d(3, 256, 3, padding=1)

        for i in [256, 128]:
            self.font_conv.append(
                nn.ConvTranspose2d(
                    i, int(i / 2), stride=2, kernel_size=3, padding=1, output_padding=1
                )
            )

        self.out_conv = nn.Conv2d(32, 3, 3, padding=1)
        # self.out_conv = nn.ConvTranspose2d(32,3, stride=2, kernel_size = 3, padding=1, output_padding=1)

    def forward(self, x):
        # original has random input of [1, 256] and that gets into [1, 256, 1, 1]

        y = torch.rand(x.shape[0], 256, 1, 1).to(self.device)
        # y = rearrange(x, 'b c -> b c () ()')

        # conv the image to make concat later
        first = []
        second = []
        x = self.init_conv(x)
        z = zz = x  # creating a copy

        # font conv up
        for i, f in enumerate(self.font_conv):
            result = f(x)
            second.append(result)
            x = result

        # font conv down
        for i, f in enumerate(self.first_conv):
            result = f(z)
            first.append(result)
            z = result

        # concat results into one list
        # first = first + second
        # [::-1] means to mirror the list
        first = second[::-1] + list(zz.unsqueeze(0)) + first

        # concat with random array and conv to original dimension
        x = torch.cat([z, y], dim=1)
        x = self.random_conv(x)

        x = self.initial_conv(x)
        x = F.normalize(x, dim=1)
        residuals = dict()

        count = 0
        for (res, (up, sle, attn)) in zip(self.res_layers, self.layers):
            if exists(attn):
                x = attn(x) + x

            x = up(x)

            if exists(sle):
                out_res = self.sle_map[res]
                residual = sle(x)
                residuals[out_res] = residual

            next_res = res + 1
            if next_res in residuals:
                x = x * residuals[next_res]

            # concat with conv picture and conv
            # stopping once final picture is reached
            if not (x.shape[2] == 256 and x.shape[3] == 256):
                print(x.shape, first[len(first) - count - 4].shape)
                x = torch.cat([x, first[len(first) - count - 4]], dim=1)
                x = self.concat_conv[count](x)
                count += 1

        return self.out_conv(x)
