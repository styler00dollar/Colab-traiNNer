"""
co_mod_gan.py (25-06-21)
https://github.com/zengxianyu/co-mod-gan-pytorch/blob/a34606ec6505146c4cf3fab4cfef9b0e77b15ae4/co_mod_gan.py

stylegan2.py (25-06-21)
https://github.com/zengxianyu/co-mod-gan-pytorch/blob/a34606ec6505146c4cf3fab4cfef9b0e77b15ae4/stylegan2.py
"""
import pytorch_lightning as pl
from arch.cpp.fused_act import FusedLeakyReLU, fused_leaky_relu
from arch.cpp.upfirdn2d import upfirdn2d
from collections import OrderedDict
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F
import functools
import math
import numpy as np
import operator
import pdb
import random
import torch


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input**2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor**2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size**2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size**2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            # layers.append(FusedLeakyReLU(out_channel, bias=bias))
            layers.append(FusedLeakyReLU(out_channel))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


# ----------------------------------------------------------------------------
# Mapping network.
# Transforms the input latent code (z) to the disentangled latent code (w).
# Used in configs B-F (Table 1).


class G_mapping(nn.Module):
    def __init__(
        self,
        latent_size=512,  # Latent vector (Z) dimensionality.
        label_size=0,  # Label dimensionality, 0 if no labels.
        dlatent_size=512,  # Disentangled latent (W) dimensionality.
        dlatent_broadcast=None,  # Output disentangled latent (W) as [minibatch, dlatent_size] or [minibatch, dlatent_broadcast, dlatent_size].
        mapping_layers=8,  # Number of mapping layers.
        mapping_fmaps=512,  # Number of activations in the mapping layers.
        mapping_lrmul=0.01,  # Learning rate multiplier for the mapping layers.
        mapping_nonlinearity="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        normalize_latents=True,  # Normalize latent vectors (Z) before feeding them to the mapping layers?
        **kwargs,
    ):
        assert mapping_nonlinearity == "lrelu"
        assert dlatent_broadcast is None
        super().__init__()
        layers = []

        # Embed labels and concatenate them with latents.
        if label_size:
            raise NotImplementedError

        # Normalize latents.
        if normalize_latents:
            layers.append(("Normalize", PixelNorm()))
        # Mapping layers.
        dim_in = latent_size
        for layer_idx in range(mapping_layers):
            fmaps = dlatent_size if layer_idx == mapping_layers - 1 else mapping_fmaps
            layers.append(
                (
                    "Dense%d" % layer_idx,
                    EqualLinear(
                        dim_in, fmaps, lr_mul=mapping_lrmul, activation="fused_lrelu"
                    ),
                )
            )
            dim_in = fmaps
        # Broadcast.
        if dlatent_broadcast is not None:
            raise NotImplementedError
        self.G_mapping = nn.Sequential(OrderedDict(layers))

    def forward(self, latents_in):
        styles = self.G_mapping(latents_in)
        return styles


# ----------------------------------------------------------------------------
# CoModGAN synthesis network.


class G_synthesis_co_mod_gan(nn.Module):
    def __init__(
        self,
        dlatent_size=512,  # Disentangled latent (W) dimensionality.
        num_channels=3,  # Number of output color channels.
        resolution=512,  # Output resolution.
        fmap_base=16 << 10,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_min=1,  # Minimum number of feature maps in any layer.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        randomize_noise=True,  # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
        architecture="skip",  # Architecture: 'orig', 'skip', 'resnet'.
        nonlinearity="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        resample_kernel=[
            1,
            3,
            3,
            1,
        ],  # Low-pass filter to apply when resampling activations. None = no filtering.
        fused_modconv=True,  # Implement modulated_conv2d_layer() as a single fused op?
        pix2pix=False,
        dropout_rate=0.5,
        cond_mod=True,
        style_mod=True,
        noise_injection=True,
        **kwargs,
    ):

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2**resolution_log2 and resolution >= 4

        def nf(stage):
            return np.clip(
                int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max
            )

        assert architecture in ["skip"]
        assert nonlinearity == "lrelu"
        assert fused_modconv
        assert cond_mod
        assert style_mod
        assert not pix2pix
        assert noise_injection

        super().__init__()
        act = nonlinearity
        self.num_layers = resolution_log2 * 2 - 2
        self.resolution_log2 = resolution_log2

        class E_fromrgb(nn.Module):  # res = 2..resolution_log2
            def __init__(self, res):
                super().__init__()
                self.FromRGB = ConvLayer(
                    num_channels + 1,
                    nf(res - 1),
                    1,
                    blur_kernel=resample_kernel,
                    activate=True,
                )

            def forward(self, data):
                y, E_features = data

                # splitting mask from rgb image
                # y = torch.split(y, [num_channels-1,1], dim=1)[0]
                t = self.FromRGB(y)
                return t, E_features

        class E_block(nn.Module):  # res = 2..resolution_log2
            def __init__(self, res):
                super().__init__()
                self.Conv0 = ConvLayer(
                    nf(res - 1), nf(res - 1), kernel_size=3, activate=True
                )
                self.Conv1_down = ConvLayer(
                    nf(res - 1),
                    nf(res - 2),
                    kernel_size=3,
                    downsample=True,
                    blur_kernel=resample_kernel,
                    activate=True,
                )
                self.res = res

            def forward(self, data):
                x, E_features = data
                x = self.Conv0(x)
                E_features[self.res] = x
                x = self.Conv1_down(x)
                return x, E_features

        class E_block_final(nn.Module):  # res = 2..resolution_log2
            def __init__(self):
                super().__init__()
                self.Conv = ConvLayer(nf(2), nf(1), kernel_size=3, activate=True)
                self.Dense0 = EqualLinear(
                    nf(1) * 4 * 4, nf(1) * 2, activation="fused_lrelu"
                )
                self.dropout = nn.Dropout(dropout_rate)

            def forward(self, data):
                x, E_features = data
                x = self.Conv(x)
                E_features[2] = x
                bsize = x.size(0)
                x = x.view(bsize, -1)
                x = self.Dense0(x)
                x = self.dropout(x)
                return x, E_features

        # Main layers.
        Es = []
        for res in range(resolution_log2, 2, -1):
            if res == resolution_log2:
                Es.append(("%dx%d_0" % (2**res, 2**res), E_fromrgb(res)))
            Es.append(("%dx%d" % (2**res, 2**res), E_block(res)))
        # Final layers.
        Es.append(("4x4", E_block_final()))
        self.E = nn.Sequential(OrderedDict(Es))

        # Single convolution layer with all the bells and whistles.
        # Building blocks for main layers.
        mod_size = 0
        if style_mod:
            mod_size += dlatent_size
        if cond_mod:
            mod_size += nf(1) * 2
        assert mod_size > 0

        def get_mod(latent, x_global):
            mod_vector = []
            if style_mod:
                mod_vector.append(latent)
            if cond_mod:
                mod_vector.append(x_global)
            mod_vector = torch.cat(mod_vector, 1)
            return mod_vector

        class Block(nn.Module):
            def __init__(self, res):
                super().__init__()
                self.res = res
                self.Conv0_up = StyledConv(
                    nf(res - 2),
                    nf(res - 1),
                    kernel_size=3,
                    style_dim=mod_size,
                    upsample=True,
                    blur_kernel=resample_kernel,
                )
                self.Conv1 = StyledConv(
                    nf(res - 1),
                    nf(res - 1),
                    kernel_size=3,
                    style_dim=mod_size,
                    upsample=False,
                )
                self.ToRGB = ToRGB(nf(res - 1), mod_size)

            def forward(self, x, y, dlatents_in, x_global, E_features):
                x_skip = E_features[self.res]
                mod_vector = get_mod(dlatents_in[:, self.res * 2 - 5], x_global)
                x = self.Conv0_up(x, mod_vector)
                x = x + x_skip
                mod_vector = get_mod(dlatents_in[:, self.res * 2 - 4], x_global)
                x = self.Conv1(x, mod_vector)
                mod_vector = get_mod(dlatents_in[:, self.res * 2 - 3], x_global)
                y = self.ToRGB(x, mod_vector, skip=y)
                return x, y

        class Block0(nn.Module):
            def __init__(self):
                super().__init__()
                self.Dense = EqualLinear(
                    nf(1) * 2, nf(1) * 4 * 4, activation="fused_lrelu"
                )
                self.Conv = StyledConv(nf(1), nf(1), kernel_size=3, style_dim=mod_size)
                self.ToRGB = ToRGB(nf(1), style_dim=mod_size, upsample=False)

            def forward(self, x, dlatents_in, x_global):
                x = self.Dense(x)
                x = x.view(-1, nf(1), 4, 4)
                mod_vector = get_mod(dlatents_in[:, 0], x_global)
                x = self.Conv(x, mod_vector)
                mod_vector = get_mod(dlatents_in[:, 1], x_global)
                y = self.ToRGB(x, mod_vector)
                return x, y

        # Early layers.
        self.G_4x4 = Block0()
        # Main layers.
        for res in range(3, resolution_log2 + 1):
            setattr(self, "G_%dx%d" % (2**res, 2**res), Block(res))

    # def forward(self, images_in, masks_in, dlatents_in):
    def forward(self, images_in, dlatents_in):
        # y = torch.cat([masks_in - 0.5, images_in * masks_in], 1)
        y = images_in
        E_features = {}
        x_global, E_features = self.E((y, E_features))
        x = x_global
        x, y = self.G_4x4(x, dlatents_in, x_global)
        for res in range(3, self.resolution_log2 + 1):
            block = getattr(self, "G_%dx%d" % (2**res, 2**res))
            x, y = block(x, y, dlatents_in, x_global, E_features)
        # images_out = y * (1 - masks_in) + images_in * masks_in
        # return images_out
        return y


# ----------------------------------------------------------------------------
# Main generator network.
# Composed of two sub-networks (mapping and synthesis) that are defined below.
# Used in configs B-F (Table 1).


class Generator(pl.LightningModule):
    def __init__(self, **kwargs):  # Arguments for sub-networks (mapping and synthesis).
        super().__init__()
        self.G_mapping = G_mapping(**kwargs)
        self.G_synthesis = G_synthesis_co_mod_gan(**kwargs)

    def forward(
        self,
        images_in,
        return_latents=False,
        inject_index=None,
        truncation=None,
        truncation_mean=None,
        # truncation=0.5,
        # truncation_mean=None,
    ):
        random = torch.randn(1, 512)
        random = random.to(device=self.device)
        latents_in = [random]  # needs to be a list, must match batch_size

        assert isinstance(latents_in, list)
        dlatents_in = [self.G_mapping(s) for s in latents_in]
        if truncation is not None:
            dlatents_t = []
            for style in dlatents_in:
                dlatents_t.append(
                    truncation_mean + truncation * (style - truncation_mean)
                )
            dlatents_in = dlatents_t
        if len(dlatents_in) < 2:
            inject_index = self.G_synthesis.num_layers
            if dlatents_in[0].ndim < 3:
                dlatent = dlatents_in[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                dlatent = dlatents_in[0]
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.G_synthesis.num_layers - 1)
            dlatent = dlatents_in[0].unsqueeze(1).repeat(1, inject_index, 1)
            dlatent2 = (
                dlatents_in[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)
            )

            dlatent = torch.cat([dlatent, dlatent2], 1)
        output = self.G_synthesis(images_in, dlatent)
        return output
