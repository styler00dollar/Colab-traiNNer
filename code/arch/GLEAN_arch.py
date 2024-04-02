"""
# glean
glean_styleganv2.py 28-5-20
https://github.com/open-mmlab/mmediting/blob/f0b2ab423e80b68c281d9e509430856213c08b95/mmedit/models/backbones/sr_backbones/glean_styleganv2.py

rrdb_net.py 28-5-20
https://github.com/open-mmlab/mmediting/blob/f0b2ab423e80b68c281d9e509430856213c08b95/mmedit/models/backbones/sr_backbones/rrdb_net.py

sr_backbone_utils.py 28-5-20
https://github.com/open-mmlab/mmediting/blob/25410895914edc5938f526fc41b1776a36ac1b51/mmedit/models/common/sr_backbone_utils.py

builder.py 28-5-20
https://github.com/open-mmlab/mmediting/blob/25410895914edc5938f526fc41b1776a36ac1b51/mmedit/models/builder.py#L7

upsample.py 28-5-20
https://github.com/open-mmlab/mmediting/blob/25410895914edc5938f526fc41b1776a36ac1b51/mmedit/models/common/upsample.py

registry.py 28-5-20
https://github.com/open-mmlab/mmediting/blob/f3c9dd6d0fdb2ac3a153c2498495944ba33be6ac/mmedit/models/registry.py


# stylegan
generator_discriminator.py 28-5-20
https://github.com/open-mmlab/mmediting/blob/f0b2ab423e80b68c281d9e509430856213c08b95/mmedit/models/components/stylegan2/generator_discriminator.py

common.py 28-5-20
https://github.com/open-mmlab/mmediting/blob/f0b2ab423e80b68c281d9e509430856213c08b95/mmedit/models/components/stylegan2/common.py

modules.py 28-5-20
https://github.com/open-mmlab/mmediting/blob/f0b2ab423e80b68c281d9e509430856213c08b95/mmedit/models/components/stylegan2/modules.py
"""

from mmcv.utils import Registry

MODELS = Registry("model")
BACKBONES = Registry("backbone")
COMPONENTS = Registry("component")
# COMPONENTS = Registry('StyleGANv2Generator')
LOSSES = Registry("loss")

import math
from copy import deepcopy
from functools import partial

import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.activation import build_activation_layer
from mmcv.cnn.utils import normal_init
from mmcv.ops.fused_bias_leakyrelu import FusedBiasLeakyReLU, fused_bias_leakyrelu
from mmcv.ops.upfirdn2d import upfirdn2d
from torch.nn.init import _calculate_correct_fan


def pixel_norm(x, eps=1e-6):
    """Pixel Normalization.

    This normalization is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    Args:
        x (torch.Tensor): Tensor to be normalized.
        eps (float, optional): Epsilon to avoid divising zero.
            Defaults to 1e-6.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    if torch.__version__ >= "1.7.0":
        norm = torch.linalg.norm(x, ord=2, dim=1, keepdim=True)
    # support older pytorch version
    else:
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
    norm = norm / torch.sqrt(torch.tensor(x.shape[1]).to(x))

    return x / (norm + eps)


class PixelNorm(nn.Module):
    """Pixel Normalization.

    This module is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    Args:
        eps (float, optional): Epsilon value. Defaults to 1e-6.
    """

    _abbr_ = "pn"

    def __init__(self, in_channels=None, eps=1e-6):
        super(PixelNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        return pixel_norm(x, self.eps)


class EqualizedLR:
    r"""Equalized Learning Rate.

    This trick is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    The general idea is to dynamically rescale the weight in training instead
    of in initializing so that the variance of the responses in each layer is
    guaranteed with some statistical properties.

    Note that this function is always combined with a convolution module which
    is initialized with :math:`\mathcal{N}(0, 1)`.

    Args:
        name (str | optional): The name of weights. Defaults to 'weight'.
        mode (str, optional): The mode of computing ``fan`` which is the
            same as ``kaiming_init`` in pytorch. You can choose one from
            ['fan_in', 'fan_out']. Defaults to 'fan_in'.
    """

    def __init__(self, name="weight", gain=2**0.5, mode="fan_in", lr_mul=1.0):
        self.name = name
        self.mode = mode
        self.gain = gain
        self.lr_mul = lr_mul

    def compute_weight(self, module):
        """Compute weight with equalized learning rate.

        Args:
            module (nn.Module): A module that is wrapped with equalized lr.

        Returns:
            torch.Tensor: Updated weight.
        """
        weight = getattr(module, self.name + "_orig")
        if weight.ndim == 5:
            # weight in shape of [b, out, in, k, k]
            fan = _calculate_correct_fan(weight[0], self.mode)
        else:
            assert weight.ndim <= 4
            fan = _calculate_correct_fan(weight, self.mode)
        weight = (
            weight
            * torch.tensor(self.gain, device=weight.device)
            * torch.sqrt(torch.tensor(1.0 / fan, device=weight.device))
            * self.lr_mul
        )

        return weight

    def __call__(self, module, inputs):
        """Standard interface for forward pre hooks."""
        setattr(module, self.name, self.compute_weight(module))

    @staticmethod
    def apply(module, name, gain=2**0.5, mode="fan_in", lr_mul=1.0):
        """Apply function.

        This function is to register an equalized learning rate hook in an
        ``nn.Module``.

        Args:
            module (nn.Module): Module to be wrapped.
            name (str | optional): The name of weights. Defaults to 'weight'.
            mode (str, optional): The mode of computing ``fan`` which is the
                same as ``kaiming_init`` in pytorch. You can choose one from
                ['fan_in', 'fan_out']. Defaults to 'fan_in'.

        Returns:
            nn.Module: Module that is registered with equalized lr hook.
        """
        # sanity check for duplicated hooks.
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, EqualizedLR):
                raise RuntimeError(
                    "Cannot register two equalized_lr hooks on the same "
                    f"parameter {name} in {module} module."
                )

        fn = EqualizedLR(name, gain=gain, mode=mode, lr_mul=lr_mul)
        weight = module._parameters[name]

        delattr(module, name)
        module.register_parameter(name + "_orig", weight)

        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a
        # plain attribute.

        setattr(module, name, weight.data)
        module.register_forward_pre_hook(fn)

        # TODO: register load state dict hook

        return fn


def equalized_lr(module, name="weight", gain=2**0.5, mode="fan_in", lr_mul=1.0):
    r"""Equalized Learning Rate.

    This trick is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    The general idea is to dynamically rescale the weight in training instead
    of in initializing so that the variance of the responses in each layer is
    guaranteed with some statistical properties.

    Note that this function is always combined with a convolution module which
    is initialized with :math:`\mathcal{N}(0, 1)`.

    Args:
        module (nn.Module): Module to be wrapped.
        name (str | optional): The name of weights. Defaults to 'weight'.
        mode (str, optional): The mode of computing ``fan`` which is the
            same as ``kaiming_init`` in pytorch. You can choose one from
            ['fan_in', 'fan_out']. Defaults to 'fan_in'.

    Returns:
        nn.Module: Module that is registered with equalized lr hook.
    """
    EqualizedLR.apply(module, name, gain=gain, mode=mode, lr_mul=lr_mul)

    return module


class EqualizedLRConvModule(ConvModule):
    r"""Equalized LR ConvModule.

    In this module, we inherit default ``mmcv.cnn.ConvModule`` and adopt
    equalized lr in convolution. The equalized learning rate is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    Note that, the initialization of ``self.conv`` will be overwrited as
    :math:`\mathcal{N}(0, 1)`.

    Args:
        equalized_lr_cfg (dict | None, optional): Config for ``EqualizedLR``.
            If ``None``, equalized learning rate is ignored. Defaults to
            dict(mode='fan_in').
    """

    def __init__(self, *args, equalized_lr_cfg=dict(mode="fan_in"), **kwargs):
        super(EqualizedLRConvModule, self).__init__(*args, **kwargs)
        self.with_equlized_lr = equalized_lr_cfg is not None
        if self.with_equlized_lr:
            self.conv = equalized_lr(self.conv, **equalized_lr_cfg)
            # initialize the conv weight with standard Gaussian noise.
            self._init_conv_weights()

    def _init_conv_weights(self):
        """Initialize conv weights as described in PGGAN."""
        normal_init(self.conv)


class EqualizedLRLinearModule(nn.Linear):
    r"""Equalized LR LinearModule.

    In this module, we adopt equalized lr in ``nn.Linear``. The equalized
    learning rate is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    Note that, the initialization of ``self.weight`` will be overwrited as
    :math:`\mathcal{N}(0, 1)`.

    Args:
        equalized_lr_cfg (dict | None, optional): Config for ``EqualizedLR``.
            If ``None``, equalized learning rate is ignored. Defaults to
            dict(mode='fan_in').
    """

    def __init__(self, *args, equalized_lr_cfg=dict(mode="fan_in"), **kwargs):
        super(EqualizedLRLinearModule, self).__init__(*args, **kwargs)
        self.with_equlized_lr = equalized_lr_cfg is not None
        if self.with_equlized_lr:
            self.lr_mul = equalized_lr_cfg.get("lr_mul", 1.0)
        else:
            # In fact, lr_mul will only be used in EqualizedLR for
            # initialization
            self.lr_mul = 1.0
        if self.with_equlized_lr:
            equalized_lr(self, **equalized_lr_cfg)
            self._init_linear_weights()

    def _init_linear_weights(self):
        """Initialize linear weights as described in PGGAN."""
        nn.init.normal_(self.weight, 0, 1.0 / self.lr_mul)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


class EqualLinearActModule(nn.Module):
    """Equalized LR Linear Module with Activation Layer.

    Args:
        nn ([type]): [description]
    """

    def __init__(
        self,
        *args,
        equalized_lr_cfg=dict(gain=1.0, lr_mul=1.0),
        bias=True,
        bias_init=0.0,
        act_cfg=None,
        **kwargs,
    ):
        super(EqualLinearActModule, self).__init__()
        self.with_activation = act_cfg is not None
        # w/o bias in linear layer
        self.linear = EqualizedLRLinearModule(
            *args, bias=False, equalized_lr_cfg=equalized_lr_cfg, **kwargs
        )

        if equalized_lr_cfg is not None:
            self.lr_mul = equalized_lr_cfg.get("lr_mul", 1.0)
        else:
            self.lr_mul = 1.0

        # define bias outside linear layer
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(self.linear.out_features).fill_(bias_init)
            )
        else:
            self.bias = None

        if self.with_activation:
            act_cfg = deepcopy(act_cfg)
            if act_cfg["type"] == "fused_bias":
                self.act_type = act_cfg.pop("type")
                assert self.bias is not None
                self.activate = partial(fused_bias_leakyrelu, **act_cfg)
            else:
                self.act_type = "normal"
                self.activate = build_activation_layer(act_cfg)
        else:
            self.act_type = None

    def forward(self, x):
        if x.ndim >= 3:
            x = x.reshape(x.size(0), -1)
        x = self.linear(x)

        if self.with_activation and self.act_type == "fused_bias":
            x = self.activate(x, self.bias * self.lr_mul)
        elif self.bias is not None and self.with_activation:
            x = self.activate(x + self.bias * self.lr_mul)
        elif self.bias is not None:
            x = x + self.bias * self.lr_mul
        elif self.with_activation:
            x = self.activate(x)

        return x


def _make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class UpsampleUpFIRDn(nn.Module):
    def __init__(self, kernel, factor=2):
        super(UpsampleUpFIRDn, self).__init__()

        self.factor = factor
        kernel = _make_kernel(kernel) * (factor**2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, x):
        out = upfirdn2d(x, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class DonwsampleUpFIRDn(nn.Module):
    def __init__(self, kernel, factor=2):
        super(DonwsampleUpFIRDn, self).__init__()

        self.factor = factor
        kernel = _make_kernel(kernel)
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
        super(Blur, self).__init__()
        kernel = _make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, x):
        return upfirdn2d(x, self.kernel, pad=self.pad)


class ModulatedConv2d(nn.Module):
    r"""Modulated Conv2d in StyleGANv2.

    Attention:

    #. ``style_bias`` is provided to check the difference between official TF
       implementation and other PyTorch implementation.
       In TF, Tero explicitly add the ``1.`` after style code, while unoffiical
       implementation adopts bias initalization with ``1.``.
       Details can be found in:
       https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py#L214
       https://github.com/NVlabs/stylegan2/blob/master/training/networks_stylegan2.py#L99
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        style_channels,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        equalized_lr_cfg=dict(mode="fan_in", lr_mul=1.0, gain=1.0),
        style_mod_cfg=dict(bias_init=1.0),
        style_bias=0.0,
        eps=1e-8,
    ):
        super(ModulatedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.style_channels = style_channels
        self.demodulate = demodulate
        # sanity check for kernel size
        assert isinstance(self.kernel_size, int) and (
            self.kernel_size >= 1 and self.kernel_size % 2 == 1
        )
        self.upsample = upsample
        self.downsample = downsample
        self.style_bias = style_bias
        self.eps = eps

        # build style modulation module
        style_mod_cfg = dict() if style_mod_cfg is None else style_mod_cfg

        self.style_modulation = EqualLinearActModule(
            style_channels, in_channels, **style_mod_cfg
        )
        # set lr_mul for conv weight
        lr_mul_ = 1.0
        if equalized_lr_cfg is not None:
            lr_mul_ = equalized_lr_cfg.get("lr_mul", 1.0)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size).div_(
                lr_mul_
            )
        )

        # build blurry layer for upsampling
        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, (pad0, pad1), upsample_factor=factor)
        # build blurry layer for downsampling
        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        # add equalized_lr hook for conv weight
        if equalized_lr_cfg is not None:
            equalized_lr(self, **equalized_lr_cfg)

        self.padding = kernel_size // 2

    def forward(self, x, style):
        n, c, h, w = x.shape
        # process style code
        style = self.style_modulation(style).view(n, 1, c, 1, 1) + self.style_bias

        # combine weight and style
        weight = self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(n, self.out_channels, 1, 1, 1)

        weight = weight.view(
            n * self.out_channels, c, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            x = x.reshape(1, n * c, h, w)
            weight = weight.view(
                n, self.out_channels, c, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                n * c, self.out_channels, self.kernel_size, self.kernel_size
            )
            x = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=n)
            x = x.reshape(n, self.out_channels, *x.shape[-2:])
            x = self.blur(x)

        elif self.downsample:
            x = self.blur(x)
            x = x.view(1, n * self.in_channels, *x.shape[-2:])
            x = F.conv2d(x, weight, stride=2, padding=0, groups=n)
            x = x.view(n, self.out_channels, *x.shape[-2:])
        else:
            x = x.view(1, n * c, h, w)
            x = F.conv2d(x, weight, stride=1, padding=self.padding, groups=n)
            x = x.view(n, self.out_channels, *x.shape[-2:])

        return x


class NoiseInjection(nn.Module):
    def __init__(self, noise_weight_init=0.0):
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1).fill_(noise_weight_init))

    def forward(self, image, noise=None, return_noise=False):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        if return_noise:
            return image + self.weight * noise, noise

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        if isinstance(size, int):
            size = [size, size]
        elif mmcv.is_seq_of(size, int):
            assert len(size) == 2, f"The length of size should be 2 but got {len(size)}"
        else:
            raise ValueError(f"Got invalid value in size, {size}")

        self.input = nn.Parameter(torch.randn(1, channel, *size))

    def forward(self, x):
        batch = x.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class ModulatedPEConv2d(nn.Module):
    r"""Modulated Conv2d in StyleGANv2.

    Attention:

    #. ``style_bias`` is provided to check the difference between official TF
       implementation and other PyTorch implementation.
       In TF, Tero explicitly add the ``1.`` after style code, while unoffiical
       implementation adopts bias initalization with ``1.``.
       Details can be found in:
       https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py#L214
       https://github.com/NVlabs/stylegan2/blob/master/training/networks_stylegan2.py#L99
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        style_channels,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        equalized_lr_cfg=dict(mode="fan_in", lr_mul=1.0, gain=1.0),
        style_mod_cfg=dict(bias_init=1.0),
        style_bias=0.0,
        eps=1e-8,
        no_pad=False,
        deconv2conv=False,
        interp_pad=None,
        up_config=dict(scale_factor=2, mode="nearest"),
        up_after_conv=False,
    ):
        super(ModulatedPEConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.style_channels = style_channels
        self.demodulate = demodulate
        # sanity check for kernel size
        assert isinstance(self.kernel_size, int) and (
            self.kernel_size >= 1 and self.kernel_size % 2 == 1
        )
        self.upsample = upsample
        self.downsample = downsample
        self.style_bias = style_bias
        self.eps = eps
        self.no_pad = no_pad
        self.deconv2conv = deconv2conv
        self.interp_pad = interp_pad
        self.with_interp_pad = interp_pad is not None
        self.up_config = deepcopy(up_config)
        self.up_after_conv = up_after_conv

        # build style modulation module
        style_mod_cfg = dict() if style_mod_cfg is None else style_mod_cfg

        self.style_modulation = EqualLinearActModule(
            style_channels, in_channels, **style_mod_cfg
        )
        # set lr_mul for conv weight
        lr_mul_ = 1.0
        if equalized_lr_cfg is not None:
            lr_mul_ = equalized_lr_cfg.get("lr_mul", 1.0)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size).div_(
                lr_mul_
            )
        )

        # build blurry layer for upsampling
        if upsample and not self.deconv2conv:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, (pad0, pad1), upsample_factor=factor)

        # build blurry layer for downsampling
        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        # add equalized_lr hook for conv weight
        if equalized_lr_cfg is not None:
            equalized_lr(self, **equalized_lr_cfg)

        # if `no_pad`, remove all of the padding in conv
        self.padding = kernel_size // 2 if not no_pad else 0

    def forward(self, x, style):
        n, c, h, w = x.shape
        # process style code
        style = self.style_modulation(style).view(n, 1, c, 1, 1) + self.style_bias

        # combine weight and style
        weight = self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(n, self.out_channels, 1, 1, 1)

        weight = weight.view(
            n * self.out_channels, c, self.kernel_size, self.kernel_size
        )

        if self.upsample and not self.deconv2conv:
            x = x.reshape(1, n * c, h, w)
            weight = weight.view(
                n, self.out_channels, c, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                n * c, self.out_channels, self.kernel_size, self.kernel_size
            )
            x = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=n)
            x = x.reshape(n, self.out_channels, *x.shape[-2:])
            x = self.blur(x)
        elif self.upsample and self.deconv2conv:
            if self.up_after_conv:
                x = x.reshape(1, n * c, h, w)
                x = F.conv2d(x, weight, padding=self.padding, groups=n)
                x = x.view(n, self.out_channels, *x.shape[2:4])

            if self.with_interp_pad:
                h_, w_ = x.shape[-2:]
                up_cfg_ = deepcopy(self.up_config)
                up_scale = up_cfg_.pop("scale_factor")
                size_ = (
                    h_ * up_scale + self.interp_pad,
                    w_ * up_scale + self.interp_pad,
                )
                x = F.interpolate(x, size=size_, **up_cfg_)
            else:
                x = F.interpolate(x, **self.up_config)

            if not self.up_after_conv:
                h_, w_ = x.shape[-2:]
                x = x.view(1, n * c, h_, w_)
                x = F.conv2d(x, weight, padding=self.padding, groups=n)
                x = x.view(n, self.out_channels, *x.shape[2:4])

        elif self.downsample:
            x = self.blur(x)
            x = x.view(1, n * self.in_channels, *x.shape[-2:])
            x = F.conv2d(x, weight, stride=2, padding=0, groups=n)
            x = x.view(n, self.out_channels, *x.shape[-2:])
        else:
            x = x.view(1, n * c, h, w)
            x = F.conv2d(x, weight, stride=1, padding=self.padding, groups=n)
            x = x.view(n, self.out_channels, *x.shape[-2:])

        return x


class ModulatedStyleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        style_channels,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
        style_mod_cfg=dict(bias_init=1.0),
        style_bias=0.0,
    ):
        super(ModulatedStyleConv, self).__init__()

        self.conv = ModulatedConv2d(
            in_channels,
            out_channels,
            kernel_size,
            style_channels,
            demodulate=demodulate,
            upsample=upsample,
            blur_kernel=blur_kernel,
            style_mod_cfg=style_mod_cfg,
            style_bias=style_bias,
        )

        self.noise_injector = NoiseInjection()
        self.activate = FusedBiasLeakyReLU(out_channels)

    def forward(self, x, style, noise=None, return_noise=False):
        out = self.conv(x, style)
        if return_noise:
            out, noise = self.noise_injector(
                out, noise=noise, return_noise=return_noise
            )
        else:
            out = self.noise_injector(out, noise=noise, return_noise=return_noise)

        out = self.activate(out)

        if return_noise:
            return out, noise
        else:
            return out


class ModulatedPEStyleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        style_channels,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
        style_mod_cfg=dict(bias_init=1.0),
        style_bias=0.0,
        **kwargs,
    ):
        super(ModulatedPEStyleConv, self).__init__()

        self.conv = ModulatedPEConv2d(
            in_channels,
            out_channels,
            kernel_size,
            style_channels,
            demodulate=demodulate,
            upsample=upsample,
            blur_kernel=blur_kernel,
            style_mod_cfg=style_mod_cfg,
            style_bias=style_bias,
            **kwargs,
        )

        self.noise_injector = NoiseInjection()
        self.activate = FusedBiasLeakyReLU(out_channels)

    def forward(self, x, style, noise=None, return_noise=False):
        out = self.conv(x, style)
        if return_noise:
            out, noise = self.noise_injector(
                out, noise=noise, return_noise=return_noise
            )
        else:
            out = self.noise_injector(out, noise=noise, return_noise=return_noise)

        out = self.activate(out)

        if return_noise:
            return out, noise
        else:
            return out


class ModulatedToRGB(nn.Module):
    def __init__(
        self,
        in_channels,
        style_channels,
        out_channels=3,
        upsample=True,
        blur_kernel=[1, 3, 3, 1],
        style_mod_cfg=dict(bias_init=1.0),
        style_bias=0.0,
    ):
        super(ModulatedToRGB, self).__init__()

        if upsample:
            self.upsample = UpsampleUpFIRDn(blur_kernel)

        self.conv = ModulatedConv2d(
            in_channels,
            out_channels=out_channels,
            kernel_size=1,
            style_channels=style_channels,
            demodulate=False,
            style_mod_cfg=style_mod_cfg,
            style_bias=style_bias,
        )

        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, style, skip=None):
        out = self.conv(x, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        return out


class ConvDownLayer(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        act_cfg=dict(type="fused_bias"),
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

        self.with_fused_bias = (
            act_cfg is not None and act_cfg.get("type") == "fused_bias"
        )
        if self.with_fused_bias:
            conv_act_cfg = None
        else:
            conv_act_cfg = act_cfg
        layers.append(
            EqualizedLRConvModule(
                in_channels,
                out_channels,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not self.with_fused_bias,
                norm_cfg=None,
                act_cfg=conv_act_cfg,
                equalized_lr_cfg=dict(mode="fan_in", gain=1.0),
            )
        )
        if self.with_fused_bias:
            layers.append(FusedBiasLeakyReLU(out_channels))

        super(ConvDownLayer, self).__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, blur_kernel=[1, 3, 3, 1]):
        super(ResBlock, self).__init__()

        self.conv1 = ConvDownLayer(in_channels, in_channels, 3, blur_kernel=blur_kernel)
        self.conv2 = ConvDownLayer(
            in_channels, out_channels, 3, downsample=True, blur_kernel=blur_kernel
        )

        self.skip = ConvDownLayer(
            in_channels,
            out_channels,
            1,
            downsample=True,
            act_cfg=None,
            bias=False,
            blur_kernel=blur_kernel,
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class ModMBStddevLayer(nn.Module):
    """Modified MiniBatch Stddev Layer.

    This layer is modified from ``MiniBatchStddevLayer`` used in PGGAN. In
    StyleGAN2, the authors add a new feature, `channel_groups`, into this
    layer.
    """

    def __init__(self, group_size=4, channel_groups=1, sync_groups=None, eps=1e-8):
        super(ModMBStddevLayer, self).__init__()
        self.group_size = group_size
        self.eps = eps
        self.channel_groups = channel_groups
        self.sync_groups = group_size if sync_groups is None else sync_groups

    def forward(self, x):
        # batch size should be smaller than or equal to group size. Otherwise,
        # batch size should be divisible by the group size.
        assert x.shape[0] <= self.group_size or x.shape[0] % self.group_size == 0, (
            "Batch size be smaller than or equal "
            "to group size. Otherwise,"
            " batch size should be divisible by the group size."
            f"But got batch size {x.shape[0]},"
            f" group size {self.group_size}"
        )
        assert x.shape[1] % self.channel_groups == 0, (
            '"channel_groups" must be divided by the feature channels. '
            f"channel_groups: {self.channel_groups}, "
            f"feature channels: {x.shape[1]}"
        )

        n, c, h, w = x.shape
        group_size = min(n, self.group_size)
        # [G, M, Gc, C', H, W]
        y = torch.reshape(
            x, (group_size, -1, self.channel_groups, c // self.channel_groups, h, w)
        )
        y = torch.var(y, dim=0, unbiased=False)
        y = torch.sqrt(y + self.eps)
        # [M, 1, 1, 1]
        y = y.mean(dim=(2, 3, 4), keepdim=True).squeeze(2)
        y = y.repeat(group_size, 1, h, w)
        return torch.cat([x, y], dim=1)




def get_module_device(module):
    """Get the device of a module.

    Args:
        module (nn.Module): A module contains the parameters.

    Returns:
        torch.device: The device of the module.
    """
    try:
        next(module.parameters())
    except StopIteration:
        raise ValueError("The input module should contain parameters.")

    if next(module.parameters()).is_cuda:
        return next(module.parameters()).get_device()
    else:
        return torch.device("cpu")


@torch.no_grad()
def get_mean_latent(generator, num_samples=4096, bs_per_repeat=1024):
    """Get mean latent of W space in Style-based GANs.
    Args:
        generator (nn.Module): Generator of a Style-based GAN.
        num_samples (int, optional): Number of sample times. Defaults to 4096.
        bs_per_repeat (int, optional): Batch size of noises per sample.
            Defaults to 1024.
    Returns:
        Tensor: Mean latent of this generator.
    """
    device = get_module_device(generator)
    mean_style = None
    n_repeat = num_samples // bs_per_repeat
    assert n_repeat * bs_per_repeat == num_samples

    for i in range(n_repeat):
        style = generator.style_mapping(
            torch.randn(bs_per_repeat, generator.style_channels).to(device)
        ).mean(0, keepdim=True)
        if mean_style is None:
            mean_style = style
        else:
            mean_style += style
    mean_style /= float(n_repeat)

    return mean_style


@torch.no_grad()
def style_mixing(
    generator,
    n_source,
    n_target,
    inject_index=1,
    truncation_latent=None,
    truncation=0.7,
    style_channels=512,
    **kwargs,
):
    device = get_module_device(generator)
    source_code = torch.randn(n_source, style_channels).to(device)
    target_code = torch.randn(n_target, style_channels).to(device)

    source_image = generator(
        source_code,
        truncation_latent=truncation_latent,
        truncation=truncation,
        **kwargs,
    )

    h, w = source_image.shape[-2:]
    images = [torch.ones(1, 3, h, w).to(device) * -1]

    target_image = generator(
        target_code,
        truncation_latent=truncation_latent,
        truncation=truncation,
        **kwargs,
    )

    images.append(source_image)

    for i in range(n_target):
        image = generator(
            [target_code[i].unsqueeze(0).repeat(n_source, 1), source_code],
            truncation_latent=truncation_latent,
            truncation=truncation,
            inject_index=inject_index,
            **kwargs,
        )
        images.append(target_image[i].unsqueeze(0))
        images.append(image)

    images = torch.cat(images, 0)

    return images


import random

import numpy as np
import torch
import torch.nn as nn
from mmcv.runner.checkpoint import _load_checkpoint_with_prefix

# from mmedit.models.registry import COMPONENTS
# from .common import get_mean_latent, get_module_device, style_mixing
# from .modules import (ConstantInput, ConvDownLayer, EqualLinearActModule,
#                      ModMBStddevLayer, ModulatedStyleConv, ModulatedToRGB,
#                      PixelNorm, ResBlock)


@COMPONENTS.register_module()
class StyleGANv2Generator(nn.Module):
    r"""StyleGAN2 Generator.

    This module comes from MMGeneration. In the future, this code will be
    removed and StyleGANv2 will be directly imported from mmgen.

    In StyleGAN2, we use a static architecture composing of a style mapping
    module and number of covolutional style blocks. More details can be found
    in: Analyzing and Improving the Image Quality of StyleGAN CVPR2020.

    You can load pretrained model through passing information into
    ``pretrained`` argument. We have already offered offical weights as
    follows:

    - styelgan2-ffhq-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-ffhq-config-f-official_20210327_171224-bce9310c.pth  # noqa
    - stylegan2-horse-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-horse-config-f-official_20210327_173203-ef3e69ca.pth  # noqa
    - stylegan2-car-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-car-config-f-official_20210327_172340-8cfe053c.pth  # noqa
    - styelgan2-cat-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-cat-config-f-official_20210327_172444-15bc485b.pth  # noqa
    - stylegan2-church-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-church-config-f-official_20210327_172657-1d42b7d1.pth  # noqa

    If you want to load the ema model, you can just use following codes:

    .. code-block:: python

        # ckpt_http is one of the valid path from http source
        generator = StyleGANv2Generator(1024, 512,
                                        pretrained=dict(
                                            ckpt_path=ckpt_http,
                                            prefix='generator_ema'))

    Of course, you can also download the checkpoint in advance and set
    ``ckpt_path`` with local path. If you just want to load the original
    generator (not the ema model), please set the prefix with 'generator'.

    Note that our implementation allows to generate BGR image, while the
    original StyleGAN2 outputs RGB images by default. Thus, we provide
    ``bgr2rgb`` argument to convert the image space.

    Args:
        out_size (int): The output size of the StyleGAN2 generator.
        style_channels (int): The number of channels for style code.
        num_mlps (int, optional): The number of MLP layers. Defaults to 8.
        channel_multiplier (int, optional): The mulitiplier factor for the
            channel number. Defaults to 2.
        blur_kernel (list, optional): The blurry kernel. Defaults
            to [1, 3, 3, 1].
        lr_mlp (float, optional): The learning rate for the style mapping
            layer. Defaults to 0.01.
        default_style_mode (str, optional): The default mode of style mixing.
            In training, we defaultly adopt mixing style mode. However, in the
            evaluation, we use 'single' style mode. `['mix', 'single']` are
            currently supported. Defaults to 'mix'.
        eval_style_mode (str, optional): The evaluation mode of style mixing.
            Defaults to 'single'.
        mix_prob (float, optional): Mixing probabilty. The value should be
            in range of [0, 1]. Defaults to 0.9.
        pretrained (dict | None, optional): Information for pretained models.
            The necessary key is 'ckpt_path'. Besides, you can also provide
            'prefix' to load the generator part from the whole state dict.
            Defaults to None.
        bgr2rgb (bool, optional): Whether to flip the image channel dimension.
            Defaults to False.
    """

    def __init__(
        self,
        out_size,
        style_channels,
        num_mlps=8,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        default_style_mode="mix",
        eval_style_mode="single",
        mix_prob=0.9,
        pretrained=None,
        bgr2rgb=False,
    ):
        super(StyleGANv2Generator, self).__init__()
        self.out_size = out_size
        self.style_channels = style_channels
        self.num_mlps = num_mlps
        self.channel_multiplier = channel_multiplier
        self.lr_mlp = lr_mlp
        self._default_style_mode = default_style_mode
        self.default_style_mode = default_style_mode
        self.eval_style_mode = eval_style_mode
        self.mix_prob = mix_prob
        self.bgr2rgb = bgr2rgb

        # define style mapping layers
        mapping_layers = [PixelNorm()]

        for _ in range(num_mlps):
            mapping_layers.append(
                EqualLinearActModule(
                    style_channels,
                    style_channels,
                    equalized_lr_cfg=dict(lr_mul=lr_mlp, gain=1.0),
                    act_cfg=dict(type="fused_bias"),
                )
            )

        self.style_mapping = nn.Sequential(*mapping_layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        # constant input layer
        self.constant_input = ConstantInput(self.channels[4])
        # 4x4 stage
        self.conv1 = ModulatedStyleConv(
            self.channels[4],
            self.channels[4],
            kernel_size=3,
            style_channels=style_channels,
            blur_kernel=blur_kernel,
        )
        self.to_rgb1 = ModulatedToRGB(self.channels[4], style_channels, upsample=False)

        # generator backbone (8x8 --> higher resolutions)
        self.log_size = int(np.log2(self.out_size))

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_channels_ = self.channels[4]

        for i in range(3, self.log_size + 1):
            out_channels_ = self.channels[2**i]

            self.convs.append(
                ModulatedStyleConv(
                    in_channels_,
                    out_channels_,
                    3,
                    style_channels,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )
            self.convs.append(
                ModulatedStyleConv(
                    out_channels_,
                    out_channels_,
                    3,
                    style_channels,
                    upsample=False,
                    blur_kernel=blur_kernel,
                )
            )
            self.to_rgbs.append(
                ModulatedToRGB(out_channels_, style_channels, upsample=True)
            )

            in_channels_ = out_channels_

        self.num_latents = self.log_size * 2 - 2
        self.num_injected_noises = self.num_latents - 1

        # register buffer for injected noises
        for layer_idx in range(self.num_injected_noises):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2**res, 2**res]
            self.register_buffer(f"injected_noise_{layer_idx}", torch.randn(*shape))

        if pretrained is not None:
            self._load_pretrained_model(**pretrained)

    def _load_pretrained_model(
        self, ckpt_path, prefix="", map_location="cpu", strict=True
    ):
        state_dict = _load_checkpoint_with_prefix(prefix, ckpt_path, map_location)
        self.load_state_dict(state_dict, strict=strict)
        mmcv.print_log(f"Load pretrained model from {ckpt_path}", "mmedit")

    def train(self, mode=True):
        if mode:
            if self.default_style_mode != self._default_style_mode:
                mmcv.print_log(
                    f"Switch to train style mode: {self._default_style_mode}", "mmgen"
                )
            self.default_style_mode = self._default_style_mode

        else:
            if self.default_style_mode != self.eval_style_mode:
                mmcv.print_log(
                    f"Switch to evaluation style mode: {self.eval_style_mode}", "mmgen"
                )
            self.default_style_mode = self.eval_style_mode

        return super(StyleGANv2Generator, self).train(mode)

    def make_injected_noise(self):
        device = get_module_device(self)

        noises = [torch.randn(1, 1, 2**2, 2**2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2**i, 2**i, device=device))

        return noises

    def get_mean_latent(self, num_samples=4096, **kwargs):
        return get_mean_latent(self, num_samples, **kwargs)

    def style_mixing(
        self, n_source, n_target, inject_index=1, truncation_latent=None, truncation=0.7
    ):
        return style_mixing(
            self,
            n_source=n_source,
            n_target=n_target,
            inject_index=inject_index,
            truncation=truncation,
            truncation_latent=truncation_latent,
            style_channels=self.style_channels,
        )

    def forward(
        self,
        styles,
        num_batches=-1,
        return_noise=False,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        injected_noise=None,
        randomize_noise=True,
    ):
        """Forward function.

        This function has been integrated with the truncation trick. Please
        refer to the usage of `truncation` and `truncation_latent`.

        Args:
            styles (torch.Tensor | list[torch.Tensor] | callable | None): In
                StyleGAN2, you can provide noise tensor or latent tensor. Given
                a list containing more than one noise or latent tensors, style
                mixing trick will be used in training. Of course, You can
                directly give a batch of noise through a ``torhc.Tensor`` or
                offer a callable function to sample a batch of noise data.
                Otherwise, the ``None`` indicates to use the default noise
                sampler.
            num_batches (int, optional): The number of batch size.
                Defaults to 0.
            return_noise (bool, optional): If True, ``noise_batch`` will be
                returned in a dict with ``fake_img``. Defaults to False.
            return_latents (bool, optional): If True, ``latent`` will be
                returned in a dict with ``fake_img``. Defaults to False.
            inject_index (int | None, optional): The index number for mixing
                style codes. Defaults to None.
            truncation (float, optional): Truncation factor. Give value less
                than 1., the truncatioin trick will be adopted. Defaults to 1.
            truncation_latent (torch.Tensor, optional): Mean truncation latent.
                Defaults to None.
            input_is_latent (bool, optional): If `True`, the input tensor is
                the latent tensor. Defaults to False.
            injected_noise (torch.Tensor | None, optional): Given a tensor, the
                random noise will be fixed as this input injected noise.
                Defaults to None.
            randomize_noise (bool, optional): If `False`, images are sampled
                with the buffered noise tensor injected to the style conv
                block. Defaults to True.

        Returns:
            torch.Tensor | dict: Generated image tensor or dictionary
                containing more data.
        """
        # receive noise and conduct sanity check.
        if isinstance(styles, torch.Tensor):
            assert styles.shape[1] == self.style_channels
            styles = [styles]
        elif mmcv.is_seq_of(styles, torch.Tensor):
            for t in styles:
                assert t.shape[-1] == self.style_channels
        # receive a noise generator and sample noise.
        elif callable(styles):
            device = get_module_device(self)
            noise_generator = styles
            assert num_batches > 0
            if self.default_style_mode == "mix" and random.random() < self.mix_prob:
                styles = [
                    noise_generator((num_batches, self.style_channels))
                    for _ in range(2)
                ]
            else:
                styles = [noise_generator((num_batches, self.style_channels))]
            styles = [s.to(device) for s in styles]
        # otherwise, we will adopt default noise sampler.
        else:
            device = get_module_device(self)
            assert num_batches > 0 and not input_is_latent
            if self.default_style_mode == "mix" and random.random() < self.mix_prob:
                styles = [
                    torch.randn((num_batches, self.style_channels)) for _ in range(2)
                ]
            else:
                styles = [torch.randn((num_batches, self.style_channels))]
            styles = [s.to(device) for s in styles]

        if not input_is_latent:
            noise_batch = styles
            styles = [self.style_mapping(s) for s in styles]
        else:
            noise_batch = None

        if injected_noise is None:
            if randomize_noise:
                injected_noise = [None] * self.num_injected_noises
            else:
                injected_noise = [
                    getattr(self, f"injected_noise_{i}")
                    for i in range(self.num_injected_noises)
                ]
        # use truncation trick
        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t
        # no style mixing
        if len(styles) < 2:
            inject_index = self.num_latents

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]
        # style mixing
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.num_latents - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = (
                styles[1].unsqueeze(1).repeat(1, self.num_latents - inject_index, 1)
            )

            latent = torch.cat([latent, latent2], 1)

        # 4x4 stage
        out = self.constant_input(latent)
        out = self.conv1(out, latent[:, 0], noise=injected_noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        _index = 1

        # 8x8 ---> higher resolutions
        for up_conv, conv, noise1, noise2, to_rgb in zip(
            self.convs[::2],
            self.convs[1::2],
            injected_noise[1::2],
            injected_noise[2::2],
            self.to_rgbs,
        ):
            out = up_conv(out, latent[:, _index], noise=noise1)
            out = conv(out, latent[:, _index + 1], noise=noise2)
            skip = to_rgb(out, latent[:, _index + 2], skip)

            _index += 2

        img = skip

        if self.bgr2rgb:
            img = torch.flip(img, dims=1)

        if return_latents or return_noise:
            output_dict = dict(
                fake_img=img,
                latent=latent,
                inject_index=inject_index,
                noise_batch=noise_batch,
            )
            return output_dict
        else:
            return img


@COMPONENTS.register_module()
class StyleGAN2Discriminator(nn.Module):
    """StyleGAN2 Discriminator.

    This module comes from MMGeneration. In the future, this code will be
    removed and StyleGANv2 will be directly imported from mmgen.

    The architecture of this discriminator is proposed in StyleGAN2. More
    details can be found in: Analyzing and Improving the Image Quality of
    StyleGAN CVPR2020.

    You can load pretrained model through passing information into
    ``pretrained`` argument. We have already offered offical weights as
    follows:

    - styelgan2-ffhq-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-ffhq-config-f-official_20210327_171224-bce9310c.pth  # noqa
    - stylegan2-horse-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-horse-config-f-official_20210327_173203-ef3e69ca.pth  # noqa
    - stylegan2-car-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-car-config-f-official_20210327_172340-8cfe053c.pth  # noqa
    - styelgan2-cat-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-cat-config-f-official_20210327_172444-15bc485b.pth  # noqa
    - stylegan2-church-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-church-config-f-official_20210327_172657-1d42b7d1.pth  # noqa

    If you want to load the ema model, you can just use following codes:

    .. code-block:: python

        # ckpt_http is one of the valid path from http source
        discriminator = StyleGAN2Discriminator(1024, 512,
                                               pretrained=dict(
                                                   ckpt_path=ckpt_http,
                                                   prefix='discriminator'))

    Of course, you can also download the checkpoint in advance and set
    ``ckpt_path`` with local path.

    Note that our implementation adopts BGR image as input, while the
    original StyleGAN2 provides RGB images to the discriminator. Thus, we
    provide ``bgr2rgb`` argument to convert the image space. If your images
    follow the RGB order, please set it to ``True`` accordingly.

    Args:
        in_size (int): The input size of images.
        channel_multiplier (int, optional): The mulitiplier factor for the
            channel number. Defaults to 2.
        blur_kernel (list, optional): The blurry kernel. Defaults
            to [1, 3, 3, 1].
        mbstd_cfg (dict, optional): Configs for minibatch-stddev layer.
            Defaults to dict(group_size=4, channel_groups=1).
        pretrained (dict | None, optional): Information for pretained models.
            The necessary key is 'ckpt_path'. Besides, you can also provide
            'prefix' to load the generator part from the whole state dict.
            Defaults to None.
        bgr2rgb (bool, optional): Whether to flip the image channel dimension.
            Defaults to False.
    """

    def __init__(
        self,
        in_size,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        mbstd_cfg=dict(group_size=4, channel_groups=1),
        pretrained=None,
        bgr2rgb=False,
    ):
        super(StyleGAN2Discriminator, self).__init__()

        self.bgr2rgb = bgr2rgb

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        log_size = int(np.log2(in_size))

        in_channels = channels[in_size]

        convs = [ConvDownLayer(3, channels[in_size], 1)]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channels, out_channel, blur_kernel))

            in_channels = out_channel

        self.convs = nn.Sequential(*convs)

        self.mbstd_layer = ModMBStddevLayer(**mbstd_cfg)

        self.final_conv = ConvDownLayer(in_channels + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinearActModule(
                channels[4] * 4 * 4, channels[4], act_cfg=dict(type="fused_bias")
            ),
            EqualLinearActModule(channels[4], 1),
        )
        if pretrained is not None:
            self._load_pretrained_model(**pretrained)

    def _load_pretrained_model(
        self, ckpt_path, prefix="", map_location="cpu", strict=True
    ):
        state_dict = _load_checkpoint_with_prefix(prefix, ckpt_path, map_location)
        self.load_state_dict(state_dict, strict=strict)
        mmcv.print_log(f"Load pretrained model from {ckpt_path}", "mmedit")

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Predict score for the input image.
        """
        if self.bgr2rgb:
            x = torch.flip(x, dims=1)

        x = self.convs(x)

        x = self.mbstd_layer(x)
        x = self.final_conv(x)
        x = x.view(x.shape[0], -1)
        x = self.final_linear(x)

        return x


import torch.nn as nn

# from .sr_backbone_utils import default_init_weights


class PixelShufflePack(nn.Module):
    """Pixel Shuffle upsample layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.

    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels, scale_factor, upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2,
        )
        self.init_weights()

    def init_weights(self):
        """Initialize weights for PixelShufflePack."""
        default_init_weights(self, 1)

    def forward(self, x):
        """Forward function for PixelShufflePack.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


import torch.nn as nn
from mmcv import build_from_cfg

# from .registry import BACKBONES, COMPONENTS, LOSSES, MODELS


def build(cfg, registry, default_args=None):
    """Build module function.

    Args:
        cfg (dict): Configuration for building modules.
        registry (obj): ``registry`` object.
        default_args (dict, optional): Default arguments. Defaults to None.
    """
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)

    return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    """Build backbone.

    Args:
        cfg (dict): Configuration for building backbone.
    """
    return build(cfg, BACKBONES)


def build_component(cfg):
    """Build component.

    Args:
        cfg (dict): Configuration for building component.
    """
    return build(cfg, COMPONENTS)


def build_loss(cfg):
    """Build loss.

    Args:
        cfg (dict): Configuration for building loss.
    """
    return build(cfg, LOSSES)


def build_model(cfg, train_cfg=None, test_cfg=None):
    """Build model.

    Args:
        cfg (dict): Configuration for building model.
        train_cfg (dict): Training configuration. Default: None.
        test_cfg (dict): Testing configuration. Default: None.
    """
    return build(cfg, MODELS, dict(train_cfg=train_cfg, test_cfg=test_cfg))


import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from mmcv.utils.parrots_wrapper import _BatchNorm


def default_init_weights(module, scale=1):
    """Initialize network weights.

    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode="fan_in", bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode="fan_in", bias=0)
            m.weight.data *= scale
        elif isinstance(m, _BatchNorm):
            constant_init(m.weight, val=1, bias=0)


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:

    ::

        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        # if res_scale < 1.0, use the default initialization, as in EDSR.
        # if res_scale = 1.0, use scaled kaiming_init, as in MSRResNet.
        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self):
        """Initialize weights for ResidualBlockNoBN.

        Initialization methods like `kaiming_init` are for VGG-style
        modules. For modules with residual paths, using smaller std is
        better for stability and performance. We empirically use 0.1.
        See more details in "ESRGAN: Enhanced Super-Resolution Generative
        Adversarial Networks"
        """

        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


import torch
import torch.nn as nn

# from mmcv.runner import load_checkpoint

# from mmedit.models.common import default_init_weights, make_layer
# from mmedit.models.registry import BACKBONES
# from mmedit.utils import get_root_logger


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        mid_channels (int): Channel number of intermediate features.
        growth_channels (int): Channels for each growth.
    """

    def __init__(self, mid_channels=64, growth_channels=32):
        super().__init__()
        for i in range(5):
            out_channels = mid_channels if i == 4 else growth_channels
            self.add_module(
                f"conv{i+1}",
                nn.Conv2d(mid_channels + i * growth_channels, out_channels, 3, 1, 1),
            )
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.init_weights()

    def init_weights(self):
        """Init weights for ResidualDenseBlock.

        Use smaller std for better stability and performance. We empirically
        use 0.1. See more details in "ESRGAN: Enhanced Super-Resolution
        Generative Adversarial Networks"
        """
        for i in range(5):
            default_init_weights(getattr(self, f"conv{i+1}"), 0.1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        mid_channels (int): Channel number of intermediate features.
        growth_channels (int): Channels for each growth.
    """

    def __init__(self, mid_channels, growth_channels=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(mid_channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(mid_channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(mid_channels, growth_channels)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


@BACKBONES.register_module()
class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
    Currently, it supports x4 upsampling scale factor.

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64
        num_blocks (int): Block number in the trunk network. Defaults: 23
        growth_channels (int): Channels for each growth. Default: 32.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=64,
        num_blocks=23,
        growth_channels=32,
    ):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.body = make_layer(
            RRDB, num_blocks, mid_channels=mid_channels, growth_channels=growth_channels
        )
        self.conv_body = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_hr = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(mid_channels, out_channels, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

    def init_weights(self, pretrained=None, strict=True):
        # Init weights for models.

        # Args:
        #    pretrained (str, optional): Path for pretrained weights. If given
        #        None, pretrained weights will not be loaded. Defaults to None.
        #    strict (boo, optional): Whether strictly load the pretrained model.
        #        Defaults to True.
        #
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            # Use smaller std for better stability and performance. We
            # use 0.1. See more details in "ESRGAN: Enhanced Super-Resolution
            # Generative Adversarial Networks"
            for m in [
                self.conv_first,
                self.conv_body,
                self.conv_up1,
                self.conv_up2,
                self.conv_hr,
                self.conv_last,
            ]:
                default_init_weights(m, 0.1)
        else:
            raise TypeError(
                f'"pretrained" must be a str or None. '
                f"But received {type(pretrained)}."
            )


import torch
import torch.nn as nn

# from mmcv.runner import load_checkpoint

# from mmedit.models.backbones.sr_backbones.rrdb_net import RRDB
# from mmedit.models.builder import build_component
# from mmedit.models.common import PixelShufflePack, make_layer
# from mmedit.models.registry import BACKBONES
# from mmedit.utils import get_root_logger


@BACKBONES.register_module()
class GLEANStyleGANv2(nn.Module):
    r"""GLEAN (using StyleGANv2) architecture for super-resolution.

    Paper:
        GLEAN: Generative Latent Bank for Large-Factor Image Super-Resolution,
        CVPR, 2021

    This method makes use of StyleGAN2 and hence the arguments mostly follow
    that in 'StyleGAN2v2Generator'.

    In StyleGAN2, we use a static architecture composing of a style mapping
    module and number of covolutional style blocks. More details can be found
    in: Analyzing and Improving the Image Quality of StyleGAN CVPR2020.

    You can load pretrained model through passing information into
    ``pretrained`` argument. We have already offered offical weights as
    follows:

    - styelgan2-ffhq-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-ffhq-config-f-official_20210327_171224-bce9310c.pth  # noqa
    - stylegan2-horse-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-horse-config-f-official_20210327_173203-ef3e69ca.pth  # noqa
    - stylegan2-car-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-car-config-f-official_20210327_172340-8cfe053c.pth  # noqa
    - styelgan2-cat-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-cat-config-f-official_20210327_172444-15bc485b.pth  # noqa
    - stylegan2-church-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-church-config-f-official_20210327_172657-1d42b7d1.pth  # noqa

    If you want to load the ema model, you can just use following codes:

    .. code-block:: python

        # ckpt_http is one of the valid path from http source
        generator = StyleGANv2Generator(1024, 512,
                                        pretrained=dict(
                                            ckpt_path=ckpt_http,
                                            prefix='generator_ema'))

    Of course, you can also download the checkpoint in advance and set
    ``ckpt_path`` with local path. If you just want to load the original
    generator (not the ema model), please set the prefix with 'generator'.

    Note that our implementation allows to generate BGR image, while the
    original StyleGAN2 outputs RGB images by default. Thus, we provide
    ``bgr2rgb`` argument to convert the image space.

    Args:
        in_size (int): The size of the input image.
        out_size (int): The output size of the StyleGAN2 generator.
        img_channels (int): Number of channels of the input images. 3 for RGB
            image and 1 for grayscale image. Default: 3.
        rrdb_channels (int): Number of channels of the RRDB features.
            Default: 64.
        num_rrdbs (int): Number of RRDB blocks in the encoder. Default: 23.
        style_channels (int): The number of channels for style code.
            Default: 512.
        num_mlps (int, optional): The number of MLP layers. Defaults to 8.
        channel_multiplier (int, optional): The mulitiplier factor for the
            channel number. Defaults to 2.
        blur_kernel (list, optional): The blurry kernel. Defaults
            to [1, 3, 3, 1].
        lr_mlp (float, optional): The learning rate for the style mapping
            layer. Defaults to 0.01.
        default_style_mode (str, optional): The default mode of style mixing.
            In training, we defaultly adopt mixing style mode. However, in the
            evaluation, we use 'single' style mode. `['mix', 'single']` are
            currently supported. Defaults to 'mix'.
        eval_style_mode (str, optional): The evaluation mode of style mixing.
            Defaults to 'single'.
        mix_prob (float, optional): Mixing probabilty. The value should be
            in range of [0, 1]. Defaults to 0.9.
        pretrained (dict | None, optional): Information for pretained models.
            The necessary key is 'ckpt_path'. Besides, you can also provide
            'prefix' to load the generator part from the whole state dict.
            Defaults to None.
        bgr2rgb (bool, optional): Whether to flip the image channel dimension.
            Defaults to False.
    """

    def __init__(
        self,
        in_size,
        out_size,
        img_channels=3,
        img_channels_out=3,
        rrdb_channels=64,
        num_rrdbs=23,
        style_channels=512,
        num_mlps=8,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        default_style_mode="mix",
        eval_style_mode="single",
        mix_prob=0.9,
        pretrained=None,
        bgr2rgb=False,
    ):
        super().__init__()

        # input size must be strictly smaller than output size
        # if in_size >= out_size:
        #    raise ValueError('in_size must be smaller than out_size, but got '
        #                     f'{in_size} and {out_size}.')

        # latent bank (StyleGANv2), with weights being fixed
        self.generator = build_component(
            dict(
                type="StyleGANv2Generator",
                out_size=out_size,
                style_channels=style_channels,
                num_mlps=num_mlps,
                channel_multiplier=channel_multiplier,
                blur_kernel=blur_kernel,
                lr_mlp=lr_mlp,
                default_style_mode=default_style_mode,
                eval_style_mode=eval_style_mode,
                mix_prob=mix_prob,
                pretrained=pretrained,
                bgr2rgb=bgr2rgb,
            )
        )
        self.generator.requires_grad_(False)

        self.in_size = in_size
        self.style_channels = style_channels
        channels = self.generator.channels

        # encoder
        num_styles = int(np.log2(out_size)) * 2 - 2
        encoder_res = [2**i for i in range(int(np.log2(in_size)), 1, -1)]
        self.encoder = nn.ModuleList()
        self.encoder.append(
            nn.Sequential(
                RRDBFeatureExtractor(img_channels, rrdb_channels, num_blocks=num_rrdbs),
                nn.Conv2d(rrdb_channels, channels[in_size], 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
        )
        for res in encoder_res:
            in_channels = channels[res]
            if res > 4:
                out_channels = channels[res // 2]
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Flatten(),
                    nn.Linear(16 * in_channels, num_styles * style_channels),
                )
            self.encoder.append(block)

        # additional modules for StyleGANv2
        self.fusion_out = nn.ModuleList()
        self.fusion_skip = nn.ModuleList()
        for res in encoder_res[::-1]:
            num_channels = channels[res]
            self.fusion_out.append(
                nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1, bias=True)
            )
            self.fusion_skip.append(nn.Conv2d(num_channels + 3, 3, 3, 1, 1, bias=True))

        # decoder
        decoder_res = [
            2**i for i in range(int(np.log2(in_size)), int(np.log2(out_size) + 1))
        ]
        self.decoder = nn.ModuleList()
        for res in decoder_res:
            if res == in_size:
                in_channels = channels[res]
            else:
                in_channels = 2 * channels[res]

            if res < out_size:
                out_channels = channels[res * 2]
                self.decoder.append(
                    PixelShufflePack(in_channels, out_channels, 2, upsample_kernel=3)
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, 64, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(64, img_channels_out, 3, 1, 1),
                    )
                )

    def forward(self, lq):
        """Forward function.

        Args:
            lq (Tensor): Input LR image with shape (n, c, h, w).

        Returns:
            Tensor: Output HR image.
        """

        h, w = lq.shape[2:]
        # if h != self.in_size or w != self.in_size:
        #    raise AssertionError(
        #        f'Spatial resolution must equal in_size ({self.in_size}).'
        #        f' Got ({h}, {w}).')

        # encoder
        feat = lq
        encoder_features = []
        for block in self.encoder:
            feat = block(feat)
            encoder_features.append(feat)
        encoder_features = encoder_features[::-1]

        latent = encoder_features[0].view(lq.size(0), -1, self.style_channels)
        encoder_features = encoder_features[1:]

        # generator
        injected_noise = [
            getattr(self.generator, f"injected_noise_{i}")
            for i in range(self.generator.num_injected_noises)
        ]
        # 4x4 stage
        out = self.generator.constant_input(latent)
        out = self.generator.conv1(out, latent[:, 0], noise=injected_noise[0])
        skip = self.generator.to_rgb1(out, latent[:, 1])

        _index = 1

        # 8x8 ---> higher res
        generator_features = []
        for up_conv, conv, noise1, noise2, to_rgb in zip(
            self.generator.convs[::2],
            self.generator.convs[1::2],
            injected_noise[1::2],
            injected_noise[2::2],
            self.generator.to_rgbs,
        ):
            # feature fusion by channel-wise concatenation
            if out.size(2) <= self.in_size:
                fusion_index = (_index - 1) // 2
                feat = encoder_features[fusion_index]

                out = torch.cat([out, feat], dim=1)
                out = self.fusion_out[fusion_index](out)

                skip = torch.cat([skip, feat], dim=1)
                skip = self.fusion_skip[fusion_index](skip)

            # original StyleGAN operations
            out = up_conv(out, latent[:, _index], noise=noise1)
            out = conv(out, latent[:, _index + 1], noise=noise2)
            skip = to_rgb(out, latent[:, _index + 2], skip)

            # store features for decoder
            if out.size(2) > self.in_size:
                generator_features.append(out)

            _index += 2

        # decoder
        hr = encoder_features[-1]
        for i, block in enumerate(self.decoder):
            if i > 0:
                hr = torch.cat([hr, generator_features[i - 1]], dim=1)
            hr = block(hr)

        return hr

    def init_weights(self, pretrained=None, strict=True):
        # Init weights for models.

        # Args:
        #    pretrained (str, optional): Path for pretrained weights. If given
        #        None, pretrained weights will not be loaded. Defaults to None.
        #    strict (boo, optional): Whether strictly load the pretrained model.
        #        Defaults to True.

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(
                f'"pretrained" must be a str or None. '
                f"But received {type(pretrained)}."
            )


class RRDBFeatureExtractor(nn.Module):
    """Feature extractor composed of Residual-in-Residual Dense Blocks (RRDBs).

    It is equivalent to ESRGAN with the upsampling module removed.

    Args:
        in_channels (int): Channel number of inputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64
        num_blocks (int): Block number in the trunk network. Default: 23
        growth_channels (int): Channels for each growth. Default: 32.
    """

    def __init__(
        self, in_channels=3, mid_channels=64, num_blocks=23, growth_channels=32
    ):
        super().__init__()

        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.body = make_layer(
            RRDB, num_blocks, mid_channels=mid_channels, growth_channels=growth_channels
        )
        self.conv_body = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        feat = self.conv_first(x)
        return feat + self.conv_body(self.body(feat))
