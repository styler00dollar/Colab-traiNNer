"""
gfpganv1_arch.py (4-jun-2021)
https://github.com/TencentARC/GFPGAN/blob/master/archs/gfpganv1_arch.py

stylegan2_arch.py (4-jun-2021)
https://github.com/xinntao/BasicSR/blob/cf1e32cdfc9710041ed497e9f5c155ceb7567c8f/basicsr/archs/stylegan2_arch.py

fused_act.py (4-jun-2021)
https://github.com/xinntao/BasicSR/blob/cf1e32cdfc9710041ed497e9f5c155ceb7567c8f/basicsr/ops/fused_act/fused_act.py

upfirdn2d.py (4-jun-2021)
https://github.com/xinntao/BasicSR/blob/cf1e32cdfc9710041ed497e9f5c155ceb7567c8f/basicsr/ops/upfirdn2d/upfirdn2d.py
"""
# modify from https://github.com/rosinality/stylegan2-pytorch/blob/master/op/upfirdn2d.py  # noqa:E501

import torch
from torch.autograd import Function
from torch.nn import functional as F

try:
    from . import upfirdn2d_ext
except ImportError:
    import os
    BASICSR_JIT = os.getenv('BASICSR_JIT')
    if BASICSR_JIT == 'True':
        from torch.utils.cpp_extension import load
        module_path = os.path.dirname(__file__)
        upfirdn2d_ext = load(
            'upfirdn2d',
            sources=[
                os.path.join(module_path, 'src', 'upfirdn2d.cpp'),
                os.path.join(module_path, 'src', 'upfirdn2d_kernel.cu'),
            ],
        )


class UpFirDn2dBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size):

        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad

        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)

        grad_input = upfirdn2d_ext.upfirdn2d(
            grad_output,
            grad_kernel,
            down_x,
            down_y,
            up_x,
            up_y,
            g_pad_x0,
            g_pad_x1,
            g_pad_y0,
            g_pad_y1,
        )
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])

        ctx.save_for_backward(kernel)

        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.in_size = in_size
        ctx.out_size = out_size

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_input):
        kernel, = ctx.saved_tensors

        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)

        gradgrad_out = upfirdn2d_ext.upfirdn2d(
            gradgrad_input,
            kernel,
            ctx.up_x,
            ctx.up_y,
            ctx.down_x,
            ctx.down_y,
            ctx.pad_x0,
            ctx.pad_x1,
            ctx.pad_y0,
            ctx.pad_y1,
        )
        # gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.out_size[0],
        #                                  ctx.out_size[1], ctx.in_size[3])
        gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1])

        return gradgrad_out, None, None, None, None, None, None, None, None


class UpFirDn2d(Function):

    @staticmethod
    def forward(ctx, input, kernel, up, down, pad):
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        kernel_h, kernel_w = kernel.shape
        batch, channel, in_h, in_w = input.shape
        ctx.in_size = input.shape

        input = input.reshape(-1, in_h, in_w, 1)

        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
        ctx.out_size = (out_h, out_w)

        ctx.up = (up_x, up_y)
        ctx.down = (down_x, down_y)
        ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1)

        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1

        ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)

        out = upfirdn2d_ext.upfirdn2d(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1)
        # out = out.view(major, out_h, out_w, minor)
        out = out.view(-1, channel, out_h, out_w)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        kernel, grad_kernel = ctx.saved_tensors

        grad_input = UpFirDn2dBackward.apply(
            grad_output,
            kernel,
            grad_kernel,
            ctx.up,
            ctx.down,
            ctx.pad,
            ctx.g_pad,
            ctx.in_size,
            ctx.out_size,
        )

        return grad_input, None, None, None, None


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    if input.device.type == 'cpu':
        out = upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])
    else:
        out = UpFirDn2d.apply(input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1]))

    return out


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, max(-pad_y0, 0):out.shape[1] - max(-pad_y1, 0), max(-pad_x0, 0):out.shape[2] - max(-pad_x1, 0), :, ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)

# modify from https://github.com/rosinality/stylegan2-pytorch/blob/master/op/fused_act.py # noqa:E501

import torch
from torch import nn
from torch.autograd import Function

try:
    from . import fused_act_ext
except ImportError:
    import os
    BASICSR_JIT = os.getenv('BASICSR_JIT')
    if BASICSR_JIT == 'True':
        from torch.utils.cpp_extension import load
        module_path = os.path.dirname(__file__)
        fused_act_ext = load(
            'fused',
            sources=[
                os.path.join(module_path, 'src', 'fused_bias_act.cpp'),
                os.path.join(module_path, 'src', 'fused_bias_act_kernel.cu'),
            ],
        )

class FusedLeakyReLUFunctionBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, out, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        empty = grad_output.new_empty(0)

        grad_input = fused_act_ext.fused_bias_act(grad_output, empty, out, 3, 1, negative_slope, scale)

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        grad_bias = grad_input.sum(dim).detach()

        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        out, = ctx.saved_tensors
        gradgrad_out = fused_act_ext.fused_bias_act(gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope,
                                                    ctx.scale)

        return gradgrad_out, None, None, None


class FusedLeakyReLUFunction(Function):

    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)
        out = fused_act_ext.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors

        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(grad_output, out, ctx.negative_slope, ctx.scale)

        return grad_input, grad_bias, None, None


class FusedLeakyReLU(nn.Module):

    def __init__(self, channel, negative_slope=0.2, scale=2**0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2**0.5):
    return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)



import math
import random
import torch
from torch import nn
from torch.nn import functional as F


class NormStyleCode(nn.Module):

    def forward(self, x):
        """Normalize the style codes.

        Args:
            x (Tensor): Style codes with shape (b, c).

        Returns:
            Tensor: Normalized tensor.
        """
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


def make_resample_kernel(k):
    """Make resampling kernel for UpFirDn.

    Args:
        k (list[int]): A list indicating the 1D resample kernel magnitude.

    Returns:
        Tensor: 2D resampled kernel.
    """
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]  # to 2D kernel, outer product
    # normalize
    k /= k.sum()
    return k


class UpFirDnUpsample(nn.Module):
    """Upsample, FIR filter, and downsample (upsampole version).

    References:
    1. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.upfirdn.html  # noqa: E501
    2. http://www.ece.northwestern.edu/local-apps/matlabhelp/toolbox/signal/upfirdn.html  # noqa: E501

    Args:
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude.
        factor (int): Upsampling scale factor. Default: 2.
    """

    def __init__(self, resample_kernel, factor=2):
        super(UpFirDnUpsample, self).__init__()
        self.kernel = make_resample_kernel(resample_kernel) * (factor**2)
        self.factor = factor

        pad = self.kernel.shape[0] - factor
        self.pad = ((pad + 1) // 2 + factor - 1, pad // 2)

    def forward(self, x):
        out = upfirdn2d(x, self.kernel.type_as(x), up=self.factor, down=1, pad=self.pad)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(factor={self.factor})')


class UpFirDnDownsample(nn.Module):
    """Upsample, FIR filter, and downsample (downsampole version).

    Args:
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude.
        factor (int): Downsampling scale factor. Default: 2.
    """

    def __init__(self, resample_kernel, factor=2):
        super(UpFirDnDownsample, self).__init__()
        self.kernel = make_resample_kernel(resample_kernel)
        self.factor = factor

        pad = self.kernel.shape[0] - factor
        self.pad = ((pad + 1) // 2, pad // 2)

    def forward(self, x):
        out = upfirdn2d(x, self.kernel.type_as(x), up=1, down=self.factor, pad=self.pad)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(factor={self.factor})')


class UpFirDnSmooth(nn.Module):
    """Upsample, FIR filter, and downsample (smooth version).

    Args:
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude.
        upsample_factor (int): Upsampling scale factor. Default: 1.
        downsample_factor (int): Downsampling scale factor. Default: 1.
        kernel_size (int): Kernel size: Deafult: 1.
    """

    def __init__(self, resample_kernel, upsample_factor=1, downsample_factor=1, kernel_size=1):
        super(UpFirDnSmooth, self).__init__()
        self.upsample_factor = upsample_factor
        self.downsample_factor = downsample_factor
        self.kernel = make_resample_kernel(resample_kernel)
        if upsample_factor > 1:
            self.kernel = self.kernel * (upsample_factor**2)

        if upsample_factor > 1:
            pad = (self.kernel.shape[0] - upsample_factor) - (kernel_size - 1)
            self.pad = ((pad + 1) // 2 + upsample_factor - 1, pad // 2 + 1)
        elif downsample_factor > 1:
            pad = (self.kernel.shape[0] - downsample_factor) + (kernel_size - 1)
            self.pad = ((pad + 1) // 2, pad // 2)
        else:
            raise NotImplementedError

    def forward(self, x):
        out = upfirdn2d(x, self.kernel.type_as(x), up=1, down=1, pad=self.pad)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(upsample_factor={self.upsample_factor}'
                f', downsample_factor={self.downsample_factor})')


class EqualLinear(nn.Module):
    """Equalized Linear as StyleGAN2.

    Args:
        in_channels (int): Size of each sample.
        out_channels (int): Size of each output sample.
        bias (bool): If set to ``False``, the layer will not learn an additive
            bias. Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
        lr_mul (float): Learning rate multiplier. Default: 1.
        activation (None | str): The activation after ``linear`` operation.
            Supported: 'fused_lrelu', None. Default: None.
    """

    def __init__(self, in_channels, out_channels, bias=True, bias_init_val=0, lr_mul=1, activation=None):
        super(EqualLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lr_mul = lr_mul
        self.activation = activation
        if self.activation not in ['fused_lrelu', None]:
            raise ValueError(f'Wrong activation value in EqualLinear: {activation}'
                             "Supported ones are: ['fused_lrelu', None].")
        self.scale = (1 / math.sqrt(in_channels)) * lr_mul

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        if self.bias is None:
            bias = None
        else:
            bias = self.bias * self.lr_mul
        if self.activation == 'fused_lrelu':
            out = F.linear(x, self.weight * self.scale)
            out = fused_leaky_relu(out, bias)
        else:
            out = F.linear(x, self.weight * self.scale, bias=bias)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, bias={self.bias is not None})')


class ModulatedConv2d(nn.Module):
    """Modulated Conv2d used in StyleGAN2.

    There is no bias in ModulatedConv2d.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        num_style_feat (int): Channel number of style features.
        demodulate (bool): Whether to demodulate in the conv layer.
            Default: True.
        sample_mode (str | None): Indicating 'upsample', 'downsample' or None.
            Default: None.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. Default: (1, 3, 3, 1).
        eps (float): A value added to the denominator for numerical stability.
            Default: 1e-8.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_style_feat,
                 demodulate=True,
                 sample_mode=None,
                 resample_kernel=(1, 3, 3, 1),
                 eps=1e-8):
        super(ModulatedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.sample_mode = sample_mode
        self.eps = eps

        if self.sample_mode == 'upsample':
            self.smooth = UpFirDnSmooth(
                resample_kernel, upsample_factor=2, downsample_factor=1, kernel_size=kernel_size)
        elif self.sample_mode == 'downsample':
            self.smooth = UpFirDnSmooth(
                resample_kernel, upsample_factor=1, downsample_factor=2, kernel_size=kernel_size)
        elif self.sample_mode is None:
            pass
        else:
            raise ValueError(f'Wrong sample mode {self.sample_mode}, '
                             "supported ones are ['upsample', 'downsample', None].")

        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)
        # modulation inside each modulated conv
        self.modulation = EqualLinear(
            num_style_feat, in_channels, bias=True, bias_init_val=1, lr_mul=1, activation=None)

        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size, kernel_size))
        self.padding = kernel_size // 2

    def forward(self, x, style):
        """Forward function.

        Args:
            x (Tensor): Tensor with shape (b, c, h, w).
            style (Tensor): Tensor with shape (b, num_style_feat).

        Returns:
            Tensor: Modulated tensor after convolution.
        """
        b, c, h, w = x.shape  # c = c_in
        # weight modulation
        style = self.modulation(style).view(b, 1, c, 1, 1)
        # self.weight: (1, c_out, c_in, k, k); style: (b, 1, c, 1, 1)
        weight = self.scale * self.weight * style  # (b, c_out, c_in, k, k)

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(b, self.out_channels, 1, 1, 1)

        weight = weight.view(b * self.out_channels, c, self.kernel_size, self.kernel_size)

        if self.sample_mode == 'upsample':
            x = x.view(1, b * c, h, w)
            weight = weight.view(b, self.out_channels, c, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(b * c, self.out_channels, self.kernel_size, self.kernel_size)
            out = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=b)
            out = out.view(b, self.out_channels, *out.shape[2:4])
            out = self.smooth(out)
        elif self.sample_mode == 'downsample':
            x = self.smooth(x)
            x = x.view(1, b * c, *x.shape[2:4])
            out = F.conv2d(x, weight, padding=0, stride=2, groups=b)
            out = out.view(b, self.out_channels, *out.shape[2:4])
        else:
            x = x.view(1, b * c, h, w)
            # weight: (b*c_out, c_in, k, k), groups=b
            out = F.conv2d(x, weight, padding=self.padding, groups=b)
            out = out.view(b, self.out_channels, *out.shape[2:4])

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, '
                f'demodulate={self.demodulate}, sample_mode={self.sample_mode})')


class StyleConv(nn.Module):
    """Style conv.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        num_style_feat (int): Channel number of style features.
        demodulate (bool): Whether demodulate in the conv layer. Default: True.
        sample_mode (str | None): Indicating 'upsample', 'downsample' or None.
            Default: None.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. Default: (1, 3, 3, 1).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_style_feat,
                 demodulate=True,
                 sample_mode=None,
                 resample_kernel=(1, 3, 3, 1)):
        super(StyleConv, self).__init__()
        self.modulated_conv = ModulatedConv2d(
            in_channels,
            out_channels,
            kernel_size,
            num_style_feat,
            demodulate=demodulate,
            sample_mode=sample_mode,
            resample_kernel=resample_kernel)
        self.weight = nn.Parameter(torch.zeros(1))  # for noise injection
        self.activate = FusedLeakyReLU(out_channels)

    def forward(self, x, style, noise=None):
        # modulate
        out = self.modulated_conv(x, style)
        # noise injection
        if noise is None:
            b, _, h, w = out.shape
            noise = out.new_empty(b, 1, h, w).normal_()
        out = out + self.weight * noise
        # activation (with bias)
        out = self.activate(out)
        return out


class ToRGB(nn.Module):
    """To RGB from features.

    Args:
        in_channels (int): Channel number of input.
        num_style_feat (int): Channel number of style features.
        upsample (bool): Whether to upsample. Default: True.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. Default: (1, 3, 3, 1).
    """

    def __init__(self, in_channels, num_style_feat, upsample=True, resample_kernel=(1, 3, 3, 1)):
        super(ToRGB, self).__init__()
        if upsample:
            self.upsample = UpFirDnUpsample(resample_kernel, factor=2)
        else:
            self.upsample = None
        self.modulated_conv = ModulatedConv2d(
            in_channels, 3, kernel_size=1, num_style_feat=num_style_feat, demodulate=False, sample_mode=None)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, style, skip=None):
        """Forward function.

        Args:
            x (Tensor): Feature tensor with shape (b, c, h, w).
            style (Tensor): Tensor with shape (b, num_style_feat).
            skip (Tensor): Base/skip tensor. Default: None.

        Returns:
            Tensor: RGB images.
        """
        out = self.modulated_conv(x, style)
        out = out + self.bias
        if skip is not None:
            if self.upsample:
                skip = self.upsample(skip)
            out = out + skip
        return out


class ConstantInput(nn.Module):
    """Constant input.

    Args:
        num_channel (int): Channel number of constant input.
        size (int): Spatial size of constant input.
    """

    def __init__(self, num_channel, size):
        super(ConstantInput, self).__init__()
        self.weight = nn.Parameter(torch.randn(1, num_channel, size, size))

    def forward(self, batch):
        out = self.weight.repeat(batch, 1, 1, 1)
        return out


@ARCH_REGISTRY.register()
class StyleGAN2Generator(nn.Module):
    """StyleGAN2 Generator.

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Layer number of MLP style layers. Default: 8.
        channel_multiplier (int): Channel multiplier for large networks of
            StyleGAN2. Default: 2.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. A cross production will be applied to extent 1D resample
            kenrel to 2D resample kernel. Default: (1, 3, 3, 1).
        lr_mlp (float): Learning rate multiplier for mlp layers. Default: 0.01.
        narrow (float): Narrow ratio for channels. Default: 1.0.
    """

    def __init__(self,
                 out_size,
                 num_style_feat=512,
                 num_mlp=8,
                 channel_multiplier=2,
                 resample_kernel=(1, 3, 3, 1),
                 lr_mlp=0.01,
                 narrow=1):
        super(StyleGAN2Generator, self).__init__()
        # Style MLP layers
        self.num_style_feat = num_style_feat
        style_mlp_layers = [NormStyleCode()]
        for i in range(num_mlp):
            style_mlp_layers.append(
                EqualLinear(
                    num_style_feat, num_style_feat, bias=True, bias_init_val=0, lr_mul=lr_mlp,
                    activation='fused_lrelu'))
        self.style_mlp = nn.Sequential(*style_mlp_layers)

        channels = {
            '4': int(512 * narrow),
            '8': int(512 * narrow),
            '16': int(512 * narrow),
            '32': int(512 * narrow),
            '64': int(256 * channel_multiplier * narrow),
            '128': int(128 * channel_multiplier * narrow),
            '256': int(64 * channel_multiplier * narrow),
            '512': int(32 * channel_multiplier * narrow),
            '1024': int(16 * channel_multiplier * narrow)
        }
        self.channels = channels

        self.constant_input = ConstantInput(channels['4'], size=4)
        self.style_conv1 = StyleConv(
            channels['4'],
            channels['4'],
            kernel_size=3,
            num_style_feat=num_style_feat,
            demodulate=True,
            sample_mode=None,
            resample_kernel=resample_kernel)
        self.to_rgb1 = ToRGB(channels['4'], num_style_feat, upsample=False, resample_kernel=resample_kernel)

        self.log_size = int(math.log(out_size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.num_latent = self.log_size * 2 - 2

        self.style_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channels = channels['4']
        # noise
        for layer_idx in range(self.num_layers):
            resolution = 2**((layer_idx + 5) // 2)
            shape = [1, 1, resolution, resolution]
            self.noises.register_buffer(f'noise{layer_idx}', torch.randn(*shape))
        # style convs and to_rgbs
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            self.style_convs.append(
                StyleConv(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    sample_mode='upsample',
                    resample_kernel=resample_kernel,
                ))
            self.style_convs.append(
                StyleConv(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    sample_mode=None,
                    resample_kernel=resample_kernel))
            self.to_rgbs.append(ToRGB(out_channels, num_style_feat, upsample=True, resample_kernel=resample_kernel))
            in_channels = out_channels

    def make_noise(self):
        """Make noise for noise injection."""
        device = self.constant_input.weight.device
        noises = [torch.randn(1, 1, 4, 4, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2**i, 2**i, device=device))

        return noises

    def get_latent(self, x):
        return self.style_mlp(x)

    def mean_latent(self, num_latent):
        latent_in = torch.randn(num_latent, self.num_style_feat, device=self.constant_input.weight.device)
        latent = self.style_mlp(latent_in).mean(0, keepdim=True)
        return latent

    def forward(self,
                styles,
                input_is_latent=False,
                noise=None,
                randomize_noise=True,
                truncation=1,
                truncation_latent=None,
                inject_index=None,
                return_latents=False):
        """Forward function for StyleGAN2Generator.

        Args:
            styles (list[Tensor]): Sample codes of styles.
            input_is_latent (bool): Whether input is latent style.
                Default: False.
            noise (Tensor | None): Input noise or None. Default: None.
            randomize_noise (bool): Randomize noise, used when 'noise' is
                False. Default: True.
            truncation (float): TODO. Default: 1.
            truncation_latent (Tensor | None): TODO. Default: None.
            inject_index (int | None): The injection index for mixing noise.
                Default: None.
            return_latents (bool): Whether to return style latents.
                Default: False.
        """
        # style codes -> latents with Style MLP layer
        if not input_is_latent:
            styles = [self.style_mlp(s) for s in styles]
        # noises
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers  # for each style conv layer
            else:  # use the stored noise
                noise = [getattr(self.noises, f'noise{i}') for i in range(self.num_layers)]
        # style truncation
        if truncation < 1:
            style_truncation = []
            for style in styles:
                style_truncation.append(truncation_latent + truncation * (style - truncation_latent))
            styles = style_truncation
        # get style latent with injection
        if len(styles) == 1:
            inject_index = self.num_latent

            if styles[0].ndim < 3:
                # repeat latent code for all the layers
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:  # used for encoder with different latent code for each layer
                latent = styles[0]
        elif len(styles) == 2:  # mixing noises
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.num_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)

        # main generation
        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.style_convs[::2], self.style_convs[1::2], noise[1::2],
                                                        noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        image = skip

        if return_latents:
            return image, latent
        else:
            return image, None


class ScaledLeakyReLU(nn.Module):
    """Scaled LeakyReLU.

    Args:
        negative_slope (float): Negative slope. Default: 0.2.
    """

    def __init__(self, negative_slope=0.2):
        super(ScaledLeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        out = F.leaky_relu(x, negative_slope=self.negative_slope)
        return out * math.sqrt(2)


class EqualConv2d(nn.Module):
    """Equalized Linear as StyleGAN2.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: 1
        padding (int): Zero-padding added to both sides of the input.
            Default: 0.
        bias (bool): If ``True``, adds a learnable bias to the output.
            Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, bias_init_val=0):
        super(EqualConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        out = F.conv2d(
            x,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size},'
                f' stride={self.stride}, padding={self.padding}, '
                f'bias={self.bias is not None})')


class ConvLayer(nn.Sequential):
    """Conv Layer used in StyleGAN2 Discriminator.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Kernel size.
        downsample (bool): Whether downsample by a factor of 2.
            Default: False.
        resample_kernel (list[int]): A list indicating the 1D resample
            kernel magnitude. A cross production will be applied to
            extent 1D resample kenrel to 2D resample kernel.
            Default: (1, 3, 3, 1).
        bias (bool): Whether with bias. Default: True.
        activate (bool): Whether use activateion. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 downsample=False,
                 resample_kernel=(1, 3, 3, 1),
                 bias=True,
                 activate=True):
        layers = []
        # downsample
        if downsample:
            layers.append(
                UpFirDnSmooth(resample_kernel, upsample_factor=1, downsample_factor=2, kernel_size=kernel_size))
            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2
        # conv
        layers.append(
            EqualConv2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=self.padding, bias=bias
                and not activate))
        # activation
        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channels))
            else:
                layers.append(ScaledLeakyReLU(0.2))

        super(ConvLayer, self).__init__(*layers)


class ResBlock(nn.Module):
    """Residual block used in StyleGAN2 Discriminator.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        resample_kernel (list[int]): A list indicating the 1D resample
            kernel magnitude. A cross production will be applied to
            extent 1D resample kenrel to 2D resample kernel.
            Default: (1, 3, 3, 1).
    """

    def __init__(self, in_channels, out_channels, resample_kernel=(1, 3, 3, 1)):
        super(ResBlock, self).__init__()

        self.conv1 = ConvLayer(in_channels, in_channels, 3, bias=True, activate=True)
        self.conv2 = ConvLayer(
            in_channels, out_channels, 3, downsample=True, resample_kernel=resample_kernel, bias=True, activate=True)
        self.skip = ConvLayer(
            in_channels, out_channels, 1, downsample=True, resample_kernel=resample_kernel, bias=False, activate=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.skip(x)
        out = (out + skip) / math.sqrt(2)
        return out


@ARCH_REGISTRY.register()
class StyleGAN2Discriminator(nn.Module):
    """StyleGAN2 Discriminator.

    Args:
        out_size (int): The spatial size of outputs.
        channel_multiplier (int): Channel multiplier for large networks of
            StyleGAN2. Default: 2.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. A cross production will be applied to extent 1D resample
            kenrel to 2D resample kernel. Default: (1, 3, 3, 1).
        stddev_group (int): For group stddev statistics. Default: 4.
        narrow (float): Narrow ratio for channels. Default: 1.0.
    """

    def __init__(self, out_size, channel_multiplier=2, resample_kernel=(1, 3, 3, 1), stddev_group=4, narrow=1):
        super(StyleGAN2Discriminator, self).__init__()

        channels = {
            '4': int(512 * narrow),
            '8': int(512 * narrow),
            '16': int(512 * narrow),
            '32': int(512 * narrow),
            '64': int(256 * channel_multiplier * narrow),
            '128': int(128 * channel_multiplier * narrow),
            '256': int(64 * channel_multiplier * narrow),
            '512': int(32 * channel_multiplier * narrow),
            '1024': int(16 * channel_multiplier * narrow)
        }

        log_size = int(math.log(out_size, 2))

        conv_body = [ConvLayer(3, channels[f'{out_size}'], 1, bias=True, activate=True)]

        in_channels = channels[f'{out_size}']
        for i in range(log_size, 2, -1):
            out_channels = channels[f'{2**(i - 1)}']
            conv_body.append(ResBlock(in_channels, out_channels, resample_kernel))
            in_channels = out_channels
        self.conv_body = nn.Sequential(*conv_body)

        self.final_conv = ConvLayer(in_channels + 1, channels['4'], 3, bias=True, activate=True)
        self.final_linear = nn.Sequential(
            EqualLinear(
                channels['4'] * 4 * 4, channels['4'], bias=True, bias_init_val=0, lr_mul=1, activation='fused_lrelu'),
            EqualLinear(channels['4'], 1, bias=True, bias_init_val=0, lr_mul=1, activation=None),
        )
        self.stddev_group = stddev_group
        self.stddev_feat = 1

    def forward(self, x):
        out = self.conv_body(x)

        b, c, h, w = out.shape
        # concatenate a group stddev statistics to out
        group = min(b, self.stddev_group)  # Minibatch must be divisible by (or smaller than) group_size
        stddev = out.view(group, -1, self.stddev_feat, c // self.stddev_feat, h, w)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, h, w)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = out.view(b, -1)
        out = self.final_linear(out)

        return out



import math
import random
import torch
from torch import nn
from torch.nn import functional as F

from basicsr.ops.fused_act import FusedLeakyReLU

class StyleGAN2GeneratorSFT(StyleGAN2Generator):
    """StyleGAN2 Generator.

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Layer number of MLP style layers. Default: 8.
        channel_multiplier (int): Channel multiplier for large networks of
            StyleGAN2. Default: 2.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. A cross production will be applied to extent 1D resample
            kenrel to 2D resample kernel. Default: [1, 3, 3, 1].
        lr_mlp (float): Learning rate multiplier for mlp layers. Default: 0.01.
    """

    def __init__(self,
                 out_size,
                 num_style_feat=512,
                 num_mlp=8,
                 channel_multiplier=2,
                 resample_kernel=(1, 3, 3, 1),
                 lr_mlp=0.01,
                 narrow=1,
                 sft_half=False):
        super(StyleGAN2GeneratorSFT, self).__init__(
            out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            resample_kernel=resample_kernel,
            lr_mlp=lr_mlp,
            narrow=narrow)
        self.sft_half = sft_half

    def forward(self,
                styles,
                conditions,
                input_is_latent=False,
                noise=None,
                randomize_noise=True,
                truncation=1,
                truncation_latent=None,
                inject_index=None,
                return_latents=False):
        """Forward function for StyleGAN2Generator.

        Args:
            styles (list[Tensor]): Sample codes of styles.
            input_is_latent (bool): Whether input is latent style.
                Default: False.
            noise (Tensor | None): Input noise or None. Default: None.
            randomize_noise (bool): Randomize noise, used when 'noise' is
                False. Default: True.
            truncation (float): TODO. Default: 1.
            truncation_latent (Tensor | None): TODO. Default: None.
            inject_index (int | None): The injection index for mixing noise.
                Default: None.
            return_latents (bool): Whether to return style latents.
                Default: False.
        """
        # style codes -> latents with Style MLP layer
        if not input_is_latent:
            styles = [self.style_mlp(s) for s in styles]
        # noises
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers  # for each style conv layer
            else:  # use the stored noise
                noise = [getattr(self.noises, f'noise{i}') for i in range(self.num_layers)]
        # style truncation
        if truncation < 1:
            style_truncation = []
            for style in styles:
                style_truncation.append(truncation_latent + truncation * (style - truncation_latent))
            styles = style_truncation
        # get style latent with injection
        if len(styles) == 1:
            inject_index = self.num_latent

            if styles[0].ndim < 3:
                # repeat latent code for all the layers
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:  # used for encoder with different latent code for each layer
                latent = styles[0]
        elif len(styles) == 2:  # mixing noises
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.num_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)

        # main generation
        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.style_convs[::2], self.style_convs[1::2], noise[1::2],
                                                        noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)

            # the conditions may have fewer levels
            if i < len(conditions):
                # SFT part to combine the conditions
                if self.sft_half:
                    out_same, out_sft = torch.split(out, int(out.size(1) // 2), dim=1)
                    out_sft = out_sft * conditions[i - 1] + conditions[i]
                    out = torch.cat([out_same, out_sft], dim=1)
                else:
                    out = out * conditions[i - 1] + conditions[i]

            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        image = skip

        if return_latents:
            return image, latent
        else:
            return image, None


class ConvUpLayer(nn.Module):
    """Conv Up Layer. Bilinear upsample + Conv.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: 1
        padding (int): Zero-padding added to both sides of the input.
            Default: 0.
        bias (bool): If ``True``, adds a learnable bias to the output.
            Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
        activate (bool): Whether use activateion. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 bias_init_val=0,
                 activate=True):
        super(ConvUpLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

        if bias and not activate:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter('bias', None)

        # activation
        if activate:
            if bias:
                self.activation = FusedLeakyReLU(out_channels)
            else:
                self.activation = ScaledLeakyReLU(0.2)
        else:
            self.activation = None

    def forward(self, x):
        # bilinear upsample
        out = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # conv
        out = F.conv2d(
            out,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        # activation
        if self.activation is not None:
            out = self.activation(out)
        return out


class ResUpBlock(nn.Module):
    """Residual block with upsampling.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
    """

    def __init__(self, in_channels, out_channels):
        super(ResUpBlock, self).__init__()

        self.conv1 = ConvLayer(in_channels, in_channels, 3, bias=True, activate=True)
        self.conv2 = ConvUpLayer(in_channels, out_channels, 3, stride=1, padding=1, bias=True, activate=True)
        self.skip = ConvUpLayer(in_channels, out_channels, 1, bias=False, activate=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.skip(x)
        out = (out + skip) / math.sqrt(2)
        return out


#@ARCH_REGISTRY.register()
class GFPGANv1(nn.Module):
    """Unet + StyleGAN2 decoder with SFT."""

    def __init__(
            self,
            out_size,
            num_style_feat=512,
            channel_multiplier=1,
            resample_kernel=(1, 3, 3, 1),
            decoder_load_path=None,
            fix_decoder=True,
            # for stylegan decoder
            num_mlp=8,
            lr_mlp=0.01,
            input_is_latent=False,
            different_w=False,
            narrow=1,
            sft_half=False):

        super(GFPGANv1, self).__init__()
        self.input_is_latent = input_is_latent
        self.different_w = different_w
        self.num_style_feat = num_style_feat

        unet_narrow = narrow * 0.5
        channels = {
            '4': int(512 * unet_narrow),
            '8': int(512 * unet_narrow),
            '16': int(512 * unet_narrow),
            '32': int(512 * unet_narrow),
            '64': int(256 * channel_multiplier * unet_narrow),
            '128': int(128 * channel_multiplier * unet_narrow),
            '256': int(64 * channel_multiplier * unet_narrow),
            '512': int(32 * channel_multiplier * unet_narrow),
            '1024': int(16 * channel_multiplier * unet_narrow)
        }

        self.log_size = int(math.log(out_size, 2))
        first_out_size = 2**(int(math.log(out_size, 2)))

        self.conv_body_first = ConvLayer(3, channels[f'{first_out_size}'], 1, bias=True, activate=True)

        # downsample
        in_channels = channels[f'{first_out_size}']
        self.conv_body_down = nn.ModuleList()
        for i in range(self.log_size, 2, -1):
            out_channels = channels[f'{2**(i - 1)}']
            self.conv_body_down.append(ResBlock(in_channels, out_channels, resample_kernel))
            in_channels = out_channels

        self.final_conv = ConvLayer(in_channels, channels['4'], 3, bias=True, activate=True)

        # upsample
        in_channels = channels['4']
        self.conv_body_up = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            self.conv_body_up.append(ResUpBlock(in_channels, out_channels))
            in_channels = out_channels

        # to RGB
        self.toRGB = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            self.toRGB.append(EqualConv2d(channels[f'{2**i}'], 3, 1, stride=1, padding=0, bias=True, bias_init_val=0))

        if different_w:
            linear_out_channel = (int(math.log(out_size, 2)) * 2 - 2) * num_style_feat
        else:
            linear_out_channel = num_style_feat

        self.final_linear = EqualLinear(
            channels['4'] * 4 * 4, linear_out_channel, bias=True, bias_init_val=0, lr_mul=1, activation=None)

        self.stylegan_decoder = StyleGAN2GeneratorSFT(
            out_size=out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            resample_kernel=resample_kernel,
            lr_mlp=lr_mlp,
            narrow=narrow,
            sft_half=sft_half)

        if decoder_load_path:
            self.stylegan_decoder.load_state_dict(
                torch.load(decoder_load_path, map_location=lambda storage, loc: storage)['params_ema'])
        if fix_decoder:
            for _, param in self.stylegan_decoder.named_parameters():
                param.requires_grad = False

        # for SFT
        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            if sft_half:
                sft_out_channels = out_channels
            else:
                sft_out_channels = out_channels * 2
            self.condition_scale.append(
                nn.Sequential(
                    EqualConv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True, bias_init_val=0),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(out_channels, sft_out_channels, 3, stride=1, padding=1, bias=True, bias_init_val=1)))
            self.condition_shift.append(
                nn.Sequential(
                    EqualConv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True, bias_init_val=0),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(out_channels, sft_out_channels, 3, stride=1, padding=1, bias=True, bias_init_val=0)))

    def forward(self,
                x,
                return_latents=False,
                save_feat_path=None,
                load_feat_path=None,
                return_rgb=True,
                randomize_noise=True):
        conditions = []
        unet_skips = []
        out_rgbs = []

        # encoder
        feat = self.conv_body_first(x)
        for i in range(self.log_size - 2):
            feat = self.conv_body_down[i](feat)
            unet_skips.insert(0, feat)

        feat = self.final_conv(feat)

        # style code
        style_code = self.final_linear(feat.view(feat.size(0), -1))
        if self.different_w:
            style_code = style_code.view(style_code.size(0), -1, self.num_style_feat)

        # decode
        for i in range(self.log_size - 2):
            # add unet skip
            feat = feat + unet_skips[i]
            # ResUpLayer
            feat = self.conv_body_up[i](feat)
            # generate scale and shift for SFT layer
            scale = self.condition_scale[i](feat)
            conditions.append(scale.clone())
            shift = self.condition_shift[i](feat)
            conditions.append(shift.clone())
            # generate rgb images
            if return_rgb:
                out_rgbs.append(self.toRGB[i](feat))

        if save_feat_path is not None:
            torch.save(conditions, save_feat_path)
        if load_feat_path is not None:
            conditions = torch.load(load_feat_path)
            conditions = [v.cuda() for v in conditions]

        # decoder
        image, _ = self.stylegan_decoder([style_code],
                                         conditions,
                                         return_latents=return_latents,
                                         input_is_latent=self.input_is_latent,
                                         randomize_noise=randomize_noise)

        return image, out_rgbs


@ARCH_REGISTRY.register()
class FacialComponentDiscriminator(nn.Module):

    def __init__(self):
        super(FacialComponentDiscriminator, self).__init__()

        self.conv1 = ConvLayer(3, 64, 3, downsample=False, resample_kernel=(1, 3, 3, 1), bias=True, activate=True)
        self.conv2 = ConvLayer(64, 128, 3, downsample=True, resample_kernel=(1, 3, 3, 1), bias=True, activate=True)
        self.conv3 = ConvLayer(128, 128, 3, downsample=False, resample_kernel=(1, 3, 3, 1), bias=True, activate=True)
        self.conv4 = ConvLayer(128, 256, 3, downsample=True, resample_kernel=(1, 3, 3, 1), bias=True, activate=True)
        self.conv5 = ConvLayer(256, 256, 3, downsample=False, resample_kernel=(1, 3, 3, 1), bias=True, activate=True)
        self.final_conv = ConvLayer(256, 1, 3, bias=True, activate=False)

    def forward(self, x, return_feats=False):
        feat = self.conv1(x)
        feat = self.conv3(self.conv2(feat))
        rlt_feats = []
        if return_feats:
            rlt_feats.append(feat.clone())
        feat = self.conv5(self.conv4(feat))
        if return_feats:
            rlt_feats.append(feat.clone())
        out = self.final_conv(feat)

        if return_feats:
            return out, rlt_feats
        else:
            return out, None