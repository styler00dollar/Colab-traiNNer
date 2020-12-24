"""
convnet.py (16-12-20)
https://github.com/zengxianyu/crfill/blob/4ed07a6a373398fcaa4c45fe926c83b20116b967/networks/convnet.py

utils.py (16-12-20)
https://github.com/zengxianyu/crfill/blob/4ed07a6a373398fcaa4c45fe926c83b20116b967/networks/utils.py
"""

from torch.nn.functional import normalize
import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.ConvTranspose2d) and m.in_channels == m.out_channels:
        initial_weight = get_upsampling_weight(
            m.in_channels, m.out_channels, m.kernel_size[0])
        m.weight.data.copy_(initial_weight)


class gen_conv(nn.Conv2d):
    def __init__(self, cin, cout, ksize, stride=1, rate=1, activation=nn.ELU()):
        """Define conv for generator

        Args:
            cin: Input Channel number.
            cout: output Channel number.
            ksize: Kernel size.
            Stride: Convolution stride.
            rate: Rate for or dilated conv.
            activation: Activation function after convolution.
        """
        p = int(rate*(ksize-1)/2)
        super(gen_conv, self).__init__(in_channels=cin, out_channels=cout, kernel_size=ksize, stride=stride, padding=p, dilation=rate, groups=1, bias=True)
        self.activation = activation

    def forward(self, x):
        x = super(gen_conv, self).forward(x)
        if self.out_channels == 3 or self.activation is None:
            return x
        x, y = torch.split(x, int(self.out_channels/2), dim=1)
        x = self.activation(x)
        y = torch.sigmoid(y)
        x = x * y
        return x

class gen_deconv(gen_conv):
    def __init__(self, cin, cout):
        """Define deconv for generator.
        The deconv is defined to be a x2 resize_nearest_neighbor operation with
        additional gen_conv operation.

        Args:
            cin: Input Channel number.
            cout: output Channel number.
            ksize: Kernel size.
        """
        super(gen_deconv, self).__init__(cin, cout, ksize=3)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2)
        x = super(gen_deconv, self).forward(x)
        return x

class dis_conv(nn.Conv2d):
    def __init__(self, cin, cout, ksize=5, stride=2):
        """Define conv for discriminator.
        Activation is set to leaky_relu.

        Args:
            cin: Input Channel number.
            cout: output Channel number.
            ksize: Kernel size.
            Stride: Convolution stride.
        """
        p = int((ksize-1)/2)
        super(dis_conv, self).__init__(in_channels=cin, out_channels=cout, kernel_size=ksize, stride=stride, padding=p, dilation=1, groups=1, bias=True)

    def forward(self, x):
        x = super(dis_conv, self).forward(x)
        x = F.leaky_relu(x)
        return x

def batch_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """Define batch convolution to use different conv. kernels in a batch.

    Args:
        x: input feature maps of shape (batch, channel, height, width)
        weight: conv.kernels of shape (batch, out_channel, in_channels, kernel_size, kernel_size)
    """
    if bias is None:
        assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
    else:
        assert x.shape[0] == weight.shape[0] and bias.shape[0] == weight.shape[
            0], "dim=0 of bias must be equal in size to dim=0 of weight"

    b_i, c, h, w = x.shape
    b_i, out_channels, in_channels, kernel_height_size, kernel_width_size = weight.shape

    out = x[None, ...].view(1, b_i * c, h, w)
    weight = weight.contiguous().view(b_i * out_channels, in_channels, kernel_height_size, kernel_width_size)

    out = F.conv2d(out, weight=weight, bias=None, stride=stride, dilation=dilation, groups=b_i,
                   padding=padding)

    out = out.view(b_i, out_channels, out.shape[-2], out.shape[-1])

    if bias is not None:
        out = out + bias.unsqueeze(2).unsqueeze(3)

    return out


def batch_transposeconv2d(x, weight, bias=None, stride=1, padding=0, output_padding=0, dilation=1):
    """Define batch transposed convolution to use different conv. kernels in a batch.

    Args:
        x: input feature maps of shape (batch, channel, height, width)
        weight: conv.kernels of shape (batch, in_channel, out_channels, kernel_size, kernel_size)
    """
    if bias is None:
        assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
    else:
        assert x.shape[0] == weight.shape[0] and bias.shape[0] == weight.shape[
            0], "dim=0 of bias must be equal in size to dim=0 of weight"

    b_i, c, h, w = x.shape
    b_i, in_channels, out_channels, kernel_height_size, kernel_width_size = weight.shape

    out = x[None, ...].view(1, b_i * c, h, w)
    weight = weight.contiguous().view(in_channels*b_i, out_channels, kernel_height_size, kernel_width_size)

    out = F.conv_transpose2d(out, weight=weight, bias=None, stride=stride, dilation=dilation, groups=b_i,
                   padding=padding, output_padding=output_padding)

    out = out.view(b_i, out_channels, out.shape[-2], out.shape[-1])

    if bias is not None:
        out = out + bias.unsqueeze(2).unsqueeze(3)
    return out


class InpaintGenerator(nn.Module):
    def __init__(self, cnum=48):
        super(InpaintGenerator, self).__init__()
        self.cnum = cnum
        # stage1
        self.conv1 = gen_conv(5, cnum, 5, 1)
        self.conv2_downsample = gen_conv(int(cnum/2), 2*cnum, 3, 2)
        self.conv3 = gen_conv(cnum, 2*cnum, 3, 1)
        self.conv4_downsample = gen_conv(cnum, 4*cnum, 3, 2)
        self.conv5 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.conv6 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.conv7_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=2)
        self.conv8_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=4)
        self.conv9_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=8)
        self.conv10_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=16)
        self.conv11 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.conv12 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.conv13_upsample_conv = gen_deconv(2*cnum, 2*cnum)
        self.conv14 = gen_conv(cnum, 2*cnum, 3, 1)
        self.conv15_upsample_conv = gen_deconv(cnum, cnum)
        self.conv16 = gen_conv(cnum//2, cnum//2, 3, 1)
        self.conv17 = gen_conv(cnum//4, 3, 3, 1, activation=None)

        # stage2
        self.xconv1 = gen_conv(3, cnum, 5, 1)
        self.xconv2_downsample = gen_conv(cnum//2, cnum, 3, 2)
        self.xconv3 = gen_conv(cnum//2, 2*cnum, 3, 1)
        self.xconv4_downsample = gen_conv(cnum, 2*cnum, 3, 2)
        self.xconv5 = gen_conv(cnum, 4*cnum, 3, 1)
        self.xconv6 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.xconv7_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=2)
        self.xconv8_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=4)
        self.xconv9_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=8)
        self.xconv10_atrous = gen_conv(2*cnum, 4*cnum, 3, rate=16)
        self.pmconv1 = gen_conv(3, cnum, 5, 1)
        self.pmconv2_downsample = gen_conv(cnum//2, cnum, 3, 2)
        self.pmconv3 = gen_conv(cnum//2, 2*cnum, 3, 1)
        self.pmconv4_downsample = gen_conv(cnum, 4*cnum, 3, 2)
        self.pmconv5 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.pmconv6 = gen_conv(2*cnum, 4*cnum, 3, 1,
                            activation=nn.ReLU())
        self.pmconv9 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.pmconv10 = gen_conv(2*cnum, 4*cnum, 3, 1)

        self.allconv11 = gen_conv(4*cnum, 4*cnum, 3, 1)
        self.allconv12 = gen_conv(2*cnum, 4*cnum, 3, 1)
        self.allconv13_upsample_conv = gen_deconv(2*cnum, 2*cnum)
        self.allconv14 = gen_conv(cnum, 2*cnum, 3, 1)
        self.allconv15_upsample_conv = gen_deconv(cnum, cnum)
        self.allconv16 = gen_conv(cnum//2, cnum//2, 3, 1)
        self.allconv17 = gen_conv(cnum//4, 3, 3, 1, activation=None)

        self.apply(weight_init)


    def forward(self, x, mask):
        xin = x
        bsize, ch, height, width = x.shape
        ones_x = torch.ones(bsize, 1, height, width).to(x.device)
        x = torch.cat([x, ones_x, ones_x*mask], 1)

        # two stage network
        ## stage1
        x = self.conv1(x)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13_upsample_conv(x)
        x = self.conv14(x)
        x = self.conv15_upsample_conv(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = torch.tanh(x)
        x_stage1 = x

        x = x*mask + xin[:, 0:3, :, :]*(1.-mask)
        xnow = x

        ###
        x = self.xconv1(xnow)
        x = self.xconv2_downsample(x)
        x = self.xconv3(x)
        x = self.xconv4_downsample(x)
        x = self.xconv5(x)
        x = self.xconv6(x)
        x = self.xconv7_atrous(x)
        x = self.xconv8_atrous(x)
        x = self.xconv9_atrous(x)
        x = self.xconv10_atrous(x)
        x_hallu = x

        ###
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)

        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], 1)

        x = self.allconv11(x)
        x = self.allconv12(x)
        x = self.allconv13_upsample_conv(x)
        x = self.allconv14(x)
        x = self.allconv15_upsample_conv(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        x_stage2 = torch.tanh(x)

        return x_stage2, x_stage1
