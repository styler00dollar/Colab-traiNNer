"""
network.py (15-12-20)
https://github.com/zhaoyuzhi/deepfillv2/blob/master/deepfillv2/network.py

network_module.py (15-12-20)
https://github.com/zhaoyuzhi/deepfillv2/blob/master/deepfillv2/network_module.py
"""

# from network_module import *
# from .convolutions import partialconv2d
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
import logging
import torch
import torch.nn.init as init

logger = logging.getLogger("base")
import pytorch_lightning as pl


# -----------------------------------------------
#                Normal ConvBlock
# -----------------------------------------------
class Conv2dLayer(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_type,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        pad_type="zero",
        activation="lrelu",
        norm="none",
        sn=False,
    ):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == "replicate":
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == "zero":
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "in":
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == "ln":
            self.norm = LayerNorm(out_channels)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            print("sn")
            self.conv2d = SpectralNorm(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding=0,
                    dilation=dilation,
                )
            )
        else:
            if conv_type == "normal":
                self.conv2d = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding=0,
                    dilation=dilation,
                )
            elif conv_type == "partial":
                self.conv2d = PartialConv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding=0,
                    dilation=dilation,
                )
            else:
                print("conv_type not implemented")

    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class TransposeConv2dLayer(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_type,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        pad_type="zero",
        activation="lrelu",
        norm="none",
        sn=False,
        scale_factor=2,
    ):
        super(TransposeConv2dLayer, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.conv2d = Conv2dLayer(
            in_channels,
            out_channels,
            conv_type,
            kernel_size,
            stride,
            padding,
            dilation,
            pad_type,
            activation,
            norm,
            sn,
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.conv2d(x)
        return x


# -----------------------------------------------
#                Gated ConvBlock
# -----------------------------------------------
class GatedConv2d(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_type,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        pad_type="reflect",
        activation="lrelu",
        norm="none",
        sn=False,
    ):
        super(GatedConv2d, self).__init__()
        # Initialize the padding scheme
        if pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == "replicate":
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == "zero":
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "in":
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == "ln":
            self.norm = LayerNorm(out_channels)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding=0,
                    dilation=dilation,
                )
            )
            self.mask_conv2d = SpectralNorm(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding=0,
                    dilation=dilation,
                )
            )
        else:
            if conv_type == "normal":
                self.conv2d = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding=0,
                    dilation=dilation,
                )
                self.mask_conv2d = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding=0,
                    dilation=dilation,
                )
            elif conv_type == "partial":
                self.conv2d = PartialConv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding=0,
                    dilation=dilation,
                )
                self.mask_conv2d = PartialConv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding=0,
                    dilation=dilation,
                )
            else:
                print("conv_type not implemented")
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.pad(x)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask)
        x = conv * gated_mask
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class TransposeGatedConv2d(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_type,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        pad_type="zero",
        activation="lrelu",
        norm="none",
        sn=True,
        scale_factor=2,
    ):
        super(TransposeGatedConv2d, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.gated_conv2d = GatedConv2d(
            in_channels,
            out_channels,
            conv_type,
            kernel_size,
            stride,
            padding,
            dilation,
            pad_type,
            activation,
            norm,
            sn,
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.gated_conv2d(x)
        return x


# ----------------------------------------
#               Layer Norm
# ----------------------------------------
class LayerNorm(pl.LightningModule):
    def __init__(self, num_features, eps=1e-8, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = Parameter(torch.Tensor(num_features).uniform_())
            self.beta = Parameter(torch.zeros(num_features))

    def forward(self, x):
        # layer norm
        shape = [-1] + [1] * (x.dim() - 1)  # for 4d input: [-1, 1, 1, 1]
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        # if it is learnable
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)  # for 4d input: [1, -1, 1, 1]
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


# -----------------------------------------------
#                  SpectralNorm
# -----------------------------------------------
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(pl.LightningModule):
    def __init__(self, module, name="weight", power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


def deepfillv2_weights_init(net, init_type="kaiming", init_gain=0.02):
    # Initialize network weights.
    # Parameters:
    #    net (network)       -- network to be initialized
    #    init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
    #    init_var (float)    -- scaling factor for normal, xavier and orthogonal.

    def init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, "weight") and classname.find("Conv") != -1:
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif classname.find("Linear") != -1:
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

    # Apply the initialization function <init_func>
    logger.info("Initialization method [{:s}]".format(init_type))
    net.apply(init_func)


# -----------------------------------------------
#                   Generator
# -----------------------------------------------
# Input: masked image + mask
# Output: filled image


# https://github.com/zhaoyuzhi/deepfillv2/blob/62dad2c601400e14d79f4d1e090c2effcb9bf3eb/deepfillv2/train.py
class GatedGenerator(pl.LightningModule):
    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        latent_channels=64,
        pad_type="zero",
        activation="lrelu",
        norm="in",
        conv_type="normal",
    ):
        super(GatedGenerator, self).__init__()

        self.coarse = nn.Sequential(
            # encoder
            GatedConv2d(
                in_channels,
                latent_channels,
                conv_type,
                7,
                1,
                3,
                pad_type=pad_type,
                activation=activation,
                norm="none",
            ),
            GatedConv2d(
                latent_channels,
                latent_channels * 2,
                conv_type,
                4,
                2,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2d(
                latent_channels * 2,
                latent_channels * 4,
                conv_type,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2d(
                latent_channels * 4,
                latent_channels * 4,
                conv_type,
                4,
                2,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            # Bottleneck
            GatedConv2d(
                latent_channels * 4,
                latent_channels * 4,
                conv_type,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2d(
                latent_channels * 4,
                latent_channels * 4,
                conv_type,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2d(
                latent_channels * 4,
                latent_channels * 4,
                conv_type,
                3,
                1,
                2,
                dilation=2,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2d(
                latent_channels * 4,
                latent_channels * 4,
                conv_type,
                3,
                1,
                4,
                dilation=4,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2d(
                latent_channels * 4,
                latent_channels * 4,
                conv_type,
                3,
                1,
                8,
                dilation=8,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2d(
                latent_channels * 4,
                latent_channels * 4,
                conv_type,
                3,
                1,
                16,
                dilation=16,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2d(
                latent_channels * 4,
                latent_channels * 4,
                conv_type,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2d(
                latent_channels * 4,
                latent_channels * 4,
                conv_type,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            # decoder
            TransposeGatedConv2d(
                latent_channels * 4,
                latent_channels * 2,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2d(
                latent_channels * 2,
                latent_channels * 2,
                conv_type,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            TransposeGatedConv2d(
                latent_channels * 2,
                latent_channels,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2d(
                latent_channels,
                out_channels,
                conv_type,
                7,
                1,
                3,
                pad_type=pad_type,
                activation="tanh",
                norm="none",
            ),
        )
        self.refinement = nn.Sequential(
            # encoder
            GatedConv2d(
                in_channels,
                latent_channels,
                conv_type,
                7,
                1,
                3,
                pad_type=pad_type,
                activation=activation,
                norm="none",
            ),
            GatedConv2d(
                latent_channels,
                latent_channels * 2,
                conv_type,
                4,
                2,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2d(
                latent_channels * 2,
                latent_channels * 4,
                conv_type,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2d(
                latent_channels * 4,
                latent_channels * 4,
                conv_type,
                4,
                2,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            # Bottleneck
            GatedConv2d(
                latent_channels * 4,
                latent_channels * 4,
                conv_type,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2d(
                latent_channels * 4,
                latent_channels * 4,
                conv_type,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2d(
                latent_channels * 4,
                latent_channels * 4,
                conv_type,
                3,
                1,
                2,
                dilation=2,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2d(
                latent_channels * 4,
                latent_channels * 4,
                conv_type,
                3,
                1,
                4,
                dilation=4,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2d(
                latent_channels * 4,
                latent_channels * 4,
                conv_type,
                3,
                1,
                8,
                dilation=8,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2d(
                latent_channels * 4,
                latent_channels * 4,
                conv_type,
                3,
                1,
                16,
                dilation=16,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2d(
                latent_channels * 4,
                latent_channels * 4,
                conv_type,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2d(
                latent_channels * 4,
                latent_channels * 4,
                conv_type,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            # decoder
            TransposeGatedConv2d(
                latent_channels * 4,
                latent_channels * 2,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2d(
                latent_channels * 2,
                latent_channels * 2,
                conv_type,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            TransposeGatedConv2d(
                latent_channels * 2,
                latent_channels,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2d(
                latent_channels,
                out_channels,
                conv_type,
                7,
                1,
                3,
                pad_type=pad_type,
                activation="tanh",
                norm="none",
            ),
        )

    def forward(self, img, mask):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # 1 - mask: unmask
        # img * (1 - mask): ground truth unmask region
        # Coarse
        # print(img.shape, mask.shape)
        first_masked_img = img * (1 - mask) + mask
        first_in = torch.cat((first_masked_img, mask), 1)  # in: [B, 4, H, W]
        first_out = self.coarse(first_in)  # out: [B, 3, H, W]
        # Refinement
        second_masked_img = img * (1 - mask) + first_out * mask
        second_in = torch.cat((second_masked_img, mask), 1)  # in: [B, 4, H, W]
        second_out = self.refinement(second_in)  # out: [B, 3, H, W]
        # return first_out, second_out
        # return second_out
        return second_out, first_out
