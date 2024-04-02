"""
DSNet.py (6-3-20)
https://github.com/wangning-001/DSNet/blob/afa174a8f8e4fbdeff086fb546c83c871e959141/modules/DSNet.py

RegionNorm.py (6-3-20)
https://github.com/wangning-001/DSNet/blob/afa174a8f8e4fbdeff086fb546c83c871e959141/modules/RegionNorm.py

ValidMigration.py (6-3-20)
https://github.com/wangning-001/DSNet/blob/afa174a8f8e4fbdeff086fb546c83c871e959141/modules/ValidMigration.py

Attention.py (6-3-20)
https://github.com/wangning-001/DSNet/blob/afa174a8f8e4fbdeff086fb546c83c871e959141/modules/Attention.py

deform_conv.py (6-3-20)
https://github.com/wangning-001/DSNet/blob/afa174a8f8e4fbdeff086fb546c83c871e959141/modules/deform_conv.py
"""

# from modules.Attention import PixelContextualAttention
# from modules.RegionNorm import RBNModule, RCNModule
# from modules.ValidMigration import ConvOffset2D
# from modules.deform_conv import th_batch_map_offsets, th_generate_grid
from __future__ import absolute_import, division
from scipy.ndimage.interpolation import map_coordinates as sp_map_coordinates
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


def th_flatten(a):
    """Flatten tensor"""
    return a.contiguous().view(a.nelement())


def th_repeat(a, repeats, axis=0):
    """Torch version of np.repeat for 1D"""
    assert len(a.size()) == 1
    return th_flatten(torch.transpose(a.repeat(repeats, 1), 0, 1))


def np_repeat_2d(a, repeats):
    """Tensorflow version of np.repeat for 2D"""

    assert len(a.shape) == 2
    a = np.expand_dims(a, 0)
    a = np.tile(a, [repeats, 1, 1])
    return a


def th_gather_2d(input, coords):
    inds = coords[:, 0] * input.size(1) + coords[:, 1]
    x = torch.index_select(th_flatten(input), 0, inds)
    return x.view(coords.size(0))


def th_map_coordinates(input, coords, order=1):
    """Tensorflow verion of scipy.ndimage.map_coordinates
    Note that coords is transposed and only 2D is supported
    Parameters
    ----------
    input : tf.Tensor. shape = (s, s)
    coords : tf.Tensor. shape = (n_points, 2)
    """

    assert order == 1
    input_size = input.size(0)

    coords = torch.clamp(coords, 0, input_size - 1)
    coords_lt = coords.floor().long()
    coords_rb = coords.ceil().long()
    coords_lb = torch.stack([coords_lt[:, 0], coords_rb[:, 1]], 1)
    coords_rt = torch.stack([coords_rb[:, 0], coords_lt[:, 1]], 1)

    vals_lt = th_gather_2d(input, coords_lt.detach())
    vals_rb = th_gather_2d(input, coords_rb.detach())
    vals_lb = th_gather_2d(input, coords_lb.detach())
    vals_rt = th_gather_2d(input, coords_rt.detach())

    coords_offset_lt = coords - coords_lt.type(coords.data.type())

    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, 1]
    return mapped_vals


def sp_batch_map_coordinates(inputs, coords):
    """Reference implementation for batch_map_coordinates"""
    # coords = coords.clip(0, inputs.shape[1] - 1)

    assert coords.shape[2] == 2
    height = coords[:, :, 0].clip(0, inputs.shape[1] - 1)
    width = coords[:, :, 1].clip(0, inputs.shape[2] - 1)
    np.concatenate((np.expand_dims(height, axis=2), np.expand_dims(width, axis=2)), 2)

    mapped_vals = np.array(
        [
            sp_map_coordinates(input, coord.T, mode="nearest", order=1)
            for input, coord in zip(inputs, coords)
        ]
    )
    return mapped_vals


def th_batch_map_coordinates(input, coords, order=1):
    """Batch version of th_map_coordinates
    Only supports 2D feature maps
    Parameters
    ----------
    input : tf.Tensor. shape = (b, s, s)
    coords : tf.Tensor. shape = (b, n_points, 2)
    Returns
    -------
    tf.Tensor. shape = (b, s, s)
    """

    batch_size = input.size(0)
    input_height = input.size(1)
    input_width = input.size(2)

    n_coords = coords.size(1)

    # coords = torch.clamp(coords, 0, input_size - 1)

    coords = torch.cat(
        (
            torch.clamp(coords.narrow(2, 0, 1), 0, input_height - 1),
            torch.clamp(coords.narrow(2, 1, 1), 0, input_width - 1),
        ),
        2,
    )

    assert coords.size(1) == n_coords

    coords_lt = coords.floor().long()
    coords_rb = coords.ceil().long()
    coords_lb = torch.stack([coords_lt[..., 0], coords_rb[..., 1]], 2)
    coords_rt = torch.stack([coords_rb[..., 0], coords_lt[..., 1]], 2)
    idx = th_repeat(torch.arange(0, batch_size), n_coords).long()
    idx = Variable(idx, requires_grad=False)
    if input.is_cuda:
        idx = idx.cuda()

    def _get_vals_by_coords(input, coords):
        indices = torch.stack(
            [idx, th_flatten(coords[..., 0]), th_flatten(coords[..., 1])], 1
        )
        inds = (
            indices[:, 0] * input.size(1) * input.size(2)
            + indices[:, 1] * input.size(2)
            + indices[:, 2]
        )
        vals = th_flatten(input).index_select(0, inds)
        vals = vals.view(batch_size, n_coords)
        return vals

    vals_lt = _get_vals_by_coords(input, coords_lt.detach())
    vals_rb = _get_vals_by_coords(input, coords_rb.detach())
    vals_lb = _get_vals_by_coords(input, coords_lb.detach())
    vals_rt = _get_vals_by_coords(input, coords_rt.detach())

    coords_offset_lt = coords - coords_lt.type(coords.data.type())
    vals_t = coords_offset_lt[..., 0] * (vals_rt - vals_lt) + vals_lt
    vals_b = coords_offset_lt[..., 0] * (vals_rb - vals_lb) + vals_lb
    mapped_vals = coords_offset_lt[..., 1] * (vals_b - vals_t) + vals_t
    return mapped_vals


def sp_batch_map_offsets(input, offsets):
    """Reference implementation for tf_batch_map_offsets"""

    batch_size = input.shape[0]
    input_height = input.shape[1]
    input_width = input.shape[2]

    offsets = offsets.reshape(batch_size, -1, 2)
    grid = np.stack(np.mgrid[:input_height, :input_width], -1).reshape(-1, 2)
    grid = np.repeat([grid], batch_size, axis=0)
    coords = offsets + grid
    # coords = coords.clip(0, input_size - 1)

    mapped_vals = sp_batch_map_coordinates(input, coords)
    return mapped_vals


def th_generate_grid(batch_size, input_height, input_width, dtype, cuda):
    grid = np.meshgrid(range(input_height), range(input_width), indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = grid.reshape(-1, 2)

    grid = np_repeat_2d(grid, batch_size)
    grid = torch.from_numpy(grid).type(dtype)
    if cuda:
        grid = grid.cuda()
    return Variable(grid, requires_grad=False)


def th_batch_map_offsets(input, offsets, grid=None, order=1):
    """Batch map offsets into input
    Parameters
    ---------
    input : torch.Tensor. shape = (b, s, s)
    offsets: torch.Tensor. shape = (b, s, s, 2)
    Returns
    -------
    torch.Tensor. shape = (b, s, s)
    """
    batch_size = input.size(0)
    input_height = input.size(1)
    input_width = input.size(2)

    offsets = offsets.view(batch_size, -1, 2)
    if grid is None:
        grid = th_generate_grid(
            batch_size,
            input_height,
            input_width,
            offsets.data.type(),
            offsets.data.is_cuda,
        )

    coords = offsets + grid

    mapped_vals = th_batch_map_coordinates(input, coords)
    return mapped_vals


class SEModule(pl.LightningModule):
    def __init__(self, num_channel, squeeze_ratio=1.0):
        super(SEModule, self).__init__()
        self.sequeeze_mod = nn.AdaptiveAvgPool2d(1)
        self.num_channel = num_channel

        blocks = [
            nn.Linear(num_channel, int(num_channel * squeeze_ratio)),
            nn.ReLU(),
            nn.Linear(int(num_channel * squeeze_ratio), num_channel),
            nn.Sigmoid(),
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        ori = x
        x = self.sequeeze_mod(x)
        x = x.view(x.size(0), 1, self.num_channel)
        x = self.blocks(x)
        x = x.view(x.size(0), self.num_channel, 1, 1)
        x = ori * x
        return x


class ContextualAttentionModule(pl.LightningModule):
    def __init__(self, patch_size=3, propagate_size=3, stride=1):
        super(ContextualAttentionModule, self).__init__()
        self.patch_size = patch_size
        self.propagate_size = propagate_size
        self.stride = stride
        self.prop_kernels = None

    def forward(self, foreground, masks):
        ###assume the masked area has value 1
        bz, nc, w, h = foreground.size()
        if masks.size(3) != foreground.size(3):
            masks = F.interpolate(masks, foreground.size()[2:])
        background = foreground.clone()
        background = background * masks
        background = F.pad(
            background,
            [
                self.patch_size // 2,
                self.patch_size // 2,
                self.patch_size // 2,
                self.patch_size // 2,
            ],
        )
        conv_kernels_all = (
            background.unfold(2, self.patch_size, self.stride)
            .unfold(3, self.patch_size, self.stride)
            .contiguous()
            .view(bz, nc, -1, self.patch_size, self.patch_size)
        )
        conv_kernels_all = conv_kernels_all.transpose(2, 1)
        output_tensor = []
        for i in range(bz):
            mask = masks[i : i + 1]
            feature_map = foreground[i : i + 1].contiguous()
            # form convolutional kernels
            conv_kernels = conv_kernels_all[i] + 0.0000001
            norm_factor = torch.sum(conv_kernels**2, [1, 2, 3], keepdim=True) ** 0.5
            conv_kernels = conv_kernels / norm_factor

            conv_result = F.conv2d(
                feature_map, conv_kernels, padding=self.patch_size // 2
            )
            """
            if self.propagate_size != 1:
                if self.prop_kernels is None:
                    self.prop_kernels = torch.ones([conv_result.size(1), 1, self.propagate_size, self.propagate_size])
                    self.prop_kernels.requires_grad = False
                    self.prop_kernels = self.prop_kernels.cuda()
                conv_result = F.conv2d(conv_result, self.prop_kernels, stride=1, padding=1, groups=conv_result.size(1))

            """

            self.prop_kernels = torch.ones(
                [conv_result.size(1), 1, self.propagate_size, self.propagate_size]
            )
            self.prop_kernels.requires_grad = False
            self.prop_kernels = self.prop_kernels.cuda()
            conv_result = F.conv2d(
                conv_result,
                self.prop_kernels,
                stride=1,
                padding=1,
                groups=conv_result.size(1),
            )

            attention_scores = F.softmax(conv_result, dim=1)
            ##propagate the scores
            recovered_foreground = F.conv_transpose2d(
                attention_scores, conv_kernels, stride=1, padding=self.patch_size // 2
            )
            # average the recovered value, at the same time make non-masked area 0
            recovered_foreground = (recovered_foreground * (1 - mask)) / (
                self.patch_size**2
            )
            # recover the image
            final_output = recovered_foreground + feature_map * mask
            output_tensor.append(final_output)
        return torch.cat(output_tensor, dim=0)


class PixelContextualAttention(pl.LightningModule):
    def __init__(
        self, inchannel, patch_size_list=[1], propagate_size_list=[3], stride_list=[1]
    ):
        assert isinstance(
            patch_size_list, list
        ), "patch_size should be a list containing scales, or you should use Contextual Attention to initialize your module"
        assert len(patch_size_list) == len(propagate_size_list) and len(
            propagate_size_list
        ) == len(stride_list), "the input_lists should have same lengths"
        super(PixelContextualAttention, self).__init__()
        for i in range(len(patch_size_list)):
            name = "CA_{:d}".format(i)
            setattr(
                self,
                name,
                ContextualAttentionModule(
                    patch_size_list[i], propagate_size_list[i], stride_list[i]
                ),
            )
        self.num_of_modules = len(patch_size_list)
        self.SqueezeExc = SEModule(inchannel * 2)
        self.combiner = nn.Conv2d(inchannel * 2, inchannel, kernel_size=1)

    def forward(self, foreground, mask):
        outputs = [foreground]
        for i in range(self.num_of_modules):
            name = "CA_{:d}".format(i)
            CA_module = getattr(self, name)
            outputs.append(CA_module(foreground, mask))
        outputs = torch.cat(outputs, dim=1)
        outputs = self.SqueezeExc(outputs)
        outputs = self.combiner(outputs)
        return outputs


class ConvOffset2D(nn.Conv2d):
    """ConvOffset2D

    Convolutional layer responsible for learning the 2D offsets and output the
    deformed feature map using bilinear interpolation

    Note that this layer does not perform convolution on the deformed feature
    map. See get_deform_cnn in cnn.py for usage
    """

    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        """Init

        Parameters
        ----------
        filters : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See Con2d layer in pytorch
        """
        self.filters = filters
        self._grid_param = None
        super(ConvOffset2D, self).__init__(
            self.filters, self.filters * 2, 3, padding=1, bias=False, **kwargs
        )
        self.weight.data.copy_(self._init_weights(self.weight, init_normal_stddev))

    def forward(self, x):
        """Return the deformed featured map"""
        x_shape = x.size()
        offsets = super(ConvOffset2D, self).forward(x)

        # offsets: (b*c, h, w, 2)
        offsets = self._to_bc_h_w_2(offsets, x_shape)

        # x: (b*c, h, w)
        x = self._to_bc_h_w(x, x_shape)

        # X_offset: (b*c, h, w)
        x_offset = th_batch_map_offsets(x, offsets, grid=self._get_grid(self, x))

        # x_offset: (b, h, w, c)
        x_offset = self._to_b_c_h_w(x_offset, x_shape)

        return x_offset

    @staticmethod
    def _get_grid(self, x):
        batch_size, input_height, input_width = x.size(0), x.size(1), x.size(2)
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_param == (batch_size, input_height, input_width, dtype, cuda):
            return self._grid
        self._grid_param = (batch_size, input_height, input_width, dtype, cuda)
        self._grid = th_generate_grid(
            batch_size, input_height, input_width, dtype, cuda
        )
        return self._grid

    @staticmethod
    def _init_weights(weights, std):
        fan_out = weights.size(0)
        fan_in = weights.size(1) * weights.size(2) * weights.size(3)
        w = np.random.normal(0.0, std, (fan_out, fan_in))
        return torch.from_numpy(w.reshape(weights.size()))

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, 2c, h, w) -> (b*c, h, w, 2)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), 2)
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, c, h, w) -> (b*c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))
        return x

    @staticmethod
    def _to_b_c_h_w(x, x_shape):
        """(b*c, h, w) -> (b, c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
        return x


class RBNModule(pl.LightningModule):
    _version = 2
    __constants__ = [
        "track_running_stats",
        "momentum",
        "eps",
        "weight",
        "bias",
        "running_mean",
        "running_var",
        "num_batches_tracked",
    ]

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.9,
        affine=True,
        track_running_stats=True,
    ):
        super(RBNModule, self).__init__()
        self.num_features = num_features
        self.track_running_stats = track_running_stats
        self.eps = eps
        self.affine = affine
        self.momentum = momentum
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(1, num_features, 1, 1))
            self.register_buffer("running_var", torch.ones(1, num_features, 1, 1))
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.uniform_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input, mask_t):
        input_m = input * mask_t
        if self.training:
            mask_mean = torch.mean(mask_t, (0, 2, 3), True)
            x_mean = torch.mean(input_m, (0, 2, 3), True) / mask_mean
            x_var = (
                torch.mean(((input_m - x_mean) * mask_t) ** 2, (0, 2, 3), True)
                / mask_mean
            )

            x_out = (
                self.weight * (input_m - x_mean) / torch.sqrt(x_var + self.eps)
                + self.bias
            )

            self.running_mean.mul_(self.momentum)
            self.running_mean.add_((1 - self.momentum) * x_mean.data)
            self.running_var.mul_(self.momentum)
            self.running_var.add_((1 - self.momentum) * x_var.data)
        else:
            x_out = (
                self.weight
                * (input_m - self.running_mean)
                / torch.sqrt(self.running_var + self.eps)
                + self.bias
            )
        return x_out * mask_t + input * (1 - mask_t)


class RCNModule(pl.LightningModule):
    _version = 2
    __constants__ = [
        "track_running_stats",
        "momentum",
        "eps",
        "weight",
        "bias",
        "running_mean",
        "running_var",
        "num_batches_tracked",
    ]

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.9,
        affine=True,
        track_running_stats=True,
    ):
        super(RCNModule, self).__init__()
        self.num_features = num_features
        self.track_running_stats = track_running_stats
        self.eps = eps
        self.affine = affine
        self.momentum = momentum
        self.mean_weight = nn.Parameter(torch.ones(3))
        self.var_weight = nn.Parameter(torch.ones(3))
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(1, num_features, 1, 1))
            self.register_buffer("running_var", torch.ones(1, num_features, 1, 1))
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.uniform_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input, mask_t):
        input_m = input * mask_t

        if self.training:
            mask_mean_bn = torch.mean(mask_t, (0, 2, 3), True)
            mean_bn = torch.mean(input_m, (0, 2, 3), True) / mask_mean_bn
            var_bn = (
                torch.mean(((input_m - mean_bn) * mask_t) ** 2, (0, 2, 3), True)
                / mask_mean_bn
            )

            self.running_mean.mul_(self.momentum)
            self.running_mean.add_((1 - self.momentum) * mean_bn.data)
            self.running_var.mul_(self.momentum)
            self.running_var.add_((1 - self.momentum) * var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)

        mask_mean_in = torch.mean(mask_t, (2, 3), True)
        mean_in = torch.mean(input_m, (2, 3), True) / mask_mean_in
        var_in = (
            torch.mean(((input_m - mean_in) * mask_t) ** 2, (2, 3), True) / mask_mean_in
        )

        mask_mean_ln = torch.mean(mask_t, (1, 2, 3), True)
        mean_ln = torch.mean(input_m, (1, 2, 3), True) / mask_mean_ln
        var_ln = (
            torch.mean(((input_m - mean_ln) * mask_t) ** 2, (1, 2, 3), True)
            / mask_mean_ln
        )

        mean_weight = F.softmax(self.mean_weight)
        var_weight = F.softmax(self.var_weight)

        x_mean = (
            mean_weight[0] * mean_in
            + mean_weight[1] * mean_ln
            + mean_weight[2] * mean_bn
        )
        x_var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn

        x_out = (
            self.weight * (input_m - x_mean) / torch.sqrt(x_var + self.eps) + self.bias
        )
        return x_out * mask_t + input * (1 - mask_t)


class DSModule(pl.LightningModule):
    def __init__(
        self,
        in_ch,
        out_ch,
        bn=False,
        rn=True,
        sample="none-3",
        activ="relu",
        conv_bias=False,
        defor=True,
    ):
        super().__init__()
        if sample == "down-5":
            self.conv = nn.Conv2d(in_ch + 1, out_ch, 5, 2, 2, bias=conv_bias)
            self.updatemask = nn.MaxPool2d(5, 2, 2)
            if defor:
                self.offset = ConvOffset2D(in_ch + 1)
        elif sample == "down-7":
            self.conv = nn.Conv2d(in_ch + 1, out_ch, 7, 2, 3, bias=conv_bias)
            self.updatemask = nn.MaxPool2d(7, 2, 3)
            if defor:
                self.offset = ConvOffset2D(in_ch + 1)
        elif sample == "down-3":
            self.conv = nn.Conv2d(in_ch + 1, out_ch, 3, 2, 1, bias=conv_bias)
            self.updatemask = nn.MaxPool2d(3, 2, 1)
            if defor:
                self.offset = ConvOffset2D(in_ch + 1)
        else:
            self.conv = nn.Conv2d(in_ch + 2, out_ch, 3, 1, 1, bias=conv_bias)
            self.updatemask = nn.MaxPool2d(3, 1, 1)
            if defor:
                self.offset0 = ConvOffset2D(in_ch - out_ch + 1)
                self.offset1 = ConvOffset2D(out_ch + 1)
        self.in_ch = in_ch
        self.out_ch = out_ch

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if rn:
            # Regional Composite Normalization
            self.rn = RCNModule(out_ch)

            # Regional Batch Normalization
            # self.rn = RBNModule(out_ch)
        if activ == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activ == "leaky":
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, input, input_mask):
        if hasattr(self, "offset"):
            input = torch.cat([input, input_mask[:, :1, :, :]], dim=1)
            h = self.offset(input)
            h = input * input_mask[:, :1, :, :] + (1 - input_mask[:, :1, :, :]) * h
            h = self.conv(h)
            h_mask = self.updatemask(input_mask[:, :1, :, :])
            h = h * h_mask
            h = self.rn(h, h_mask)
        elif hasattr(self, "offset0"):
            h1_in = torch.cat(
                [input[:, self.in_ch - self.out_ch :, :, :], input_mask[:, 1:, :, :]],
                dim=1,
            )
            m1_in = input_mask[:, 1:, :, :]
            h0 = torch.cat(
                [input[:, : self.in_ch - self.out_ch, :, :], input_mask[:, :1, :, :]],
                dim=1,
            )
            h1 = self.offset1(h1_in)
            h1 = m1_in * h1_in + (1 - m1_in) * h1
            h = self.conv(torch.cat([h0, h1], dim=1))
            h = self.rn(h, input_mask[:, :1, :, :])
            h_mask = F.interpolate(
                input_mask[:, :1, :, :], scale_factor=2, mode="nearest"
            )
        else:
            h = self.conv(torch.cat([input, input_mask[:, :, :, :]], dim=1))
            h_mask = self.updatemask(input_mask[:, :1, :, :])
            h = h * h_mask

        if hasattr(self, "bn"):
            h = self.bn(h)
        if hasattr(self, "activation"):
            h = self.activation(h)
        return h, h_mask


class DSNet(pl.LightningModule):
    def __init__(self, layer_size=8, input_channels=3, upsampling_mode="nearest"):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = DSModule(
            input_channels, 64, rn=False, sample="down-7", defor=False
        )
        self.enc_2 = DSModule(64, 128, sample="down-5")
        self.enc_3 = DSModule(128, 256, sample="down-5")
        self.enc_4 = DSModule(256, 512, sample="down-3")
        for i in range(4, self.layer_size):
            name = "enc_{:d}".format(i + 1)
            setattr(self, name, DSModule(512, 512, sample="down-3"))

        for i in range(4, self.layer_size):
            name = "dec_{:d}".format(i + 1)
            setattr(self, name, DSModule(512 + 512, 512, activ="leaky"))
        self.dec_4 = DSModule(512 + 256, 256, activ="leaky")
        self.dec_3 = DSModule(256 + 128, 128, activ="leaky")
        self.dec_2 = DSModule(128 + 64, 64, activ="leaky")
        self.dec_1 = DSModule(
            64 + input_channels, input_channels, rn=False, activ=None, defor=False
        )
        self.att = PixelContextualAttention(128)

    def forward(self, input, input_mask):
        input = input.type(torch.cuda.FloatTensor)
        input_mask = input_mask.type(torch.cuda.FloatTensor)

        input_mask = input_mask[:, 0:1, :, :]
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N

        h_dict["h_0"], h_mask_dict["h_0"] = input, input_mask

        h_key_prev = "h_0"
        for i in range(1, self.layer_size + 1):
            l_key = "enc_{:d}".format(i)
            h_key = "h_{:d}".format(i)
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
                h_dict[h_key_prev], h_mask_dict[h_key_prev]
            )
            h_key_prev = h_key

        h_key = "h_{:d}".format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]
        h_mask = F.interpolate(h_mask, scale_factor=2, mode="nearest")

        for i in range(self.layer_size, 0, -1):
            enc_h_key = "h_{:d}".format(i - 1)
            dec_l_key = "dec_{:d}".format(i)

            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)

            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)
            if i == 3:
                h = self.att(h, input_mask[:, :1, :, :])
        # return h, h_mask
        return h


# @title [Deoldify](https://github.com/alfagao/DeOldify) (2018) (dataloader not implemented)

"""
torch_imports.py (9-3-20)
https://github.com/alfagao/DeOldify/blob/bc9d4562bf2014f5268f5c616ae31873577d9fde/fastai/torch_imports.py

conv_learner.py (9-3-20)
https://github.com/alfagao/DeOldify/blob/bc9d4562bf2014f5268f5c616ae31873577d9fde/fastai/conv_learner.py

model.py (9-3-20)
https://github.com/alfagao/DeOldify/blob/bc9d4562bf2014f5268f5c616ae31873577d9fde/fastai/model.py

modules.py (9-3-20)
https://github.com/alfagao/DeOldify/blob/bc9d4562bf2014f5268f5c616ae31873577d9fde/fasterai/modules.py

generators.py (9-3-20)
https://github.com/alfagao/DeOldify/blob/bc9d4562bf2014f5268f5c616ae31873577d9fde/fasterai/generators.py
"""

from abc import ABC, abstractmethod
from torch.nn.utils.spectral_norm import spectral_norm
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
import pytorch_lightning as pl
from torchvision.models import vgg16_bn, vgg19_bn


def vgg16(pre):
    return children(vgg16_bn(pre))[0]


def vgg19(pre):
    return children(vgg19_bn(pre))[0]


def cut_model(m, cut):
    return list(m.children())[:cut] if cut else [m]


"""
model_meta = {
    resnet18:[8,6], resnet34:[8,6], resnet50:[8,6], resnet101:[8,6], resnet152:[8,6],
    vgg16:[0,22], vgg19:[0,22],
    resnext50:[8,6], resnext101:[8,6], resnext101_64:[8,6],
    wrn:[8,6], inceptionresnet_2:[-2,9], inception_4:[-1,9],
    dn121:[0,7], dn161:[0,7], dn169:[0,7], dn201:[0,7],
}
"""

model_meta = {
    resnet18: [8, 6],
    resnet34: [8, 6],
    resnet50: [8, 6],
    resnet101: [8, 6],
    resnet152: [8, 6],
    vgg16: [0, 22],
    vgg19: [0, 22],
}


class ConvBlock(pl.LightningModule):
    def __init__(
        self,
        ni: int,
        no: int,
        ks: int = 3,
        stride: int = 1,
        pad: int = None,
        actn: bool = True,
        bn: bool = True,
        bias: bool = True,
        sn: bool = False,
        leakyReLu: bool = False,
        self_attention: bool = False,
        inplace_relu: bool = True,
    ):
        super().__init__()
        if pad is None:
            pad = ks // 2 // stride

        if sn:
            layers = [
                spectral_norm(nn.Conv2d(ni, no, ks, stride, padding=pad, bias=bias))
            ]
        else:
            layers = [nn.Conv2d(ni, no, ks, stride, padding=pad, bias=bias)]
        if actn:
            (
                layers.append(nn.LeakyReLU(0.2, inplace=inplace_relu))
                if leakyReLu
                else layers.append(nn.ReLU(inplace=inplace_relu))
            )
        if bn:
            layers.append(nn.BatchNorm2d(no))
        if self_attention:
            layers.append(SelfAttention(no, 1))

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class UpSampleBlock(pl.LightningModule):
    @staticmethod
    def _conv(
        ni: int,
        nf: int,
        ks: int = 3,
        bn: bool = True,
        sn: bool = False,
        leakyReLu: bool = False,
    ):
        layers = [
            ConvBlock(ni, nf, ks=ks, sn=sn, bn=bn, actn=False, leakyReLu=leakyReLu)
        ]
        return nn.Sequential(*layers)

    @staticmethod
    def _icnr(x: torch.Tensor, scale: int = 2):
        init = nn.init.kaiming_normal_
        new_shape = [int(x.shape[0] / (scale**2))] + list(x.shape[1:])
        subkernel = torch.zeros(new_shape)
        subkernel = init(subkernel)
        subkernel = subkernel.transpose(0, 1)
        subkernel = subkernel.contiguous().view(
            subkernel.shape[0], subkernel.shape[1], -1
        )
        kernel = subkernel.repeat(1, 1, scale**2)
        transposed_shape = [x.shape[1]] + [x.shape[0]] + list(x.shape[2:])
        kernel = kernel.contiguous().view(transposed_shape)
        kernel = kernel.transpose(0, 1)
        return kernel

    def __init__(
        self,
        ni: int,
        nf: int,
        scale: int = 2,
        ks: int = 3,
        bn: bool = True,
        sn: bool = False,
        leakyReLu: bool = False,
    ):
        super().__init__()
        layers = []
        assert (math.log(scale, 2)).is_integer()

        for i in range(int(math.log(scale, 2))):
            layers += [
                UpSampleBlock._conv(
                    ni, nf * 4, ks=ks, bn=bn, sn=sn, leakyReLu=leakyReLu
                ),
                nn.PixelShuffle(2),
            ]
            if bn:
                layers += [nn.BatchNorm2d(nf)]

            ni = nf

        self.sequence = nn.Sequential(*layers)
        self._icnr_init()

    def _icnr_init(self):
        conv_shuffle = self.sequence[0][0].seq[0]
        kernel = UpSampleBlock._icnr(conv_shuffle.weight)
        conv_shuffle.weight.data.copy_(kernel)

    def forward(self, x):
        return self.sequence(x)


class UnetBlock(pl.LightningModule):
    def __init__(
        self,
        up_in: int,
        x_in: int,
        n_out: int,
        bn: bool = True,
        sn: bool = False,
        leakyReLu: bool = False,
        self_attention: bool = False,
        inplace_relu: bool = True,
    ):
        super().__init__()
        up_out = x_out = n_out // 2
        self.x_conv = ConvBlock(
            x_in, x_out, ks=1, bn=False, actn=False, sn=sn, inplace_relu=inplace_relu
        )
        self.tr_conv = UpSampleBlock(
            up_in, up_out, 2, bn=bn, sn=sn, leakyReLu=leakyReLu
        )
        self.relu = (
            nn.LeakyReLU(0.2, inplace=inplace_relu)
            if leakyReLu
            else nn.ReLU(inplace=inplace_relu)
        )
        out_layers = []
        if bn:
            out_layers.append(nn.BatchNorm2d(n_out))
        if self_attention:
            out_layers.append(SelfAttention(n_out))
        self.out = nn.Sequential(*out_layers)

    def forward(self, up_p: int, x_p: int):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        x = torch.cat([up_p, x_p], dim=1)
        x = self.relu(x)
        return self.out(x)


class SaveFeatures:
    features = None

    def __init__(self, m: pl.LightningModule):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def remove(self):
        self.hook.remove()


class SelfAttention(pl.LightningModule):
    def __init__(self, in_channel: int, gain: int = 1):
        super().__init__()
        self.query = self._spectral_init(
            nn.Conv1d(in_channel, in_channel // 8, 1), gain=gain
        )
        self.key = self._spectral_init(
            nn.Conv1d(in_channel, in_channel // 8, 1), gain=gain
        )
        self.value = self._spectral_init(
            nn.Conv1d(in_channel, in_channel, 1), gain=gain
        )
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def _spectral_init(self, module: pl.LightningModule, gain: int = 1):
        nn.init.kaiming_uniform_(module.weight, gain)
        if module.bias is not None:
            module.bias.data.zero_()

        return spectral_norm(module)

    def forward(self, input: torch.Tensor):
        shape = input.shape
        flatten = input.view(shape[0], shape[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)
        out = self.gamma * attn + input
        return out


class GeneratorModule(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    def set_trainable(self, trainable: bool):
        set_trainable(self, trainable)

    @abstractmethod
    def get_layer_groups(self, precompute: bool = False) -> []:
        pass

    @abstractmethod
    def forward(self, x_in: torch.Tensor, max_render_sz: int = 400):
        pass

    def freeze_to(self, n: int):
        c = self.get_layer_groups()
        for l in c:
            set_trainable(l, False)
        for l in c[n:]:
            set_trainable(l, True)

    def get_device(self):
        return next(self.parameters()).device


class AbstractUnet(GeneratorModule):
    def __init__(self, nf_factor: int = 1, scale: int = 1):
        super().__init__()
        assert (math.log(scale, 2)).is_integer()
        self.rn, self.lr_cut = self._get_pretrained_resnet_base()
        ups = self._get_decoding_layers(nf_factor=nf_factor, scale=scale)
        self.relu = nn.ReLU()
        self.up1 = ups[0]
        self.up2 = ups[1]
        self.up3 = ups[2]
        self.up4 = ups[3]
        self.up5 = ups[4]
        self.out = nn.Sequential(
            ConvBlock(32 * nf_factor, 3, ks=3, actn=False, bn=False, sn=True), nn.Tanh()
        )

    @abstractmethod
    def _get_pretrained_resnet_base(self, layers_cut: int = 0):
        pass

    @abstractmethod
    def _get_decoding_layers(self, nf_factor: int, scale: int):
        pass

    # Gets around irritating inconsistent halving coming from resnet
    def _pad(
        self, x: torch.Tensor, target: torch.Tensor, total_padh: int, total_padw: int
    ) -> torch.Tensor:
        h = x.shape[2]
        w = x.shape[3]

        target_h = target.shape[2] * 2
        target_w = target.shape[3] * 2

        if h < target_h or w < target_w:
            padh = target_h - h if target_h > h else 0
            total_padh = total_padh + padh
            padw = target_w - w if target_w > w else 0
            total_padw = total_padw + padw
            return (F.pad(x, (0, padw, 0, padh), "reflect", 0), total_padh, total_padw)

        return (x, total_padh, total_padw)

    def _remove_padding(self, x: torch.Tensor, padh: int, padw: int) -> torch.Tensor:
        if padw == 0 and padh == 0:
            return x

        target_h = x.shape[2] - padh
        target_w = x.shape[3] - padw
        return x[:, :, :target_h, :target_w]

    def _encode(self, x: torch.Tensor):
        x = self.rn[0](x)
        x = self.rn[1](x)
        x = self.rn[2](x)
        enc0 = x
        x = self.rn[3](x)
        x = self.rn[4](x)
        enc1 = x
        x = self.rn[5](x)
        enc2 = x
        x = self.rn[6](x)
        enc3 = x
        x = self.rn[7](x)
        return (x, enc0, enc1, enc2, enc3)

    def _decode(
        self,
        x: torch.Tensor,
        enc0: torch.Tensor,
        enc1: torch.Tensor,
        enc2: torch.Tensor,
        enc3: torch.Tensor,
    ):
        padh = 0
        padw = 0
        x = self.relu(x)
        enc3, padh, padw = self._pad(enc3, x, padh, padw)
        x = self.up1(x, enc3)
        enc2, padh, padw = self._pad(enc2, x, padh, padw)
        x = self.up2(x, enc2)
        enc1, padh, padw = self._pad(enc1, x, padh, padw)
        x = self.up3(x, enc1)
        enc0, padh, padw = self._pad(enc0, x, padh, padw)
        x = self.up4(x, enc0)
        # This is a bit too much padding being removed, but I
        # haven't yet figured out a good way to determine what
        # exactly should be removed.  This is consistently more
        # than enough though.
        x = self.up5(x)
        x = self.out(x)
        x = self._remove_padding(x, padh, padw)
        return x

    def forward(self, x: torch.Tensor):
        x, enc0, enc1, enc2, enc3 = self._encode(x)
        x = self._decode(x, enc0, enc1, enc2, enc3)
        return x

    def get_layer_groups(self, precompute: bool = False) -> []:
        lgs = list(split_by_idxs(children(self.rn), [self.lr_cut]))
        return lgs + [children(self)[1:]]

    def close(self):
        for sf in self.sfs:
            sf.remove()


class Unet34(AbstractUnet):
    def __init__(self, nf_factor: int = 1, scale: int = 1):
        super().__init__(nf_factor=nf_factor, scale=scale)

    def _get_pretrained_resnet_base(self, layers_cut: int = 0):
        f = resnet34
        cut, lr_cut = model_meta[f]
        cut -= layers_cut
        layers = cut_model(f(True), cut)
        return nn.Sequential(*layers), lr_cut

    def _get_decoding_layers(self, nf_factor: int, scale: int):
        self_attention = True
        bn = True
        sn = True
        leakyReLu = False
        layers = []
        layers.append(
            UnetBlock(512, 256, 512 * nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn)
        )
        layers.append(
            UnetBlock(
                512 * nf_factor, 128, 512 * nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn
            )
        )
        layers.append(
            UnetBlock(
                512 * nf_factor,
                64,
                512 * nf_factor,
                sn=sn,
                self_attention=self_attention,
                leakyReLu=leakyReLu,
                bn=bn,
            )
        )
        layers.append(
            UnetBlock(
                512 * nf_factor, 64, 256 * nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn
            )
        )
        layers.append(
            UpSampleBlock(
                256 * nf_factor,
                32 * nf_factor,
                2 * scale,
                sn=sn,
                leakyReLu=leakyReLu,
                bn=bn,
            )
        )
        return layers


class Unet101(AbstractUnet):
    def __init__(self, nf_factor: int = 1, scale: int = 1):
        super().__init__(nf_factor=nf_factor, scale=scale)

    def _get_pretrained_resnet_base(self, layers_cut: int = 0):
        f = resnet101
        cut, lr_cut = model_meta[f]
        cut -= layers_cut
        layers = cut_model(f(True), cut)
        return nn.Sequential(*layers), lr_cut

    def _get_decoding_layers(self, nf_factor: int, scale: int):
        self_attention = True
        bn = True
        sn = True
        leakyReLu = False
        layers = []
        layers.append(
            UnetBlock(2048, 1024, 512 * nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn)
        )
        layers.append(
            UnetBlock(
                512 * nf_factor, 512, 512 * nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn
            )
        )
        layers.append(
            UnetBlock(
                512 * nf_factor,
                256,
                512 * nf_factor,
                sn=sn,
                self_attention=self_attention,
                leakyReLu=leakyReLu,
                bn=bn,
            )
        )
        layers.append(
            UnetBlock(
                512 * nf_factor, 64, 256 * nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn
            )
        )
        layers.append(
            UpSampleBlock(
                256 * nf_factor,
                32 * nf_factor,
                2 * scale,
                sn=sn,
                leakyReLu=leakyReLu,
                bn=bn,
            )
        )
        return layers


class Unet152(AbstractUnet):
    def __init__(self, nf_factor: int = 1, scale: int = 1):
        super().__init__(nf_factor=nf_factor, scale=scale)

    def _get_pretrained_resnet_base(self, layers_cut: int = 0):
        f = resnet152
        cut, lr_cut = model_meta[f]
        cut -= layers_cut
        layers = cut_model(f(True), cut)
        return nn.Sequential(*layers), lr_cut

    def _get_decoding_layers(self, nf_factor: int, scale: int):
        self_attention = True
        bn = True
        sn = True
        leakyReLu = False
        layers = []
        layers.append(
            UnetBlock(2048, 1024, 512 * nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn)
        )
        layers.append(
            UnetBlock(
                512 * nf_factor, 512, 512 * nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn
            )
        )
        layers.append(
            UnetBlock(
                512 * nf_factor,
                256,
                512 * nf_factor,
                sn=sn,
                self_attention=self_attention,
                leakyReLu=leakyReLu,
                bn=bn,
            )
        )
        layers.append(
            UnetBlock(
                512 * nf_factor, 64, 256 * nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn
            )
        )
        layers.append(
            UpSampleBlock(
                256 * nf_factor,
                32 * nf_factor,
                2 * scale,
                sn=sn,
                leakyReLu=leakyReLu,
                bn=bn,
            )
        )
        return layers


class DSNetDeoldify(pl.LightningModule):
    def __init__(self, layer_size=8, input_channels=3, upsampling_mode="nearest"):
        super(DSNetDeoldify, self).__init__()
        self.netG1 = DSNet(
            layer_size=layer_size,
            input_channels=input_channels,
            upsampling_mode=upsampling_mode,
        )
        self.netG2 = Unet34()

    def forward(self, input, input_mask):
        result1 = self.netG1(input, input_mask)

        result1 = input * input_mask + result1 * (1 - input_mask)

        # concat = torch.cat((result1, input_mask), dim=1)
        # result2 = self.netG2(concat)

        result2 = self.netG2(result1)
        return result2
