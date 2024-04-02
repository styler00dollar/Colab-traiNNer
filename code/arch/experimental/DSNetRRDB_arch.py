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


"""
block.py (9-3-20)
https://github.com/victorca25/BasicSR/blob/master/codes/models/modules/architectures/block.py
"""


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


"""
RRDBNet_arch.py (12-2-20)
https://github.com/victorca25/BasicSR/blob/master/codes/models/modules/architectures/RRDBNet_arch.py
"""
import math
import torch.nn as nn

# import torchvision
# from . import block as B
import functools

# from . import spectral_norm as SN


####################
# RRDBNet Generator (original architecture)
####################


class RRDBNet(pl.LightningModule):
    def __init__(
        self,
        in_nc,
        out_nc,
        nf,
        nb,
        nr=3,
        gc=32,
        upscale=4,
        norm_type=None,
        act_type="leakyrelu",
        mode="CNA",
        upsample_mode="upconv",
        convtype="Conv2D",
        finalact=None,
        gaussian_noise=False,
        plus=False,
    ):
        super(RRDBNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = conv_block(
            in_nc, nf, kernel_size=3, norm_type=None, act_type=None, convtype=convtype
        )
        rb_blocks = [
            RRDB(
                nf,
                nr,
                kernel_size=3,
                gc=32,
                stride=1,
                bias=1,
                pad_type="zero",
                norm_type=norm_type,
                act_type=act_type,
                mode="CNA",
                convtype=convtype,
                gaussian_noise=gaussian_noise,
                plus=plus,
            )
            for _ in range(nb)
        ]
        LR_conv = conv_block(
            nf,
            nf,
            kernel_size=3,
            norm_type=norm_type,
            act_type=None,
            mode=mode,
            convtype=convtype,
        )

        if upsample_mode == "upconv":
            upsample_block = upconv_block
        elif upsample_mode == "pixelshuffle":
            upsample_block = pixelshuffle_block
        else:
            raise NotImplementedError(
                "upsample mode [{:s}] is not found".format(upsample_mode)
            )
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type, convtype=convtype)
        else:
            upsampler = [
                upsample_block(nf, nf, act_type=act_type, convtype=convtype)
                for _ in range(n_upscale)
            ]
        HR_conv0 = conv_block(
            nf, nf, kernel_size=3, norm_type=None, act_type=act_type, convtype=convtype
        )
        HR_conv1 = conv_block(
            nf, out_nc, kernel_size=3, norm_type=None, act_type=None, convtype=convtype
        )

        # Note: this option adds new parameters to the architecture, another option is to use "outm" in the forward
        outact = act(finalact) if finalact else None

        self.model = sequential(
            fea_conv,
            ShortcutBlock(sequential(*rb_blocks, LR_conv)),
            *upsampler,
            HR_conv0,
            HR_conv1,
            outact
        )

    def forward(self, x, outm=None):
        x = self.model(x)

        if (
            outm == "scaltanh"
        ):  # limit output range to [-1,1] range with tanh and rescale to [0,1] Idea from: https://github.com/goldhuang/SRGAN-PyTorch/blob/master/model.py
            return (torch.tanh(x) + 1.0) / 2.0
        elif outm == "tanh":  # limit output to [-1,1] range
            return torch.tanh(x)
        elif outm == "sigmoid":  # limit output to [0,1] range
            return torch.sigmoid(x)
        elif outm == "clamp":
            return torch.clamp(x, min=0.0, max=1.0)
        else:  # Default, no cap for the output
            return x


class RRDB(pl.LightningModule):
    """
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    """

    def __init__(
        self,
        nf,
        nr=3,
        kernel_size=3,
        gc=32,
        stride=1,
        bias=1,
        pad_type="zero",
        norm_type=None,
        act_type="leakyrelu",
        mode="CNA",
        convtype="Conv2D",
        spectral_norm=False,
        gaussian_noise=False,
        plus=False,
    ):
        super(RRDB, self).__init__()
        # This is for backwards compatibility with existing models
        if nr == 3:
            self.RDB1 = ResidualDenseBlock_5C(
                nf,
                kernel_size,
                gc,
                stride,
                bias,
                pad_type,
                norm_type,
                act_type,
                mode,
                convtype,
                spectral_norm=spectral_norm,
                gaussian_noise=gaussian_noise,
                plus=plus,
            )
            self.RDB2 = ResidualDenseBlock_5C(
                nf,
                kernel_size,
                gc,
                stride,
                bias,
                pad_type,
                norm_type,
                act_type,
                mode,
                convtype,
                spectral_norm=spectral_norm,
                gaussian_noise=gaussian_noise,
                plus=plus,
            )
            self.RDB3 = ResidualDenseBlock_5C(
                nf,
                kernel_size,
                gc,
                stride,
                bias,
                pad_type,
                norm_type,
                act_type,
                mode,
                convtype,
                spectral_norm=spectral_norm,
                gaussian_noise=gaussian_noise,
                plus=plus,
            )
        else:
            RDB_list = [
                ResidualDenseBlock_5C(
                    nf,
                    kernel_size,
                    gc,
                    stride,
                    bias,
                    pad_type,
                    norm_type,
                    act_type,
                    mode,
                    convtype,
                    spectral_norm=spectral_norm,
                    gaussian_noise=gaussian_noise,
                    plus=plus,
                )
                for _ in range(nr)
            ]
            self.RDBs = nn.Sequential(*RDB_list)

    def forward(self, x):
        if hasattr(self, "RDB1"):
            out = self.RDB1(x)
            out = self.RDB2(out)
            out = self.RDB3(out)
        else:
            out = self.RDBs(x)
        return out * 0.2 + x


class ResidualDenseBlock_5C(pl.LightningModule):
    """
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    Modified options that can be used:
        - "Partial Convolution based Padding" arXiv:1811.11718
        - "Spectral normalization" arXiv:1802.05957
        - "ICASSP 2020 - ESRGAN+ : Further Improving ESRGAN" N. C.
            {Rakotonirina} and A. {Rasoanaivo}

    Args:
        nf (int): Channel number of intermediate features (num_feat).
        gc (int): Channels for each growth (num_grow_ch: growth channel,
            i.e. intermediate channels).
        convtype (str): the type of convolution to use. Default: 'Conv2D'
        gaussian_noise (bool): enable the ESRGAN+ gaussian noise (no new
            trainable parameters)
        plus (bool): enable the additional residual paths from ESRGAN+
            (adds trainable parameters)
    """

    def __init__(
        self,
        nf=64,
        kernel_size=3,
        gc=32,
        stride=1,
        bias=1,
        pad_type="zero",
        norm_type=None,
        act_type="leakyrelu",
        mode="CNA",
        convtype="Conv2D",
        spectral_norm=False,
        gaussian_noise=False,
        plus=False,
    ):
        super(ResidualDenseBlock_5C, self).__init__()

        ## +
        self.noise = GaussianNoise() if gaussian_noise else None
        self.conv1x1 = conv1x1(nf, gc) if plus else None
        ## +

        self.conv1 = conv_block(
            nf,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            convtype=convtype,
            spectral_norm=spectral_norm,
        )
        self.conv2 = conv_block(
            nf + gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            convtype=convtype,
            spectral_norm=spectral_norm,
        )
        self.conv3 = conv_block(
            nf + 2 * gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            convtype=convtype,
            spectral_norm=spectral_norm,
        )
        self.conv4 = conv_block(
            nf + 3 * gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            convtype=convtype,
            spectral_norm=spectral_norm,
        )
        if mode == "CNA":
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(
            nf + 4 * gc,
            nf,
            3,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=last_act,
            mode=mode,
            convtype=convtype,
            spectral_norm=spectral_norm,
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        if self.conv1x1:
            x2 = x2 + self.conv1x1(x)  # +
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        if self.conv1x1:
            x4 = x4 + x2  # +
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        if self.noise:
            return self.noise(x5.mul(0.2) + x)
        else:
            return x5 * 0.2 + x


####################
# RRDBNet Generator (modified/"new" architecture)
####################


class MRRDBNet(pl.LightningModule):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(MRRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDBM, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(
            self.upconv1(
                torch.nn.functional.interpolate(fea, scale_factor=2, mode="nearest")
            )
        )
        fea = self.lrelu(
            self.upconv2(
                torch.nn.functional.interpolate(fea, scale_factor=2, mode="nearest")
            )
        )
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out


class ResidualDenseBlock_5CM(pl.LightningModule):
    """
    Residual Dense Block
    """

    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5CM, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], scale=0.1
        )

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDBM(pl.LightningModule):
    """Residual in Residual Dense Block"""

    def __init__(self, nf, gc=32):
        super(RRDBM, self).__init__()
        self.RDB1 = ResidualDenseBlock_5CM(nf, gc)
        self.RDB2 = ResidualDenseBlock_5CM(nf, gc)
        self.RDB3 = ResidualDenseBlock_5CM(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class DSNetRRDB(pl.LightningModule):
    def __init__(
        self,
        layer_size=8,
        input_channels=3,
        upsampling_mode="nearest",
        in_nc=4,
        out_nc=3,
        nf=128,
        nb=8,
        gc=32,
        upscale=1,
        norm_type=None,
        act_type="leakyrelu",
        mode="CNA",
        upsample_mode="upconv",
        convtype="Conv2D",
        finalact=None,
        gaussian_noise=True,
        plus=False,
        nr=3,
    ):
        super(DSNetRRDB, self).__init__()
        self.netG1 = DSNet(
            layer_size=layer_size,
            input_channels=input_channels,
            upsampling_mode=upsampling_mode,
        )
        self.netG2 = RRDBNet(
            in_nc=in_nc,
            out_nc=out_nc,
            nf=nf,
            nb=nb,
            gc=gc,
            upscale=upscale,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            upsample_mode=upsample_mode,
            convtype=convtype,
            finalact=finalact,
            gaussian_noise=gaussian_noise,
            plus=plus,
            nr=nr,
        )

    def forward(self, input, input_mask):
        result1 = self.netG1(input, input_mask)

        result1 = input * input_mask + result1 * (1 - input_mask)

        concat = torch.cat((result1, input_mask), dim=1)
        result2 = self.netG2(concat)
        return result2
