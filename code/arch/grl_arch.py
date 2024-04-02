# https://github.com/ofsoundof/GRL-Image-Restoration/blob/main/models/networks/grl.py
# https://github.com/ofsoundof/GRL-Image-Restoration/blob/3123fcadfee2107d813c5e8fff58227f14c989bc/models/common/upsample.py
# https://github.com/ofsoundof/GRL-Image-Restoration/blob/3123fcadfee2107d813c5e8fff58227f14c989bc/models/common/mixed_attn_block_efficient.py
# https://github.com/ofsoundof/GRL-Image-Restoration/blob/3123fcadfee2107d813c5e8fff58227f14c989bc/models/common/ops.py
# https://github.com/ofsoundof/GRL-Image-Restoration/blob/3123fcadfee2107d813c5e8fff58227f14c989bc/models/common/swin_v1_block.py#L469
# https://github.com/ofsoundof/GRL-Image-Restoration/blob/3123fcadfee2107d813c5e8fff58227f14c989bc/models/common/mixed_attn_block.py#L739

"""
Efficient and Explicit Modelling of Image Hierarchies for Image Restoration
Image restoration transformers with global, regional, and local modelling
A clean version of the.
Shared buffers are used for relative_coords_table, relative_position_index, and attn_mask.
"""
from abc import ABC
from math import prod
from omegaconf import OmegaConf
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from typing import Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CPB_MLP(nn.Sequential):
    def __init__(self, in_channels, out_channels, channels=512):
        m = [
            nn.Linear(in_channels, channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels, out_channels, bias=False),
        ]
        super(CPB_MLP, self).__init__(*m)


class AffineTransformWindow(nn.Module):
    r"""Affine transformation of the attention map.
    The window is a square window.
    Supports attention between different window sizes
    """

    def __init__(
        self,
        num_heads,
        input_resolution,
        window_size,
        pretrained_window_size=[0, 0],
        shift_size=0,
        anchor_window_down_factor=1,
        args=None,
    ):
        super(AffineTransformWindow, self).__init__()
        # print("AffineTransformWindow", args)
        self.num_heads = num_heads
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.shift_size = shift_size
        self.anchor_window_down_factor = anchor_window_down_factor
        self.use_buffer = args.use_buffer

        logit_scale = torch.log(10 * torch.ones((num_heads, 1, 1)))
        self.logit_scale = nn.Parameter(logit_scale, requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = CPB_MLP(2, num_heads)
        if self.use_buffer:
            table = get_relative_coords_table_all(
                window_size, pretrained_window_size, anchor_window_down_factor
            )
            index = get_relative_position_index_simple(
                window_size, anchor_window_down_factor
            )
            self.register_buffer("relative_coords_table", table)
            self.register_buffer("relative_position_index", index)

            if self.shift_size > 0:
                attn_mask = calculate_mask(
                    input_resolution, self.window_size, self.shift_size
                )
            else:
                attn_mask = None
            self.register_buffer("attn_mask", attn_mask)

    def forward(self, attn, x_size):
        B_, H, N, _ = attn.shape
        device = attn.device
        # logit scale
        attn = attn * torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()

        # relative position bias
        if self.use_buffer:
            table = self.relative_coords_table
            index = self.relative_position_index
        else:
            table = get_relative_coords_table_all(
                self.window_size,
                self.pretrained_window_size,
                self.anchor_window_down_factor,
            ).to(device)
            index = get_relative_position_index_simple(
                self.window_size, self.anchor_window_down_factor
            ).to(device)

        bias_table = self.cpb_mlp(table)  # 2*Wh-1, 2*Ww-1, num_heads
        bias_table = bias_table.view(-1, self.num_heads)

        win_dim = prod(self.window_size)
        bias = bias_table[index.view(-1)]
        bias = bias.view(win_dim, win_dim, -1).permute(2, 0, 1).contiguous()
        # nH, Wh*Ww, Wh*Ww
        bias = 16 * torch.sigmoid(bias)
        attn = attn + bias.unsqueeze(0)

        # W-MSA/SW-MSA
        if self.use_buffer:
            mask = self.attn_mask
            # during test and window shift, recalculate the mask
            if self.input_resolution != x_size and self.shift_size > 0:
                mask = calculate_mask(x_size, self.window_size, self.shift_size)
                mask = mask.to(attn.device)
        else:
            if self.shift_size > 0:
                mask = calculate_mask(x_size, self.window_size, self.shift_size)
                mask = mask.to(attn.device)
            else:
                mask = None

        # shift attention mask
        if mask is not None:
            nW = mask.shape[0]
            mask = mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask
            attn = attn.view(-1, self.num_heads, N, N)

        return attn


class AffineTransformStripe(nn.Module):
    r"""Affine transformation of the attention map.
    The window is a stripe window. Supports attention between different window sizes
    """

    def __init__(
        self,
        num_heads,
        input_resolution,
        stripe_size,
        stripe_groups,
        stripe_shift,
        pretrained_stripe_size=[0, 0],
        anchor_window_down_factor=1,
        window_to_anchor=True,
        args=None,
    ):
        super(AffineTransformStripe, self).__init__()
        self.num_heads = num_heads
        self.input_resolution = input_resolution
        self.stripe_size = stripe_size
        self.stripe_groups = stripe_groups
        self.pretrained_stripe_size = pretrained_stripe_size
        # TODO: be careful when determining the pretrained_stripe_size
        self.stripe_shift = stripe_shift
        stripe_size, shift_size = self._get_stripe_info(input_resolution)
        self.anchor_window_down_factor = anchor_window_down_factor
        self.window_to_anchor = window_to_anchor
        self.use_buffer = args.use_buffer

        logit_scale = torch.log(10 * torch.ones((num_heads, 1, 1)))
        self.logit_scale = nn.Parameter(logit_scale, requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = CPB_MLP(2, num_heads)
        if self.use_buffer:
            table = get_relative_coords_table_all(
                stripe_size, pretrained_stripe_size, anchor_window_down_factor
            )
            index = get_relative_position_index_simple(
                stripe_size, anchor_window_down_factor, window_to_anchor
            )
            self.register_buffer("relative_coords_table", table)
            self.register_buffer("relative_position_index", index)

            if self.stripe_shift:
                attn_mask = calculate_mask_all(
                    input_resolution,
                    stripe_size,
                    shift_size,
                    anchor_window_down_factor,
                    window_to_anchor,
                )
            else:
                attn_mask = None
            self.register_buffer("attn_mask", attn_mask)

    def forward(self, attn, x_size):
        B_, H, N1, N2 = attn.shape
        device = attn.device
        # logit scale
        attn = attn * torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()

        # relative position bias
        stripe_size, shift_size = self._get_stripe_info(x_size)
        fixed_stripe_size = (
            self.stripe_groups[0] is None and self.stripe_groups[1] is None
        )
        if not self.use_buffer or (
            self.use_buffer
            and self.input_resolution != x_size
            and not fixed_stripe_size
        ):
            # during test and stripe size is not fixed.
            pretrained_stripe_size = (
                self.pretrained_stripe_size
            )  # or stripe_size; Needs further pondering
            table = get_relative_coords_table_all(
                stripe_size, pretrained_stripe_size, self.anchor_window_down_factor
            )
            table = table.to(device)
            index = get_relative_position_index_simple(
                stripe_size, self.anchor_window_down_factor, self.window_to_anchor
            ).to(device)
        else:
            table = self.relative_coords_table
            index = self.relative_position_index
        # The same table size-> 1, Wh+AWh-1, Ww+AWw-1, 2
        # But different index size -> # Wh*Ww, AWh*AWw
        # if N1 < N2:
        #     index = index.transpose(0, 1)

        bias_table = self.cpb_mlp(table).view(-1, self.num_heads)
        # if not self.training:
        #     print(bias_table.shape, index.max(), index.min())
        bias = bias_table[index.view(-1)]
        bias = bias.view(N1, N2, -1).permute(2, 0, 1).contiguous()
        # nH, Wh*Ww, Wh*Ww
        bias = 16 * torch.sigmoid(bias)
        # print(N1, N2, attn.shape, bias.unsqueeze(0).shape)
        attn = attn + bias.unsqueeze(0)

        # W-MSA/SW-MSA
        if self.use_buffer:
            mask = self.attn_mask
            # during test and window shift, recalculate the mask
            if self.input_resolution != x_size and self.stripe_shift > 0:
                mask = calculate_mask_all(
                    x_size,
                    stripe_size,
                    shift_size,
                    self.anchor_window_down_factor,
                    self.window_to_anchor,
                )
                mask = mask.to(device)
        else:
            if self.stripe_shift > 0:
                mask = calculate_mask_all(
                    x_size,
                    stripe_size,
                    shift_size,
                    self.anchor_window_down_factor,
                    self.window_to_anchor,
                )
                mask = mask.to(attn.device)
            else:
                mask = None

        # shift attention mask
        if mask is not None:
            nW = mask.shape[0]
            mask = mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(B_ // nW, nW, self.num_heads, N1, N2) + mask
            attn = attn.view(-1, self.num_heads, N1, N2)

        return attn

    def _get_stripe_info(self, input_resolution):
        stripe_size, shift_size = [], []
        for s, g, d in zip(self.stripe_size, self.stripe_groups, input_resolution):
            if g is None:
                stripe_size.append(s)
                shift_size.append(s // 2 if self.stripe_shift else 0)
            else:
                stripe_size.append(d // g)
                shift_size.append(0 if g == 1 else d // (g * 2))
        return stripe_size, shift_size


class Attention(ABC, nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def attn(self, q, k, v, attn_transform, x_size, reshape=True):
        # cosine attention map
        B_, _, H, head_dim = q.shape
        if self.euclidean_dist:
            attn = torch.norm(q.unsqueeze(-2) - k.unsqueeze(-3), dim=-1)
        else:
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        attn = attn_transform(attn, x_size)
        # attention
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v  # B_, H, N1, head_dim
        if reshape:
            x = x.transpose(1, 2).reshape(B_, -1, H * head_dim)
        # B_, N, C
        return x


class WindowAttention(Attention):
    r"""Window attention. QKV is the input to the forward method.
    Args:
        num_heads (int): Number of attention heads.
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(
        self,
        input_resolution,
        window_size,
        num_heads,
        window_shift=False,
        attn_drop=0.0,
        pretrained_window_size=[0, 0],
        args=None,
    ):
        super(WindowAttention, self).__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.shift_size = window_size[0] // 2 if window_shift else 0
        self.euclidean_dist = args.euclidean_dist

        self.attn_transform = AffineTransformWindow(
            num_heads,
            input_resolution,
            window_size,
            pretrained_window_size,
            self.shift_size,
            args=args,
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, x_size):
        """
        Args:
            qkv: input QKV features with shape of (B, L, 3C)
            x_size: use x_size to determine whether the relative positional bias table and index
            need to be regenerated.
        """
        H, W = x_size
        B, L, C = qkv.shape
        qkv = qkv.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            qkv = torch.roll(
                qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )

        # partition windows
        qkv = window_partition(qkv, self.window_size)  # nW*B, wh, ww, C
        qkv = qkv.view(-1, prod(self.window_size), C)  # nW*B, wh*ww, C

        B_, N, _ = qkv.shape
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # attention
        x = self.attn(q, k, v, self.attn_transform, x_size)

        # merge windows
        x = x.view(-1, *self.window_size, C // 3)
        x = window_reverse(x, self.window_size, x_size)  # B, H, W, C/3

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(B, L, C // 3)

        return x

    def extra_repr(self) -> str:
        return (
            f"window_size={self.window_size}, shift_size={self.shift_size}, "
            f"pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}"
        )

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class StripeAttention(Attention):
    r"""Stripe attention
    Args:
        stripe_size (tuple[int]): The height and width of the stripe.
        num_heads (int): Number of attention heads.
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        pretrained_stripe_size (tuple[int]): The height and width of the stripe in pre-training.
    """

    def __init__(
        self,
        input_resolution,
        stripe_size,
        stripe_groups,
        stripe_shift,
        num_heads,
        attn_drop=0.0,
        pretrained_stripe_size=[0, 0],
        args=None,
    ):
        super(StripeAttention, self).__init__()
        self.input_resolution = input_resolution
        self.stripe_size = stripe_size  # Wh, Ww
        self.stripe_groups = stripe_groups
        self.stripe_shift = stripe_shift
        self.num_heads = num_heads
        self.pretrained_stripe_size = pretrained_stripe_size
        self.euclidean_dist = args.euclidean_dist

        self.attn_transform = AffineTransformStripe(
            num_heads,
            input_resolution,
            stripe_size,
            stripe_groups,
            stripe_shift,
            pretrained_stripe_size,
            anchor_window_down_factor=1,
            args=args,
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, x_size):
        """
        Args:
            x: input features with shape of (B, L, C)
            stripe_size: use stripe_size to determine whether the relative positional bias table and index
            need to be regenerated.
        """
        H, W = x_size
        B, L, C = qkv.shape
        qkv = qkv.view(B, H, W, C)

        running_stripe_size, running_shift_size = self.attn_transform._get_stripe_info(
            x_size
        )
        # cyclic shift
        if self.stripe_shift:
            qkv = torch.roll(
                qkv,
                shifts=(-running_shift_size[0], -running_shift_size[1]),
                dims=(1, 2),
            )

        # partition windows
        qkv = window_partition(qkv, running_stripe_size)  # nW*B, wh, ww, C
        qkv = qkv.view(-1, prod(running_stripe_size), C)  # nW*B, wh*ww, C

        B_, N, _ = qkv.shape
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # attention
        x = self.attn(q, k, v, self.attn_transform, x_size)

        # merge windows
        x = x.view(-1, *running_stripe_size, C // 3)
        x = window_reverse(x, running_stripe_size, x_size)  # B H W C/3

        # reverse the shift
        if self.stripe_shift:
            x = torch.roll(x, shifts=running_shift_size, dims=(1, 2))

        x = x.view(B, L, C // 3)
        return x

    def extra_repr(self) -> str:
        return (
            f"stripe_size={self.stripe_size}, stripe_groups={self.stripe_groups}, stripe_shift={self.stripe_shift}, "
            f"pretrained_stripe_size={self.pretrained_stripe_size}, num_heads={self.num_heads}"
        )

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class AnchorStripeAttention(Attention):
    r"""Stripe attention
    Args:
        stripe_size (tuple[int]): The height and width of the stripe.
        num_heads (int): Number of attention heads.
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        pretrained_stripe_size (tuple[int]): The height and width of the stripe in pre-training.
    """

    def __init__(
        self,
        input_resolution,
        stripe_size,
        stripe_groups,
        stripe_shift,
        num_heads,
        attn_drop=0.0,
        pretrained_stripe_size=[0, 0],
        anchor_window_down_factor=1,
        args=None,
    ):
        super(AnchorStripeAttention, self).__init__()
        self.input_resolution = input_resolution
        self.stripe_size = stripe_size  # Wh, Ww
        self.stripe_groups = stripe_groups
        self.stripe_shift = stripe_shift
        self.num_heads = num_heads
        self.pretrained_stripe_size = pretrained_stripe_size
        self.anchor_window_down_factor = anchor_window_down_factor
        self.euclidean_dist = args.euclidean_dist

        self.attn_transform1 = AffineTransformStripe(
            num_heads,
            input_resolution,
            stripe_size,
            stripe_groups,
            stripe_shift,
            pretrained_stripe_size,
            anchor_window_down_factor,
            window_to_anchor=False,
            args=args,
        )

        self.attn_transform2 = AffineTransformStripe(
            num_heads,
            input_resolution,
            stripe_size,
            stripe_groups,
            stripe_shift,
            pretrained_stripe_size,
            anchor_window_down_factor,
            window_to_anchor=True,
            args=args,
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, anchor, x_size):
        """
        Args:
            qkv: input features with shape of (B, L, C)
            anchor:
            x_size: use stripe_size to determine whether the relative positional bias table and index
            need to be regenerated.
        """
        H, W = x_size
        B, L, C = qkv.shape
        qkv = qkv.view(B, H, W, C)

        stripe_size, shift_size = self.attn_transform1._get_stripe_info(x_size)
        anchor_stripe_size = [s // self.anchor_window_down_factor for s in stripe_size]
        anchor_shift_size = [s // self.anchor_window_down_factor for s in shift_size]
        # cyclic shift
        if self.stripe_shift:
            qkv = torch.roll(qkv, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            anchor = torch.roll(
                anchor,
                shifts=(-anchor_shift_size[0], -anchor_shift_size[1]),
                dims=(1, 2),
            )

        # partition windows
        qkv = window_partition(qkv, stripe_size)  # nW*B, wh, ww, C
        qkv = qkv.view(-1, prod(stripe_size), C)  # nW*B, wh*ww, C
        anchor = window_partition(anchor, anchor_stripe_size)
        anchor = anchor.view(-1, prod(anchor_stripe_size), C // 3)

        B_, N1, _ = qkv.shape
        N2 = anchor.shape[1]
        qkv = qkv.reshape(B_, N1, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        anchor = anchor.reshape(B_, N2, self.num_heads, -1).permute(0, 2, 1, 3)

        # attention
        x = self.attn(anchor, k, v, self.attn_transform1, x_size, False)
        x = self.attn(q, anchor, x, self.attn_transform2, x_size)

        # merge windows
        x = x.view(B_, *stripe_size, C // 3)
        x = window_reverse(x, stripe_size, x_size)  # B H' W' C

        # reverse the shift
        if self.stripe_shift:
            x = torch.roll(x, shifts=shift_size, dims=(1, 2))

        x = x.view(B, H * W, C // 3)
        return x

    def extra_repr(self) -> str:
        return (
            f"stripe_size={self.stripe_size}, stripe_groups={self.stripe_groups}, stripe_shift={self.stripe_shift}, "
            f"pretrained_stripe_size={self.pretrained_stripe_size}, num_heads={self.num_heads}, anchor_window_down_factor={self.anchor_window_down_factor}"
        )

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias, args):
        m = [
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                groups=in_channels,
                bias=bias,
            )
        ]
        if args.separable_conv_act:
            m.append(nn.GELU())
        m.append(nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias))
        super(SeparableConv, self).__init__(*m)


class QKVProjection(nn.Module):
    def __init__(self, dim, qkv_bias, proj_type, args):
        super(QKVProjection, self).__init__()
        self.proj_type = proj_type
        if proj_type == "linear":
            self.body = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.body = SeparableConv(dim, dim * 3, 3, 1, qkv_bias, args)

    def forward(self, x, x_size):
        if self.proj_type == "separable_conv":
            x = blc_to_bchw(x, x_size)
        x = self.body(x)
        if self.proj_type == "separable_conv":
            x = bchw_to_blc(x)
        return x


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.reduction = nn.Linear(4 * in_dim, out_dim, bias=False)

    def forward(self, x, x_size):
        """
        x: B, H*W, C
        """
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)

        return x


class AnchorLinear(nn.Module):
    r"""Linear anchor projection layer
    Args:
        dim (int): Number of input channels.
    """

    def __init__(self, in_channels, out_channels, down_factor, pooling_mode, bias):
        super().__init__()
        self.down_factor = down_factor
        if pooling_mode == "maxpool":
            self.pooling = nn.MaxPool2d(down_factor, down_factor)
        elif pooling_mode == "avgpool":
            self.pooling = nn.AvgPool2d(down_factor, down_factor)
        self.reduction = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x, x_size):
        """
        x: B, H*W, C
        """
        x = blc_to_bchw(x, x_size)
        x = bchw_to_blc(self.pooling(x))
        x = blc_to_bhwc(self.reduction(x), [s // self.down_factor for s in x_size])
        return x


class AnchorProjection(nn.Module):
    def __init__(self, dim, proj_type, one_stage, anchor_window_down_factor, args):
        super(AnchorProjection, self).__init__()
        self.proj_type = proj_type
        self.body = nn.ModuleList([])
        if one_stage:
            if proj_type == "patchmerging":
                m = PatchMerging(dim, dim // 2)
            elif proj_type == "conv2d":
                kernel_size = anchor_window_down_factor + 1
                stride = anchor_window_down_factor
                padding = kernel_size // 2
                m = nn.Conv2d(dim, dim // 2, kernel_size, stride, padding)
            elif proj_type == "separable_conv":
                kernel_size = anchor_window_down_factor + 1
                stride = anchor_window_down_factor
                m = SeparableConv(dim, dim // 2, kernel_size, stride, True, args)
            elif proj_type.find("pool") >= 0:
                m = AnchorLinear(
                    dim, dim // 2, anchor_window_down_factor, proj_type, True
                )
            self.body.append(m)
        else:
            for i in range(int(math.log2(anchor_window_down_factor))):
                cin = dim if i == 0 else dim // 2
                if proj_type == "patchmerging":
                    m = PatchMerging(cin, dim // 2)
                elif proj_type == "conv2d":
                    m = nn.Conv2d(cin, dim // 2, 3, 2, 1)
                elif proj_type == "separable_conv":
                    m = SeparableConv(cin, dim // 2, 3, 2, True, args)
                self.body.append(m)

    def forward(self, x, x_size):
        if self.proj_type.find("conv") >= 0:
            x = blc_to_bchw(x, x_size)
            for m in self.body:
                x = m(x)
            x = bchw_to_bhwc(x)
        elif self.proj_type.find("pool") >= 0:
            for m in self.body:
                x = m(x, x_size)
        else:
            for i, m in enumerate(self.body):
                x = m(x, [s // 2**i for s in x_size])
            x = blc_to_bhwc(x, [s // 2 ** (i + 1) for s in x_size])
        return x


class MixedAttention(nn.Module):
    r"""Mixed window attention and stripe attention
    Args:
        dim (int): Number of input channels.
        stripe_size (tuple[int]): The height and width of the stripe.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_stripe_size (tuple[int]): The height and width of the stripe in pre-training.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads_w,
        num_heads_s,
        window_size,
        window_shift,
        stripe_size,
        stripe_groups,
        stripe_shift,
        qkv_bias=True,
        qkv_proj_type="linear",
        anchor_proj_type="separable_conv",
        anchor_one_stage=True,
        anchor_window_down_factor=1,
        attn_drop=0.0,
        proj_drop=0.0,
        pretrained_window_size=[0, 0],
        pretrained_stripe_size=[0, 0],
        args=None,
    ):
        super(MixedAttention, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.use_anchor = anchor_window_down_factor > 1
        self.args = args
        # print(args)
        self.qkv = QKVProjection(dim, qkv_bias, qkv_proj_type, args)
        if self.use_anchor:
            # anchor is only used for stripe attention
            self.anchor = AnchorProjection(
                dim, anchor_proj_type, anchor_one_stage, anchor_window_down_factor, args
            )

        self.window_attn = WindowAttention(
            input_resolution,
            window_size,
            num_heads_w,
            window_shift,
            attn_drop,
            pretrained_window_size,
            args,
        )

        if self.args.double_window:
            self.stripe_attn = WindowAttention(
                input_resolution,
                window_size,
                num_heads_w,
                window_shift,
                attn_drop,
                pretrained_window_size,
                args,
            )
        else:
            if self.use_anchor:
                self.stripe_attn = AnchorStripeAttention(
                    input_resolution,
                    stripe_size,
                    stripe_groups,
                    stripe_shift,
                    num_heads_s,
                    attn_drop,
                    pretrained_stripe_size,
                    anchor_window_down_factor,
                    args,
                )
            else:
                if self.args.stripe_square:
                    self.stripe_attn = StripeAttention(
                        input_resolution,
                        window_size,
                        [None, None],
                        window_shift,
                        num_heads_s,
                        attn_drop,
                        pretrained_stripe_size,
                        args,
                    )
                else:
                    self.stripe_attn = StripeAttention(
                        input_resolution,
                        stripe_size,
                        stripe_groups,
                        stripe_shift,
                        num_heads_s,
                        attn_drop,
                        pretrained_stripe_size,
                        args,
                    )
        if self.args.out_proj_type == "linear":
            self.proj = nn.Linear(dim, dim)
        else:
            self.proj = nn.Conv2d(dim, dim, 3, 1, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x_size):
        """
        Args:
            x: input features with shape of (B, L, C)
            stripe_size: use stripe_size to determine whether the relative positional bias table and index
            need to be regenerated.
        """
        B, L, C = x.shape

        # qkv projection
        qkv = self.qkv(x, x_size)
        qkv_window, qkv_stripe = torch.split(qkv, C * 3 // 2, dim=-1)
        # anchor projection
        if self.use_anchor:
            anchor = self.anchor(x, x_size)

        # attention
        x_window = self.window_attn(qkv_window, x_size)
        if self.use_anchor:
            x_stripe = self.stripe_attn(qkv_stripe, anchor, x_size)
        else:
            x_stripe = self.stripe_attn(qkv_stripe, x_size)
        x = torch.cat([x_window, x_stripe], dim=-1)

        # output projection
        if self.args.out_proj_type == "linear":
            x = self.proj(x)
        else:
            x = blc_to_bchw(x, x_size)
            x = bchw_to_blc(self.proj(x))
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}"

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        reduction (int): Channel reduction factor. Default: 16.
    """

    def __init__(self, num_feat, reduction=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // reduction, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // reduction, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=4, reduction=18):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, reduction),
        )

    def forward(self, x, x_size):
        x = self.cab(blc_to_bchw(x, x_size).contiguous())
        return bchw_to_blc(x)


class MixAttnTransformerBlock(nn.Module):
    r"""Mix attention transformer block with shared QKV projection and output projection for mixed attention modules.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_stripe_size (int): Window size in pre-training.
        attn_type (str, optional): Attention type. Default: cwhv.
                    c: residual blocks
                    w: window attention
                    h: horizontal stripe attention
                    v: vertical stripe attention
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads_w,
        num_heads_s,
        window_size=7,
        window_shift=False,
        stripe_size=[8, 8],
        stripe_groups=[None, None],
        stripe_shift=False,
        stripe_type="H",
        mlp_ratio=4.0,
        qkv_bias=True,
        qkv_proj_type="linear",
        anchor_proj_type="separable_conv",
        anchor_one_stage=True,
        anchor_window_down_factor=1,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=[0, 0],
        pretrained_stripe_size=[0, 0],
        res_scale=1.0,
        args=None,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads_w = num_heads_w
        self.num_heads_s = num_heads_s
        self.window_size = window_size
        self.window_shift = window_shift
        self.stripe_shift = stripe_shift
        self.stripe_type = stripe_type
        self.args = args
        if self.stripe_type == "W":
            self.stripe_size = stripe_size[::-1]
            self.stripe_groups = stripe_groups[::-1]
        else:
            self.stripe_size = stripe_size
            self.stripe_groups = stripe_groups
        self.mlp_ratio = mlp_ratio
        self.res_scale = res_scale

        self.attn = MixedAttention(
            dim,
            input_resolution,
            num_heads_w,
            num_heads_s,
            window_size,
            window_shift,
            self.stripe_size,
            self.stripe_groups,
            stripe_shift,
            qkv_bias,
            qkv_proj_type,
            anchor_proj_type,
            anchor_one_stage,
            anchor_window_down_factor,
            attn_drop,
            drop,
            pretrained_window_size,
            pretrained_stripe_size,
            args,
        )
        self.norm1 = norm_layer(dim)
        if self.args.local_connection:
            self.conv = CAB(dim)

    def forward(self, x, x_size):
        # Mixed attention
        if self.args.local_connection:
            x = (
                x
                + self.res_scale * self.drop_path(self.norm1(self.attn(x, x_size)))
                + self.conv(x, x_size)
            )
        else:
            x = x + self.res_scale * self.drop_path(self.norm1(self.attn(x, x_size)))
        # FFN
        x = x + self.res_scale * self.drop_path(self.norm2(self.mlp(x)))

    #         return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads=({self.num_heads_w}, {self.num_heads_s}), "
            f"window_size={self.window_size}, window_shift={self.window_shift}, "
            f"stripe_size={self.stripe_size}, stripe_groups={self.stripe_groups}, stripe_shift={self.stripe_shift}, self.stripe_type={self.stripe_type}, "
            f"mlp_ratio={self.mlp_ratio}, res_scale={self.res_scale}"
        )


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttentionV1(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_pe=True,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.use_pe = use_pe

        if self.use_pe:
            # define a parameter table of relative position bias
            ws = self.window_size
            table = torch.zeros((2 * ws[0] - 1) * (2 * ws[1] - 1), num_heads)
            self.relative_position_bias_table = nn.Parameter(table)
            # 2*Wh-1 * 2*Ww-1, nH
            trunc_normal_(self.relative_position_bias_table, std=0.02)

            self.get_relative_position_index(self.window_size)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def get_relative_position_index(self, window_size):
        # get pair-wise relative position index for each token inside the window
        coord_h = torch.arange(window_size[0])
        coord_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coord_h, coord_w]))  # 2, Wh, Ww
        coords = torch.flatten(coords, 1)  # 2, Wh*Ww
        coords = coords[:, :, None] - coords[:, None, :]  # 2, Wh*Ww, Wh*Ww
        coords = coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        coords[:, :, 1] += window_size[1] - 1
        coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        # qkv projection
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # attention map
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # positional encoding
        if self.use_pe:
            win_dim = prod(self.window_size)
            bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ]
            bias = bias.view(win_dim, win_dim, -1).permute(2, 0, 1).contiguous()
            # nH, Wh*Ww, Wh*Ww
            attn = attn + bias.unsqueeze(0)

        # shift attention mask
        if mask is not None:
            nW = mask.shape[0]
            mask = mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask
            attn = attn.view(-1, self.num_heads, N, N)

        # attention
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        # output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class WindowAttentionWrapperV1(WindowAttentionV1):
    def __init__(self, shift_size, input_resolution, **kwargs):
        super(WindowAttentionWrapperV1, self).__init__(**kwargs)
        self.shift_size = shift_size
        self.input_resolution = input_resolution

        if self.shift_size > 0:
            attn_mask = calculate_mask(input_resolution, self.window_size, shift_size)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # partition windows
        x = window_partition(x, self.window_size)  # nW*B, wh, ww, C
        x = x.view(-1, prod(self.window_size), C)  # nW*B, wh*ww, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_mask = self.attn_mask
        else:
            attn_mask = calculate_mask(x_size, self.window_size, self.shift_size)
            attn_mask = attn_mask.to(x.device)

        # attention
        x = super(WindowAttentionWrapperV1, self).forward(x, mask=attn_mask)
        # nW*B, wh*ww, C

        # merge windows
        x = x.view(-1, *self.window_size, C)
        x = window_reverse(x, self.window_size, x_size)  # B, H, W, C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(B, H * W, C)

        return x


class SwinTransformerBlockV1(nn.Module):
    r"""Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_pe=True,
        res_scale=1.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"
        self.res_scale = res_scale

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttentionWrapperV1(
            shift_size=self.shift_size,
            input_resolution=self.input_resolution,
            dim=dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_pe=use_pe,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, x_size):
        # Window attention
        x = x + self.res_scale * self.drop_path(self.attn(self.norm1(x), x_size))
        # FFN
        x = x + self.res_scale * self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}, res_scale={self.res_scale}"
        )

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r"""Image to Patch Unembedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        B, C, H, W = x.shape
        x = bchw_to_blc(x)
        x = super(Linear, self).forward(x)
        x = blc_to_bchw(x, (H, W))
        return x


def build_last_conv(conv_type, dim):
    if conv_type == "1conv":
        block = nn.Conv2d(dim, dim, 3, 1, 1)
    elif conv_type == "3conv":
        # to save parameters and memory
        block = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim // 4, dim, 3, 1, 1),
        )
    elif conv_type == "1conv1x1":
        block = nn.Conv2d(dim, dim, 1, 1, 0)
    elif conv_type == "linear":
        block = Linear(dim, dim)
    return block


def bchw_to_bhwc(x: torch.Tensor) -> torch.Tensor:
    """Permutes a tensor from the shape (B, C, H, W) to (B, H, W, C)."""
    return x.permute(0, 2, 3, 1)


def bhwc_to_bchw(x: torch.Tensor) -> torch.Tensor:
    """Permutes a tensor from the shape (B, H, W, C) to (B, C, H, W)."""
    return x.permute(0, 3, 1, 2)


def bchw_to_blc(x: torch.Tensor) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, C, H, W) to (B, L, C)."""
    return x.flatten(2).transpose(1, 2)


def blc_to_bchw(x: torch.Tensor, x_size: Tuple) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, L, C) to (B, C, H, W)."""
    B, L, C = x.shape
    return x.transpose(1, 2).view(B, C, *x_size)


def blc_to_bhwc(x: torch.Tensor, x_size: Tuple) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, L, C) to (B, H, W, C)."""
    B, L, C = x.shape
    return x.view(B, *x_size, C)


def window_partition(x, window_size: Tuple[int, int]):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(
        B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size[0], window_size[1], C)
    )
    return windows


def window_reverse(windows, window_size: Tuple[int, int], img_size: Tuple[int, int]):
    """
    Args:
        windows: (num_windows * B, window_size[0], window_size[1], C)
        window_size (Tuple[int, int]): Window size
        img_size (Tuple[int, int]): Image size

    Returns:
        x: (B, H, W, C)
    """
    H, W = img_size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(
        B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def _fill_window(input_resolution, window_size, shift_size=None):
    if shift_size is None:
        shift_size = [s // 2 for s in window_size]

    img_mask = torch.zeros((1, *input_resolution, 1))  # 1 H W 1
    h_slices = (
        slice(0, -window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    )
    w_slices = (
        slice(0, -window_size[1]),
        slice(-window_size[1], -shift_size[1]),
        slice(-shift_size[1], None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, prod(window_size))
    return mask_windows


#####################################
# Different versions of the functions
# 1) Swin Transformer, SwinIR, Square window attention in GRL;
# 2) Early development of the decomposition-based efficient attention mechanism (efficient_win_attn.py);
# 3) GRL. Window-anchor attention mechanism.
# 1) & 3) are still useful
#####################################


def calculate_mask(input_resolution, window_size, shift_size):
    """
    Use case: 1)
    """
    # calculate attention mask for SW-MSA
    if isinstance(shift_size, int):
        shift_size = to_2tuple(shift_size)
    mask_windows = _fill_window(input_resolution, window_size, shift_size)

    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )  # nW, window_size**2, window_size**2

    return attn_mask


def calculate_mask_all(
    input_resolution,
    window_size,
    shift_size,
    anchor_window_down_factor=1,
    window_to_anchor=True,
):
    """
    Use case: 3)
    """
    # calculate attention mask for SW-MSA
    anchor_resolution = [s // anchor_window_down_factor for s in input_resolution]
    aws = [s // anchor_window_down_factor for s in window_size]
    anchor_shift = [s // anchor_window_down_factor for s in shift_size]

    # mask of window1: nW, Wh**Ww
    mask_windows = _fill_window(input_resolution, window_size, shift_size)
    # mask of window2: nW, AWh*AWw
    mask_anchor = _fill_window(anchor_resolution, aws, anchor_shift)

    if window_to_anchor:
        attn_mask = mask_windows.unsqueeze(2) - mask_anchor.unsqueeze(1)
    else:
        attn_mask = mask_anchor.unsqueeze(2) - mask_windows.unsqueeze(1)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )  # nW, Wh**Ww, AWh*AWw

    return attn_mask


def calculate_win_mask(
    input_resolution1, input_resolution2, window_size1, window_size2
):
    """
    Use case: 2)
    """
    # calculate attention mask for SW-MSA

    # mask of window1: nW, Wh**Ww
    mask_windows1 = _fill_window(input_resolution1, window_size1)
    # mask of window2: nW, AWh*AWw
    mask_windows2 = _fill_window(input_resolution2, window_size2)

    attn_mask = mask_windows1.unsqueeze(2) - mask_windows2.unsqueeze(1)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )  # nW, Wh**Ww, AWh*AWw

    return attn_mask


def _get_meshgrid_coords(start_coords, end_coords):
    coord_h = torch.arange(start_coords[0], end_coords[0])
    coord_w = torch.arange(start_coords[1], end_coords[1])
    coords = torch.stack(torch.meshgrid([coord_h, coord_w], indexing="ij"))  # 2, Wh, Ww
    coords = torch.flatten(coords, 1)  # 2, Wh*Ww
    return coords


def get_relative_coords_table(
    window_size, pretrained_window_size=[0, 0], anchor_window_down_factor=1
):
    """
    Use case: 1)
    """
    # get relative_coords_table
    ws = window_size
    aws = [w // anchor_window_down_factor for w in window_size]
    pws = pretrained_window_size
    paws = [w // anchor_window_down_factor for w in pretrained_window_size]

    ts = [(w1 + w2) // 2 for w1, w2 in zip(ws, aws)]
    pts = [(w1 + w2) // 2 for w1, w2 in zip(pws, paws)]

    # TODO: pretrained window size and pretrained anchor window size is only used here.
    # TODO: Investigate whether it is really important to use this setting when finetuning large window size
    # TODO: based on pretrained weights with small window size.

    coord_h = torch.arange(-(ts[0] - 1), ts[0], dtype=torch.float32)
    coord_w = torch.arange(-(ts[1] - 1), ts[1], dtype=torch.float32)
    table = torch.stack(torch.meshgrid([coord_h, coord_w], indexing="ij")).permute(
        1, 2, 0
    )
    table = table.contiguous().unsqueeze(0)  # 1, Wh+AWh-1, Ww+AWw-1, 2
    if pts[0] > 0:
        table[:, :, :, 0] /= pts[0] - 1
        table[:, :, :, 1] /= pts[1] - 1
    else:
        table[:, :, :, 0] /= ts[0] - 1
        table[:, :, :, 1] /= ts[1] - 1
    table *= 8  # normalize to -8, 8
    table = torch.sign(table) * torch.log2(torch.abs(table) + 1.0) / np.log2(8)
    return table


def get_relative_coords_table_all(
    window_size, pretrained_window_size=[0, 0], anchor_window_down_factor=1
):
    """
    Use case: 3)

    Support all window shapes.
    Args:
        window_size:
        pretrained_window_size:
        anchor_window_down_factor:

    Returns:

    """
    # get relative_coords_table
    ws = window_size
    aws = [w // anchor_window_down_factor for w in window_size]
    pws = pretrained_window_size
    paws = [w // anchor_window_down_factor for w in pretrained_window_size]

    # positive table size: (Ww - 1) - (Ww - AWw) // 2
    ts_p = [w1 - 1 - (w1 - w2) // 2 for w1, w2 in zip(ws, aws)]
    # negative table size: -(AWw - 1) - (Ww - AWw) // 2
    ts_n = [-(w2 - 1) - (w1 - w2) // 2 for w1, w2 in zip(ws, aws)]
    pts = [w1 - 1 - (w1 - w2) // 2 for w1, w2 in zip(pws, paws)]

    # TODO: pretrained window size and pretrained anchor window size is only used here.
    # TODO: Investigate whether it is really important to use this setting when finetuning large window size
    # TODO: based on pretrained weights with small window size.

    coord_h = torch.arange(ts_n[0], ts_p[0] + 1, dtype=torch.float32)
    coord_w = torch.arange(ts_n[1], ts_p[1] + 1, dtype=torch.float32)
    table = torch.stack(torch.meshgrid([coord_h, coord_w], indexing="ij")).permute(
        1, 2, 0
    )
    table = table.contiguous().unsqueeze(0)  # 1, Wh+AWh-1, Ww+AWw-1, 2
    if pts[0] > 0:
        table[:, :, :, 0] /= pts[0]
        table[:, :, :, 1] /= pts[1]
    else:
        table[:, :, :, 0] /= ts_p[0]
        table[:, :, :, 1] /= ts_p[1]
    table *= 8  # normalize to -8, 8
    table = torch.sign(table) * torch.log2(torch.abs(table) + 1.0) / np.log2(8)
    # 1, Wh+AWh-1, Ww+AWw-1, 2
    return table


def coords_diff(coords1, coords2, max_diff):
    # The coordinates starts from (-start_coord[0], -start_coord[1])
    coords = coords1[:, :, None] - coords2[:, None, :]  # 2, Wh*Ww, AWh*AWw
    coords = coords.permute(1, 2, 0).contiguous()  # Wh*Ww, AWh*AWw, 2
    coords[:, :, 0] += max_diff[0] - 1  # shift to start from 0
    coords[:, :, 1] += max_diff[1] - 1
    coords[:, :, 0] *= 2 * max_diff[1] - 1
    idx = coords.sum(-1)  # Wh*Ww, AWh*AWw
    return idx


def get_relative_position_index(
    window_size, anchor_window_down_factor=1, window_to_anchor=True
):
    """
    Use case: 1)
    """
    # get pair-wise relative position index for each token inside the window
    ws = window_size
    aws = [w // anchor_window_down_factor for w in window_size]
    coords_anchor_end = [(w1 + w2) // 2 for w1, w2 in zip(ws, aws)]
    coords_anchor_start = [(w1 - w2) // 2 for w1, w2 in zip(ws, aws)]

    coords = _get_meshgrid_coords((0, 0), window_size)  # 2, Wh*Ww
    coords_anchor = _get_meshgrid_coords(coords_anchor_start, coords_anchor_end)
    # 2, AWh*AWw

    if window_to_anchor:
        idx = coords_diff(coords, coords_anchor, max_diff=coords_anchor_end)
    else:
        idx = coords_diff(coords_anchor, coords, max_diff=coords_anchor_end)
    return idx  # Wh*Ww, AWh*AWw or AWh*AWw, Wh*Ww


def coords_diff_odd(coords1, coords2, start_coord, max_diff):
    # The coordinates starts from (-start_coord[0], -start_coord[1])
    coords = coords1[:, :, None] - coords2[:, None, :]  # 2, Wh*Ww, AWh*AWw
    coords = coords.permute(1, 2, 0).contiguous()  # Wh*Ww, AWh*AWw, 2
    coords[:, :, 0] += start_coord[0]  # shift to start from 0
    coords[:, :, 1] += start_coord[1]
    coords[:, :, 0] *= max_diff
    idx = coords.sum(-1)  # Wh*Ww, AWh*AWw
    return idx


def get_relative_position_index_all(
    window_size, anchor_window_down_factor=1, window_to_anchor=True
):
    """
    Use case: 3)
    Support all window shapes:
        square window - square window
        rectangular window - rectangular window
        window - anchor
        anchor - window
        [8, 8] - [8, 8]
        [4, 86] - [2, 43]
    """
    # get pair-wise relative position index for each token inside the window
    ws = window_size
    aws = [w // anchor_window_down_factor for w in window_size]
    coords_anchor_start = [(w1 - w2) // 2 for w1, w2 in zip(ws, aws)]
    coords_anchor_end = [s + w2 for s, w2 in zip(coords_anchor_start, aws)]

    coords = _get_meshgrid_coords((0, 0), window_size)  # 2, Wh*Ww
    coords_anchor = _get_meshgrid_coords(coords_anchor_start, coords_anchor_end)
    # 2, AWh*AWw

    max_horizontal_diff = aws[1] + ws[1] - 1
    if window_to_anchor:
        offset = [w2 + s - 1 for s, w2 in zip(coords_anchor_start, aws)]
        idx = coords_diff_odd(coords, coords_anchor, offset, max_horizontal_diff)
    else:
        offset = [w1 - s - 1 for s, w1 in zip(coords_anchor_start, ws)]
        idx = coords_diff_odd(coords_anchor, coords, offset, max_horizontal_diff)
    return idx  # Wh*Ww, AWh*AWw or AWh*AWw, Wh*Ww


def get_relative_position_index_simple(
    window_size, anchor_window_down_factor=1, window_to_anchor=True
):
    """
    Use case: 3)
    This is a simplified version of get_relative_position_index_all
    The start coordinate of anchor window is also (0, 0)
    get pair-wise relative position index for each token inside the window
    """
    ws = window_size
    aws = [w // anchor_window_down_factor for w in window_size]

    coords = _get_meshgrid_coords((0, 0), window_size)  # 2, Wh*Ww
    coords_anchor = _get_meshgrid_coords((0, 0), aws)
    # 2, AWh*AWw

    max_horizontal_diff = aws[1] + ws[1] - 1
    if window_to_anchor:
        offset = [w2 - 1 for w2 in aws]
        idx = coords_diff_odd(coords, coords_anchor, offset, max_horizontal_diff)
    else:
        offset = [w1 - 1 for w1 in ws]
        idx = coords_diff_odd(coords_anchor, coords, offset, max_horizontal_diff)
    return idx  # Wh*Ww, AWh*AWw or AWh*AWw, Wh*Ww


def get_relative_win_position_index(window_size, anchor_window_size):
    """
    Use case: 2)
    """
    # get pair-wise relative position index for each token inside the window
    ws = window_size
    aws = anchor_window_size
    coords_anchor_end = [(w1 + w2) // 2 for w1, w2 in zip(ws, aws)]
    coords_anchor_start = [(w1 - w2) // 2 for w1, w2 in zip(ws, aws)]

    coords = _get_meshgrid_coords((0, 0), window_size)  # 2, Wh*Ww
    coords_anchor = _get_meshgrid_coords(coords_anchor_start, coords_anchor_end)
    # 2, AWh*AWw
    coords = coords[:, :, None] - coords_anchor[:, None, :]  # 2, Wh*Ww, AWh*AWw
    coords = coords.permute(1, 2, 0).contiguous()  # Wh*Ww, AWh*AWw, 2
    coords[:, :, 0] += coords_anchor_end[0] - 1  # shift to start from 0
    coords[:, :, 1] += coords_anchor_end[1] - 1
    coords[:, :, 0] *= 2 * coords_anchor_end[1] - 1
    idx = coords.sum(-1)  # Wh*Ww, AWh*AWw
    return idx


def get_relative_win_coords_table(
    window_size,
    anchor_window_size,
    pretrained_window_size=[0, 0],
    pretrained_anchor_window_size=[0, 0],
):
    """
    Use case: 2)
    """
    # get relative_coords_table
    ws = window_size
    aws = anchor_window_size
    pws = pretrained_window_size
    paws = pretrained_anchor_window_size

    # TODO: pretrained window size and pretrained anchor window size is only used here.
    # TODO: Investigate whether it is really important to use this setting when finetuning large window size
    # TODO: based on pretrained weights with small window size.

    table_size = [(wsi + awsi) // 2 for wsi, awsi in zip(ws, aws)]
    table_size_pretrained = [(pwsi + pawsi) // 2 for pwsi, pawsi in zip(pws, paws)]
    coord_h = torch.arange(-(table_size[0] - 1), table_size[0], dtype=torch.float32)
    coord_w = torch.arange(-(table_size[1] - 1), table_size[1], dtype=torch.float32)
    table = torch.stack(torch.meshgrid([coord_h, coord_w], indexing="ij")).permute(
        1, 2, 0
    )
    table = table.contiguous().unsqueeze(0)  # 1, Wh+AWh-1, Ww+AWw-1, 2
    if table_size_pretrained[0] > 0:
        table[:, :, :, 0] /= table_size_pretrained[0] - 1
        table[:, :, :, 1] /= table_size_pretrained[1] - 1
    else:
        table[:, :, :, 0] /= table_size[0] - 1
        table[:, :, :, 1] /= table_size[1] - 1
    table *= 8  # normalize to -8, 8
    table = torch.sign(table) * torch.log2(torch.abs(table) + 1.0) / np.log2(8)
    return table


class AffineTransform(nn.Module):
    r"""Affine transformation of the attention map.
    The window could be a square window or a stripe window. Supports attention between different window sizes
    """

    def __init__(self, num_heads):
        super(AffineTransform, self).__init__()
        logit_scale = torch.log(10 * torch.ones((num_heads, 1, 1)))
        self.logit_scale = nn.Parameter(logit_scale, requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = CPB_MLP(2, num_heads)

    def forward(self, attn, relative_coords_table, relative_position_index, mask):
        B_, H, N1, N2 = attn.shape
        # logit scale
        attn = attn * torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()

        bias_table = self.cpb_mlp(relative_coords_table)  # 2*Wh-1, 2*Ww-1, num_heads
        bias_table = bias_table.view(-1, H)

        bias = bias_table[relative_position_index.view(-1)]
        bias = bias.view(N1, N2, -1).permute(2, 0, 1).contiguous()
        # nH, Wh*Ww, Wh*Ww
        bias = 16 * torch.sigmoid(bias)
        attn = attn + bias.unsqueeze(0)

        # W-MSA/SW-MSA
        # shift attention mask
        if mask is not None:
            nW = mask.shape[0]
            mask = mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(B_ // nW, nW, H, N1, N2) + mask
            attn = attn.view(-1, H, N1, N2)

        return attn


def _get_stripe_info(stripe_size_in, stripe_groups_in, stripe_shift, input_resolution):
    stripe_size, shift_size = [], []
    for s, g, d in zip(stripe_size_in, stripe_groups_in, input_resolution):
        if g is None:
            stripe_size.append(s)
            shift_size.append(s // 2 if stripe_shift else 0)
        else:
            stripe_size.append(d // g)
            shift_size.append(0 if g == 1 else d // (g * 2))
    return stripe_size, shift_size


class Attention(ABC, nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def attn(self, q, k, v, attn_transform, table, index, mask, reshape=True):
        # q, k, v: # nW*B, H, wh*ww, dim
        # cosine attention map
        B_, _, H, head_dim = q.shape
        if self.euclidean_dist:
            # print("use euclidean distance")
            attn = torch.norm(q.unsqueeze(-2) - k.unsqueeze(-3), dim=-1)
        else:
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        attn = attn_transform(attn, table, index, mask)
        # attention
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v  # B_, H, N1, head_dim
        if reshape:
            x = x.transpose(1, 2).reshape(B_, -1, H * head_dim)
        # B_, N, C
        return x


class WindowAttention(Attention):
    r"""Window attention. QKV is the input to the forward method.
    Args:
        num_heads (int): Number of attention heads.
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(
        self,
        input_resolution,
        window_size,
        num_heads,
        window_shift=False,
        attn_drop=0.0,
        pretrained_window_size=[0, 0],
        args=None,
    ):
        super(WindowAttention, self).__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.shift_size = window_size[0] // 2 if window_shift else 0
        self.euclidean_dist = args.euclidean_dist

        self.attn_transform = AffineTransform(num_heads)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, x_size, table, index, mask):
        """
        Args:
            qkv: input QKV features with shape of (B, L, 3C)
            x_size: use x_size to determine whether the relative positional bias table and index
            need to be regenerated.
        """
        H, W = x_size
        B, L, C = qkv.shape
        qkv = qkv.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            qkv = torch.roll(
                qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )

        # partition windows
        qkv = window_partition(qkv, self.window_size)  # nW*B, wh, ww, C
        qkv = qkv.view(-1, prod(self.window_size), C)  # nW*B, wh*ww, C

        B_, N, _ = qkv.shape
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # nW*B, H, wh*ww, dim

        # attention
        x = self.attn(q, k, v, self.attn_transform, table, index, mask)

        # merge windows
        x = x.view(-1, *self.window_size, C // 3)
        x = window_reverse(x, self.window_size, x_size)  # B, H, W, C/3

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(B, L, C // 3)

        return x

    def extra_repr(self) -> str:
        return (
            f"window_size={self.window_size}, shift_size={self.shift_size}, "
            f"pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}"
        )

    def flops(self, N):
        pass


class AnchorStripeAttention(Attention):
    r"""Stripe attention
    Args:
        stripe_size (tuple[int]): The height and width of the stripe.
        num_heads (int): Number of attention heads.
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        pretrained_stripe_size (tuple[int]): The height and width of the stripe in pre-training.
    """

    def __init__(
        self,
        input_resolution,
        stripe_size,
        stripe_groups,
        stripe_shift,
        num_heads,
        attn_drop=0.0,
        pretrained_stripe_size=[0, 0],
        anchor_window_down_factor=1,
        args=None,
    ):
        super(AnchorStripeAttention, self).__init__()
        self.input_resolution = input_resolution
        self.stripe_size = stripe_size  # Wh, Ww
        self.stripe_groups = stripe_groups
        self.stripe_shift = stripe_shift
        self.num_heads = num_heads
        self.pretrained_stripe_size = pretrained_stripe_size
        self.anchor_window_down_factor = anchor_window_down_factor
        self.euclidean_dist = args.euclidean_dist

        self.attn_transform1 = AffineTransform(num_heads)
        self.attn_transform2 = AffineTransform(num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, qkv, anchor, x_size, table, index_a2w, index_w2a, mask_a2w, mask_w2a
    ):
        """
        Args:
            qkv: input features with shape of (B, L, C)
            anchor:
            x_size: use stripe_size to determine whether the relative positional bias table and index
            need to be regenerated.
        """
        H, W = x_size
        B, L, C = qkv.shape
        qkv = qkv.view(B, H, W, C)

        stripe_size, shift_size = _get_stripe_info(
            self.stripe_size, self.stripe_groups, self.stripe_shift, x_size
        )
        anchor_stripe_size = [s // self.anchor_window_down_factor for s in stripe_size]
        anchor_shift_size = [s // self.anchor_window_down_factor for s in shift_size]
        # cyclic shift
        if self.stripe_shift:
            qkv = torch.roll(qkv, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            anchor = torch.roll(
                anchor,
                shifts=(-anchor_shift_size[0], -anchor_shift_size[1]),
                dims=(1, 2),
            )

        # partition windows
        qkv = window_partition(qkv, stripe_size)  # nW*B, wh, ww, C
        qkv = qkv.view(-1, prod(stripe_size), C)  # nW*B, wh*ww, C
        anchor = window_partition(anchor, anchor_stripe_size)
        anchor = anchor.view(-1, prod(anchor_stripe_size), C // 3)

        B_, N1, _ = qkv.shape
        N2 = anchor.shape[1]
        qkv = qkv.reshape(B_, N1, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        anchor = anchor.reshape(B_, N2, self.num_heads, -1).permute(0, 2, 1, 3)

        # attention
        x = self.attn(
            anchor, k, v, self.attn_transform1, table, index_a2w, mask_a2w, False
        )
        x = self.attn(q, anchor, x, self.attn_transform2, table, index_w2a, mask_w2a)

        # merge windows
        x = x.view(B_, *stripe_size, C // 3)
        x = window_reverse(x, stripe_size, x_size)  # B H' W' C

        # reverse the shift
        if self.stripe_shift:
            x = torch.roll(x, shifts=shift_size, dims=(1, 2))

        x = x.view(B, H * W, C // 3)
        return x

    def extra_repr(self) -> str:
        return (
            f"stripe_size={self.stripe_size}, stripe_groups={self.stripe_groups}, stripe_shift={self.stripe_shift}, "
            f"pretrained_stripe_size={self.pretrained_stripe_size}, num_heads={self.num_heads}, anchor_window_down_factor={self.anchor_window_down_factor}"
        )

    def flops(self, N):
        pass


class MixedAttention(nn.Module):
    r"""Mixed window attention and stripe attention
    Args:
        dim (int): Number of input channels.
        stripe_size (tuple[int]): The height and width of the stripe.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_stripe_size (tuple[int]): The height and width of the stripe in pre-training.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads_w,
        num_heads_s,
        window_size,
        window_shift,
        stripe_size,
        stripe_groups,
        stripe_shift,
        qkv_bias=True,
        qkv_proj_type="linear",
        anchor_proj_type="separable_conv",
        anchor_one_stage=True,
        anchor_window_down_factor=1,
        attn_drop=0.0,
        proj_drop=0.0,
        pretrained_window_size=[0, 0],
        pretrained_stripe_size=[0, 0],
        args=None,
    ):
        super(MixedAttention, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.args = args
        # print(args)
        self.qkv = QKVProjection(dim, qkv_bias, qkv_proj_type, args)
        # anchor is only used for stripe attention
        self.anchor = AnchorProjection(
            dim, anchor_proj_type, anchor_one_stage, anchor_window_down_factor, args
        )

        self.window_attn = WindowAttention(
            input_resolution,
            window_size,
            num_heads_w,
            window_shift,
            attn_drop,
            pretrained_window_size,
            args,
        )
        self.stripe_attn = AnchorStripeAttention(
            input_resolution,
            stripe_size,
            stripe_groups,
            stripe_shift,
            num_heads_s,
            attn_drop,
            pretrained_stripe_size,
            anchor_window_down_factor,
            args,
        )
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x_size, table_index_mask):
        """
        Args:
            x: input features with shape of (B, L, C)
            stripe_size: use stripe_size to determine whether the relative positional bias table and index
            need to be regenerated.
        """
        B, L, C = x.shape

        # qkv projection
        qkv = self.qkv(x, x_size)
        qkv_window, qkv_stripe = torch.split(qkv, C * 3 // 2, dim=-1)
        # anchor projection
        anchor = self.anchor(x, x_size)

        # attention
        x_window = self.window_attn(
            qkv_window, x_size, *self._get_table_index_mask(table_index_mask, True)
        )
        x_stripe = self.stripe_attn(
            qkv_stripe,
            anchor,
            x_size,
            *self._get_table_index_mask(table_index_mask, False),
        )
        x = torch.cat([x_window, x_stripe], dim=-1)

        # output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _get_table_index_mask(self, table_index_mask, window_attn=True):
        if window_attn:
            return (
                table_index_mask["table_w"],
                table_index_mask["index_w"],
                table_index_mask["mask_w"],
            )
        else:
            return (
                table_index_mask["table_s"],
                table_index_mask["index_a2w"],
                table_index_mask["index_w2a"],
                table_index_mask["mask_a2w"],
                table_index_mask["mask_w2a"],
            )

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}"

    def flops(self, N):
        pass


class EfficientMixAttnTransformerBlock(nn.Module):
    r"""Mix attention transformer block with shared QKV projection and output projection for mixed attention modules.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_stripe_size (int): Window size in pre-training.
        attn_type (str, optional): Attention type. Default: cwhv.
                    c: residual blocks
                    w: window attention
                    h: horizontal stripe attention
                    v: vertical stripe attention
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads_w,
        num_heads_s,
        window_size=7,
        window_shift=False,
        stripe_size=[8, 8],
        stripe_groups=[None, None],
        stripe_shift=False,
        stripe_type="H",
        mlp_ratio=4.0,
        qkv_bias=True,
        qkv_proj_type="linear",
        anchor_proj_type="separable_conv",
        anchor_one_stage=True,
        anchor_window_down_factor=1,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=[0, 0],
        pretrained_stripe_size=[0, 0],
        res_scale=1.0,
        args=None,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads_w = num_heads_w
        self.num_heads_s = num_heads_s
        self.window_size = window_size
        self.window_shift = window_shift
        self.stripe_shift = stripe_shift
        self.stripe_type = stripe_type
        self.args = args
        if self.stripe_type == "W":
            self.stripe_size = stripe_size[::-1]
            self.stripe_groups = stripe_groups[::-1]
        else:
            self.stripe_size = stripe_size
            self.stripe_groups = stripe_groups
        self.mlp_ratio = mlp_ratio
        self.res_scale = res_scale

        self.attn = MixedAttention(
            dim,
            input_resolution,
            num_heads_w,
            num_heads_s,
            window_size,
            window_shift,
            self.stripe_size,
            self.stripe_groups,
            stripe_shift,
            qkv_bias,
            qkv_proj_type,
            anchor_proj_type,
            anchor_one_stage,
            anchor_window_down_factor,
            attn_drop,
            drop,
            pretrained_window_size,
            pretrained_stripe_size,
            args,
        )
        self.norm1 = norm_layer(dim)
        if self.args.local_connection:
            self.conv = CAB(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.norm2 = norm_layer(dim)

    def _get_table_index_mask(self, all_table_index_mask):
        table_index_mask = {
            "table_w": all_table_index_mask["table_w"],
            "index_w": all_table_index_mask["index_w"],
        }
        if self.stripe_type == "W":
            table_index_mask["table_s"] = all_table_index_mask["table_sv"]
            table_index_mask["index_a2w"] = all_table_index_mask["index_sv_a2w"]
            table_index_mask["index_w2a"] = all_table_index_mask["index_sv_w2a"]
        else:
            table_index_mask["table_s"] = all_table_index_mask["table_sh"]
            table_index_mask["index_a2w"] = all_table_index_mask["index_sh_a2w"]
            table_index_mask["index_w2a"] = all_table_index_mask["index_sh_w2a"]
        if self.window_shift:
            table_index_mask["mask_w"] = all_table_index_mask["mask_w"]
        else:
            table_index_mask["mask_w"] = None
        if self.stripe_shift:
            if self.stripe_type == "W":
                table_index_mask["mask_a2w"] = all_table_index_mask["mask_sv_a2w"]
                table_index_mask["mask_w2a"] = all_table_index_mask["mask_sv_w2a"]
            else:
                table_index_mask["mask_a2w"] = all_table_index_mask["mask_sh_a2w"]
                table_index_mask["mask_w2a"] = all_table_index_mask["mask_sh_w2a"]
        else:
            table_index_mask["mask_a2w"] = None
            table_index_mask["mask_w2a"] = None
        return table_index_mask

    def forward(self, x, x_size, all_table_index_mask):
        # Mixed attention
        table_index_mask = self._get_table_index_mask(all_table_index_mask)
        if self.args.local_connection:
            x = (
                x
                + self.res_scale
                * self.drop_path(self.norm1(self.attn(x, x_size, table_index_mask)))
                + self.conv(x, x_size)
            )
        else:
            x = x + self.res_scale * self.drop_path(
                self.norm1(self.attn(x, x_size, table_index_mask))
            )
        # FFN
        x = x + self.res_scale * self.drop_path(self.norm2(self.mlp(x)))

        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads=({self.num_heads_w}, {self.num_heads_s}), "
            f"window_size={self.window_size}, window_shift={self.window_shift}, "
            f"stripe_size={self.stripe_size}, stripe_groups={self.stripe_groups}, stripe_shift={self.stripe_shift}, self.stripe_type={self.stripe_type}, "
            f"mlp_ratio={self.mlp_ratio}, res_scale={self.res_scale}"
        )

    def flops(self):
        pass


class Upsample(nn.Module):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        super(Upsample, self).__init__()
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(
                f"scale {scale} is not supported. " "Supported scales: 2^n and 3."
            )
        self.up = nn.Sequential(*m)

    def forward(self, x):
        return self.up(x)


class UpsampleOneStep(nn.Module):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, num_out_ch):
        super(UpsampleOneStep, self).__init__()
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        self.up = nn.Sequential(*m)

    def forward(self, x):
        return self.up(x)


class TransformerStage(nn.Module):
    """Transformer stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads_window (list[int]): Number of window attention heads in different layers.
        num_heads_stripe (list[int]): Number of stripe attention heads in different layers.
        stripe_size (list[int]): Stripe size. Default: [8, 8]
        stripe_groups (list[int]): Number of stripe groups. Default: [None, None].
        stripe_shift (bool): whether to shift the stripes. This is used as an ablation study.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qkv_proj_type (str): QKV projection type. Default: linear. Choices: linear, separable_conv.
        anchor_proj_type (str): Anchor projection type. Default: avgpool. Choices: avgpool, maxpool, conv2d, separable_conv, patchmerging.
        anchor_one_stage (bool): Whether to use one operator or multiple progressive operators to reduce feature map resolution. Default: True.
        anchor_window_down_factor (int): The downscale factor used to get the anchors.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        pretrained_window_size (list[int]): pretrained window size. This is actually not used. Default: [0, 0].
        pretrained_stripe_size (list[int]): pretrained stripe size. This is actually not used. Default: [0, 0].
        conv_type: The convolutional block before residual connection.
        init_method: initialization method of the weight parameters used to train large scale models.
            Choices: n, normal -- Swin V1 init method.
                    l, layernorm -- Swin V2 init method. Zero the weight and bias in the post layer normalization layer.
                    r, res_rescale -- EDSR rescale method. Rescale the residual blocks with a scaling factor 0.1
                    w, weight_rescale -- MSRResNet rescale method. Rescale the weight parameter in residual blocks with a scaling factor 0.1
                    t, trunc_normal_ -- nn.Linear, trunc_normal; nn.Conv2d, weight_rescale
        fairscale_checkpoint (bool): Whether to use fairscale checkpoint.
        offload_to_cpu (bool): used by fairscale_checkpoint
        args:
            out_proj_type (str): Type of the output projection in the self-attention modules. Default: linear. Choices: linear, conv2d.
            local_connection (bool): Whether to enable the local modelling module (two convs followed by Channel attention). For GRL base model, this is used.                "local_connection": local_connection,
            euclidean_dist (bool): use Euclidean distance or inner product as the similarity metric. An ablation study.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads_window,
        num_heads_stripe,
        window_size,
        stripe_size,
        stripe_groups,
        stripe_shift,
        mlp_ratio=4.0,
        qkv_bias=True,
        qkv_proj_type="linear",
        anchor_proj_type="avgpool",
        anchor_one_stage=True,
        anchor_window_down_factor=1,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=[0, 0],
        pretrained_stripe_size=[0, 0],
        conv_type="1conv",
        init_method="",
        fairscale_checkpoint=False,
        offload_to_cpu=False,
        args=None,
    ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.init_method = init_method

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = EfficientMixAttnTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads_w=num_heads_window,
                num_heads_s=num_heads_stripe,
                window_size=window_size,
                window_shift=i % 2 == 0,
                stripe_size=stripe_size,
                stripe_groups=stripe_groups,
                stripe_type="H" if i % 2 == 0 else "W",
                stripe_shift=i % 4 in [2, 3] if stripe_shift else False,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qkv_proj_type=qkv_proj_type,
                anchor_proj_type=anchor_proj_type,
                anchor_one_stage=anchor_one_stage,
                anchor_window_down_factor=anchor_window_down_factor,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
                pretrained_stripe_size=pretrained_stripe_size,
                res_scale=0.1 if init_method == "r" else 1.0,
                args=args,
            )
            self.blocks.append(block)

        self.conv = build_last_conv(conv_type, dim)

    def _init_weights(self):
        for n, m in self.named_modules():
            if self.init_method == "w":
                if isinstance(m, (nn.Linear, nn.Conv2d)) and n.find("cpb_mlp") < 0:
                    print("nn.Linear and nn.Conv2d weight initilization")
                    m.weight.data *= 0.1
            elif self.init_method == "l":
                if isinstance(m, nn.LayerNorm):
                    print("nn.LayerNorm initialization")
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 0)
            elif self.init_method.find("t") >= 0:
                scale = 0.1 ** (len(self.init_method) - 1) * int(self.init_method[-1])
                if isinstance(m, nn.Linear) and n.find("cpb_mlp") < 0:
                    trunc_normal_(m.weight, std=scale)
                elif isinstance(m, nn.Conv2d):
                    m.weight.data *= 0.1
                print(
                    "Initialization nn.Linear - trunc_normal; nn.Conv2d - weight rescale."
                )
            else:
                raise NotImplementedError(
                    f"Parameter initialization method {self.init_method} not implemented in TransformerStage."
                )

    def forward(self, x, x_size, table_index_mask):
        res = x
        for blk in self.blocks:
            res = blk(res, x_size, table_index_mask)
        res = bchw_to_blc(self.conv(blc_to_bchw(res, x_size)))

        return res + x

    def flops(self):
        pass


class GRL(nn.Module):
    r"""Image restoration transformer with global, non-local, and local connections
    Args:
        img_size (int | list[int]): Input image size. Default 64
        in_channels (int): Number of input image channels. Default: 3
        out_channels (int): Number of output image channels. Default: None
        embed_dim (int): Patch embedding dimension. Default: 96
        upscale (int): Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range (float): Image range. 1. or 255.
        upsampler (str): The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        depths (list[int]): Depth of each Swin Transformer layer.
        num_heads_window (list[int]): Number of window attention heads in different layers.
        num_heads_stripe (list[int]): Number of stripe attention heads in different layers.
        window_size (int): Window size. Default: 8.
        stripe_size (list[int]): Stripe size. Default: [8, 8]
        stripe_groups (list[int]): Number of stripe groups. Default: [None, None].
        stripe_shift (bool): whether to shift the stripes. This is used as an ablation study.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qkv_proj_type (str): QKV projection type. Default: linear. Choices: linear, separable_conv.
        anchor_proj_type (str): Anchor projection type. Default: avgpool. Choices: avgpool, maxpool, conv2d, separable_conv, patchmerging.
        anchor_one_stage (bool): Whether to use one operator or multiple progressive operators to reduce feature map resolution. Default: True.
        anchor_window_down_factor (int): The downscale factor used to get the anchors.
        out_proj_type (str): Type of the output projection in the self-attention modules. Default: linear. Choices: linear, conv2d.
        local_connection (bool): Whether to enable the local modelling module (two convs followed by Channel attention). For GRL base model, this is used.
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        pretrained_window_size (list[int]): pretrained window size. This is actually not used. Default: [0, 0].
        pretrained_stripe_size (list[int]): pretrained stripe size. This is actually not used. Default: [0, 0].
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        conv_type (str): The convolutional block before residual connection. Default: 1conv. Choices: 1conv, 3conv, 1conv1x1, linear
        init_method: initialization method of the weight parameters used to train large scale models.
            Choices: n, normal -- Swin V1 init method.
                    l, layernorm -- Swin V2 init method. Zero the weight and bias in the post layer normalization layer.
                    r, res_rescale -- EDSR rescale method. Rescale the residual blocks with a scaling factor 0.1
                    w, weight_rescale -- MSRResNet rescale method. Rescale the weight parameter in residual blocks with a scaling factor 0.1
                    t, trunc_normal_ -- nn.Linear, trunc_normal; nn.Conv2d, weight_rescale
        fairscale_checkpoint (bool): Whether to use fairscale checkpoint.
        offload_to_cpu (bool): used by fairscale_checkpoint
        euclidean_dist (bool): use Euclidean distance or inner product as the similarity metric. An ablation study.

    """

    def __init__(
        self,
        img_size=64,
        in_channels=3,
        out_channels=None,
        embed_dim=96,
        upscale=2,
        img_range=1.0,
        upsampler="",
        depths=[6, 6, 6, 6, 6, 6],
        num_heads_window=[3, 3, 3, 3, 3, 3],
        num_heads_stripe=[3, 3, 3, 3, 3, 3],
        window_size=8,
        stripe_size=[8, 8],  # used for stripe window attention
        stripe_groups=[None, None],
        stripe_shift=False,
        mlp_ratio=4.0,
        qkv_bias=True,
        qkv_proj_type="linear",
        anchor_proj_type="avgpool",
        anchor_one_stage=True,
        anchor_window_down_factor=1,
        out_proj_type="linear",
        local_connection=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=[0, 0],
        pretrained_stripe_size=[0, 0],
        conv_type="1conv",
        init_method="n",  # initialization method of the weight parameters used to train large scale models.
        fairscale_checkpoint=False,  # fairscale activation checkpointing
        offload_to_cpu=False,
        euclidean_dist=False,
        **kwargs,
    ):
        super(GRL, self).__init__()
        # Process the input arguments
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        num_out_feats = 64
        self.embed_dim = embed_dim
        self.upscale = upscale
        self.upsampler = upsampler
        self.img_range = img_range
        if in_channels == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        max_stripe_size = max([0 if s is None else s for s in stripe_size])
        max_stripe_groups = max([0 if s is None else s for s in stripe_groups])
        max_stripe_groups *= anchor_window_down_factor
        self.pad_size = max(window_size, max_stripe_size, max_stripe_groups)

        self.input_resolution = to_2tuple(img_size)
        self.window_size = to_2tuple(window_size)
        self.shift_size = [w // 2 for w in self.window_size]
        self.stripe_size = stripe_size
        self.stripe_groups = stripe_groups
        self.pretrained_window_size = pretrained_window_size
        self.pretrained_stripe_size = pretrained_stripe_size
        self.anchor_window_down_factor = anchor_window_down_factor

        # Head of the network. First convolution.
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)

        # Body of the network
        self.norm_start = norm_layer(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # stochastic depth decay rule
        args = OmegaConf.create(
            {
                "out_proj_type": out_proj_type,
                "local_connection": local_connection,
                "euclidean_dist": euclidean_dist,
            }
        )
        for k, v in self.set_table_index_mask(self.input_resolution).items():
            self.register_buffer(k, v)

        self.layers = nn.ModuleList()
        for i in range(len(depths)):
            layer = TransformerStage(
                dim=embed_dim,
                input_resolution=self.input_resolution,
                depth=depths[i],
                num_heads_window=num_heads_window[i],
                num_heads_stripe=num_heads_stripe[i],
                window_size=self.window_size,
                stripe_size=stripe_size,
                stripe_groups=stripe_groups,
                stripe_shift=stripe_shift,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qkv_proj_type=qkv_proj_type,
                anchor_proj_type=anchor_proj_type,
                anchor_one_stage=anchor_one_stage,
                anchor_window_down_factor=anchor_window_down_factor,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i]) : sum(depths[: i + 1])
                ],  # no impact on SR results
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
                pretrained_stripe_size=pretrained_stripe_size,
                conv_type=conv_type,
                init_method=init_method,
                fairscale_checkpoint=fairscale_checkpoint,
                offload_to_cpu=offload_to_cpu,
                args=args,
            )
            self.layers.append(layer)
        self.norm_end = norm_layer(embed_dim)

        # Tail of the network
        self.conv_after_body = build_last_conv(conv_type, embed_dim)

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == "pixelshuffle":
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_out_feats, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_out_feats)
            self.conv_last = nn.Conv2d(num_out_feats, out_channels, 3, 1, 1)
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(
                upscale,
                embed_dim,
                out_channels,
            )
        elif self.upsampler == "nearest+conv":
            # for real-world SR (less artifacts)
            assert self.upscale == 4, "only support x4 now."
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_out_feats, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.conv_up1 = nn.Conv2d(num_out_feats, num_out_feats, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_out_feats, num_out_feats, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_out_feats, num_out_feats, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_out_feats, out_channels, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, out_channels, 3, 1, 1)

        self.apply(self._init_weights)
        if init_method in ["l", "w"] or init_method.find("t") >= 0:
            for layer in self.layers:
                layer._init_weights()

    def set_table_index_mask(self, x_size):
        """
        Two used cases:
        1) At initialization: set the shared buffers.
        2) During forward pass: get the new buffers if the resolution of the input changes
        """
        # ss - stripe_size, sss - stripe_shift_size
        ss, sss = _get_stripe_info(self.stripe_size, self.stripe_groups, True, x_size)
        df = self.anchor_window_down_factor

        table_w = get_relative_coords_table_all(
            self.window_size, self.pretrained_window_size
        )
        table_sh = get_relative_coords_table_all(ss, self.pretrained_stripe_size, df)
        table_sv = get_relative_coords_table_all(
            ss[::-1], self.pretrained_stripe_size, df
        )

        index_w = get_relative_position_index_simple(self.window_size)
        index_sh_a2w = get_relative_position_index_simple(ss, df, False)
        index_sh_w2a = get_relative_position_index_simple(ss, df, True)
        index_sv_a2w = get_relative_position_index_simple(ss[::-1], df, False)
        index_sv_w2a = get_relative_position_index_simple(ss[::-1], df, True)

        mask_w = calculate_mask(x_size, self.window_size, self.shift_size)
        mask_sh_a2w = calculate_mask_all(x_size, ss, sss, df, False)
        mask_sh_w2a = calculate_mask_all(x_size, ss, sss, df, True)
        mask_sv_a2w = calculate_mask_all(x_size, ss[::-1], sss[::-1], df, False)
        mask_sv_w2a = calculate_mask_all(x_size, ss[::-1], sss[::-1], df, True)
        return {
            "table_w": table_w,
            "table_sh": table_sh,
            "table_sv": table_sv,
            "index_w": index_w,
            "index_sh_a2w": index_sh_a2w,
            "index_sh_w2a": index_sh_w2a,
            "index_sv_a2w": index_sv_a2w,
            "index_sv_w2a": index_sv_w2a,
            "mask_w": mask_w,
            "mask_sh_a2w": mask_sh_a2w,
            "mask_sh_w2a": mask_sh_w2a,
            "mask_sv_a2w": mask_sv_a2w,
            "mask_sv_w2a": mask_sv_w2a,
        }

    def get_table_index_mask(self, device=None, input_resolution=None):
        # Used during forward pass
        if input_resolution == self.input_resolution:
            return {
                "table_w": self.table_w,
                "table_sh": self.table_sh,
                "table_sv": self.table_sv,
                "index_w": self.index_w,
                "index_sh_a2w": self.index_sh_a2w,
                "index_sh_w2a": self.index_sh_w2a,
                "index_sv_a2w": self.index_sv_a2w,
                "index_sv_w2a": self.index_sv_w2a,
                "mask_w": self.mask_w,
                "mask_sh_a2w": self.mask_sh_a2w,
                "mask_sh_w2a": self.mask_sh_w2a,
                "mask_sv_a2w": self.mask_sv_a2w,
                "mask_sv_w2a": self.mask_sv_w2a,
            }
        else:
            table_index_mask = self.set_table_index_mask(input_resolution)
            for k, v in table_index_mask.items():
                table_index_mask[k] = v.to(device)
            return table_index_mask

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.pad_size - h % self.pad_size) % self.pad_size
        mod_pad_w = (self.pad_size - w % self.pad_size) % self.pad_size

        try:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        except BaseException:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "constant")
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = bchw_to_blc(x)
        x = self.norm_start(x)
        x = self.pos_drop(x)

        table_index_mask = self.get_table_index_mask(x.device, x_size)
        for layer in self.layers:
            x = layer(x, x_size, table_index_mask)

        x = self.norm_end(x)  # B L C
        x = blc_to_bchw(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == "pixelshuffle":
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == "nearest+conv":
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(
                self.conv_up1(
                    torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
                )
            )
            x = self.lrelu(
                self.conv_up2(
                    torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
                )
            )
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            if self.in_channels == self.out_channels:
                x = x + self.conv_last(res)
            else:
                x = self.conv_last(res)

        x = x / self.img_range + self.mean

        return x[:, :, : H * self.upscale, : W * self.upscale]

    def flops(self):
        pass

    def convert_checkpoint(self, state_dict):
        for k in list(state_dict.keys()):
            if (
                k.find("relative_coords_table") >= 0
                or k.find("relative_position_index") >= 0
                or k.find("attn_mask") >= 0
                or k.find("model.table_") >= 0
                or k.find("model.index_") >= 0
                or k.find("model.mask_") >= 0
            ):
                state_dict.pop(k)
                print(k)
        return state_dict
