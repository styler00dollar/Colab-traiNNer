"""
lvvit.py (07-06-21)
https://github.com/zihangJiang/TokenLabeling/blob/4a734ea96a3b36daebf1446c1cb2e0010a8e5f48/models/lvvit.py

layers.py (07-06-21)
https://github.com/zihangJiang/TokenLabeling/blob/main/models/layers.py
"""

import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath, to_2tuple

DROPOUT_FLOPS = 4
LAYER_NORM_FLOPS = 5
ACTIVATION_FLOPS = 8
SOFTMAX_FLOPS = 5


class GroupLinear(nn.Module):
    """
    Group Linear operator
    """

    def __init__(self, in_planes, out_channels, groups=1, bias=True):
        super(GroupLinear, self).__init__()
        assert in_planes % groups == 0
        assert out_channels % groups == 0
        self.in_dim = in_planes
        self.out_dim = out_channels
        self.groups = groups
        self.bias = bias
        self.group_in_dim = int(self.in_dim / self.groups)
        self.group_out_dim = int(self.out_dim / self.groups)

        self.group_weight = nn.Parameter(
            torch.zeros(self.groups, self.group_in_dim, self.group_out_dim)
        )
        self.group_bias = nn.Parameter(torch.zeros(self.out_dim))

    def forward(self, x):
        t, b, d = x.size()
        x = x.view(t, b, self.groups, int(d / self.groups))
        out = (
            torch.einsum("tbgd,gdf->tbgf", (x, self.group_weight)).reshape(
                t, b, self.out_dim
            )
            + self.group_bias
        )
        return out

    def extra_repr(self):
        s = "{in_dim}, {out_dim}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)


class Mlp(nn.Module):
    """
    MLP with support to use group linear operator
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        group=1,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if group == 1:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
        else:
            self.fc1 = GroupLinear(in_features, hidden_features, group)
            self.fc2 = GroupLinear(hidden_features, out_features, group)
        self.act = act_layer()

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GroupNorm(nn.Module):
    def __init__(self, num_groups, embed_dim, eps=1e-5, affine=True):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, embed_dim, eps, affine)

    def forward(self, x):
        B, T, C = x.shape
        x = x.view(B * T, C)
        x = self.gn(x)
        x = x.view(B, T, C)
        return x


class Attention(nn.Module):
    """
    Multi-head self-attention
    from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with some modification to support different num_heads and head_dim.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        head_dim=None,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            head_dim = dim // num_heads
            self.head_dim = head_dim
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, self.head_dim * self.num_heads * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, padding_mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        # B,heads,N,C/heads
        q, k, v = qkv[0], qkv[1], qkv[2]

        # trick here to make q@k.t more stable
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if padding_mask is not None:
            attn = attn.view(B, self.num_heads, N, N)
            attn = attn.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_float = attn.softmax(dim=-1, dtype=torch.float32)
            attn = attn_float.type_as(attn)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.head_dim * self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """
    Pre-layernorm transformer block
    """

    def __init__(
        self,
        dim,
        num_heads,
        head_dim=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        group=1,
        skip_lam=1.0,
    ):
        super().__init__()
        self.dim = dim
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.skip_lam = skip_lam

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            head_dim=head_dim,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=self.mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            group=group,
        )

    def forward(self, x, padding_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), padding_mask)) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x

    def flops(self, s):
        heads = self.attn.num_heads
        h = self.dim
        i = self.mlp_hidden_dim
        mha_block_flops = dict(
            kqv=3 * h * h,
            attention_scores=h * s,
            attn_softmax=SOFTMAX_FLOPS * s * heads,
            attention_dropout=DROPOUT_FLOPS * s * heads,
            attention_scale=s * heads,
            attention_weighted_avg_values=h * s,
            attn_output=h * h,
            attn_output_bias=h,
            attn_output_dropout=DROPOUT_FLOPS * h,
            attn_output_residual=h,
            attn_output_layer_norm=LAYER_NORM_FLOPS * h,
        )
        ffn_block_flops = dict(
            intermediate=h * i,
            intermediate_act=ACTIVATION_FLOPS * i,
            intermediate_bias=i,
            output=h * i,
            output_bias=h,
            output_dropout=DROPOUT_FLOPS * h,
            output_residual=h,
            output_layer_norm=LAYER_NORM_FLOPS * h,
        )

        return sum(mha_block_flops.values()) * s + sum(ffn_block_flops.values()) * s


class MHABlock(nn.Module):
    """
    Multihead Attention block with residual branch
    """

    def __init__(
        self,
        dim,
        num_heads,
        head_dim=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        group=1,
        skip_lam=1.0,
    ):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.skip_lam = skip_lam
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            head_dim=head_dim,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, padding_mask=None):
        x = (
            x
            + self.drop_path(self.attn(self.norm1(x * self.skip_lam), padding_mask))
            / self.skip_lam
        )
        return x

    def flops(self, s):
        heads = self.attn.num_heads
        h = self.dim
        block_flops = dict(
            kqv=3 * h * h,
            attention_scores=h * s,
            attn_softmax=SOFTMAX_FLOPS * s * heads,
            attention_dropout=DROPOUT_FLOPS * s * heads,
            attention_scale=s * heads,
            attention_weighted_avg_values=h * s,
            attn_output=h * h,
            attn_output_bias=h,
            attn_output_dropout=DROPOUT_FLOPS * h,
            attn_output_residual=h,
            attn_output_layer_norm=LAYER_NORM_FLOPS * h,
        )

        return sum(block_flops.values()) * s


class FFNBlock(nn.Module):
    """
    Feed forward network with residual branch
    """

    def __init__(
        self,
        dim,
        num_heads,
        head_dim=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        group=1,
        skip_lam=1.0,
    ):
        super().__init__()
        self.skip_lam = skip_lam
        self.dim = dim
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=self.mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            group=group,
        )

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(x * self.skip_lam))) / self.skip_lam
        return x

    def flops(self, s):
        h = self.dim
        i = self.mlp_hidden_dim
        block_flops = dict(
            intermediate=h * i,
            intermediate_act=ACTIVATION_FLOPS * i,
            intermediate_bias=i,
            output=h * i,
            output_bias=h,
            output_dropout=DROPOUT_FLOPS * h,
            output_residual=h,
            output_layer_norm=LAYER_NORM_FLOPS * h,
        )

        return sum(block_flops.values()) * s


class HybridEmbed(nn.Module):
    """CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(
        self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768
    ):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[
                    -1
                ]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = self.proj(x)
        return x


class PatchEmbedNaive(nn.Module):
    """
    Image to Patch Embedding
    from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        return x

    def flops(self):
        img_size = self.img_size[0]
        block_flops = dict(
            proj=img_size * img_size * 3 * self.embed_dim,
        )
        return sum(block_flops.values())


class PatchEmbed4_2(nn.Module):
    """
    Image to Patch Embedding with 4 layer convolution
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        new_patch_size = to_2tuple(patch_size // 2)

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(
            in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False
        )  # 112x112
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, bias=False
        )  # 112x112
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.proj = nn.Conv2d(
            64, embed_dim, kernel_size=new_patch_size, stride=new_patch_size
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.proj(x)  # [B, C, W, H]

        return x

    def flops(self):
        img_size = self.img_size[0]
        block_flops = dict(
            conv1=img_size / 2 * img_size / 2 * 3 * 64 * 7 * 7,
            conv2=img_size / 2 * img_size / 2 * 64 * 64 * 3 * 3,
            conv3=img_size / 2 * img_size / 2 * 64 * 64 * 3 * 3,
            proj=img_size / 2 * img_size / 2 * 64 * self.embed_dim,
        )
        return sum(block_flops.values())


class PatchEmbed4_2_128(nn.Module):
    """
    Image to Patch Embedding with 4 layer convolution and 128 filters
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        new_patch_size = to_2tuple(patch_size // 2)

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(
            in_chans, 128, kernel_size=7, stride=2, padding=3, bias=False
        )  # 112x112
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=False
        )  # 112x112
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.proj = nn.Conv2d(
            128, embed_dim, kernel_size=new_patch_size, stride=new_patch_size
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.proj(x)  # [B, C, W, H]

        return x

    def flops(self):
        img_size = self.img_size[0]
        block_flops = dict(
            conv1=img_size / 2 * img_size / 2 * 3 * 128 * 7 * 7,
            conv2=img_size / 2 * img_size / 2 * 128 * 128 * 3 * 3,
            conv3=img_size / 2 * img_size / 2 * 128 * 128 * 3 * 3,
            proj=img_size / 2 * img_size / 2 * 128 * self.embed_dim,
        )
        return sum(block_flops.values())


import torch.nn as nn

from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

# from .layers import *


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "LV_ViT_Tiny": _cfg(),
    "LV_ViT": _cfg(),
    "LV_ViT_Medium": _cfg(crop_pct=1.0),
    "LV_ViT_Large": _cfg(crop_pct=1.0),
}


def get_block(block_type, **kargs):
    if block_type == "mha":
        # multi-head attention block
        return MHABlock(**kargs)
    elif block_type == "ffn":
        # feed forward block
        return FFNBlock(**kargs)
    elif block_type == "tr":
        # transformer block
        return Block(**kargs)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_dpr(drop_path_rate, depth, drop_path_decay="linear"):
    if drop_path_decay == "linear":
        # linear dpr decay
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
    elif drop_path_decay == "fix":
        # use fixed dpr
        dpr = [drop_path_rate] * depth
    else:
        # use predefined drop_path_rate list
        assert len(drop_path_rate) == depth
        dpr = drop_path_rate
    return dpr


class LV_ViT(nn.Module):
    """Vision Transformer with tricks
    Arguements:
        p_emb: different conv based position embedding (default: 4 layer conv)
        skip_lam: residual scalar for skip connection (default: 1.0)
        order: which order of layers will be used (default: None, will override depth if given)
        mix_token: use mix token augmentation for batch of tokens (default: False)
        return_dense: whether to return feature of all tokens with an additional aux_head (default: False)
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        drop_path_decay="linear",
        hybrid_backbone=None,
        norm_layer=nn.LayerNorm,
        p_emb="4_2",
        head_dim=None,
        skip_lam=1.0,
        order=None,
        mix_token=False,
        return_dense=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.output_dim = embed_dim if num_classes == 0 else num_classes
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone,
                img_size=img_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:
            if p_emb == "4_2":
                patch_embed_fn = PatchEmbed4_2
            elif p_emb == "4_2_128":
                patch_embed_fn = PatchEmbed4_2_128
            else:
                patch_embed_fn = PatchEmbedNaive

            self.patch_embed = patch_embed_fn(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if order is None:
            dpr = get_dpr(drop_path_rate, depth, drop_path_decay)
            self.blocks = nn.ModuleList(
                [
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                        skip_lam=skip_lam,
                    )
                    for i in range(depth)
                ]
            )
        else:
            # use given order to sequentially generate modules
            dpr = get_dpr(drop_path_rate, len(order), drop_path_decay)
            self.blocks = nn.ModuleList(
                [
                    get_block(
                        order[i],
                        dim=embed_dim,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                        skip_lam=skip_lam,
                    )
                    for i in range(len(order))
                ]
            )

        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        self.return_dense = return_dense
        self.mix_token = mix_token

        if return_dense:
            self.aux_head = (
                nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            )
        if mix_token:
            self.beta = 1.0
            assert return_dense, "always return all features when mixtoken is enabled"

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, GroupLinear):
            trunc_normal_(m.group_weight, std=0.02)
            if isinstance(m, GroupLinear) and m.group_bias is not None:
                nn.init.constant_(m.group_bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward_features(self, x):
        # simple forward to obtain feature map (without mixtoken)
        x = self.forward_embeddings(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.forward_tokens(x)
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)

        # token level mixtoken augmentation
        if self.mix_token and self.training:
            lam = np.random.beta(self.beta, self.beta)
            patch_h, patch_w = x.shape[2], x.shape[3]
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
            temp_x = x.clone()
            temp_x[:, :, bbx1:bbx2, bby1:bby2] = x.flip(0)[:, :, bbx1:bbx2, bby1:bby2]
            x = temp_x
        else:
            bbx1, bby1, bbx2, bby2 = 0, 0, 0, 0

        x = x.flatten(2).transpose(1, 2)
        x = self.forward_tokens(x)
        x_cls = self.head(x[:, 0])

        if self.return_dense:
            x_aux = self.aux_head(x[:, 1:])
            if not self.training:
                return x_cls + 0.5 * x_aux.max(1)[0]

            # recover the mixed part
            if self.mix_token and self.training:
                x_aux = x_aux.reshape(x_aux.shape[0], patch_h, patch_w, x_aux.shape[-1])
                temp_x = x_aux.clone()
                temp_x[:, bbx1:bbx2, bby1:bby2, :] = x_aux.flip(0)[
                    :, bbx1:bbx2, bby1:bby2, :
                ]
                x_aux = temp_x
                x_aux = x_aux.reshape(
                    x_aux.shape[0], patch_h * patch_w, x_aux.shape[-1]
                )

            return x_cls, x_aux, (bbx1, bby1, bbx2, bby2)
        return x_cls


@register_model
def vit(pretrained=False, **kwargs):
    model = LV_ViT(
        patch_size=16,
        embed_dim=384,
        depth=16,
        num_heads=6,
        mlp_ratio=3.0,
        p_emb=1,
        **kwargs,
    )
    model.default_cfg = default_cfgs["LV_ViT"]
    return model


@register_model
def lvvit(pretrained=False, **kwargs):
    model = LV_ViT(
        patch_size=16,
        embed_dim=384,
        depth=16,
        num_heads=6,
        mlp_ratio=3.0,
        p_emb="4_2",
        skip_lam=2.0,
        **kwargs,
    )
    model.default_cfg = default_cfgs["LV_ViT"]
    return model


@register_model
def lvvit_s(pretrained=False, **kwargs):
    model = LV_ViT(
        patch_size=16,
        embed_dim=384,
        depth=16,
        num_heads=6,
        mlp_ratio=3.0,
        p_emb="4_2",
        skip_lam=2.0,
        return_dense=True,
        mix_token=True,
        **kwargs,
    )
    model.default_cfg = default_cfgs["LV_ViT"]
    return model


@register_model
def lvvit_m(pretrained=False, **kwargs):
    model = LV_ViT(
        patch_size=16,
        embed_dim=512,
        depth=20,
        num_heads=8,
        mlp_ratio=3.0,
        p_emb="4_2",
        skip_lam=2.0,
        return_dense=True,
        mix_token=True,
        **kwargs,
    )
    model.default_cfg = default_cfgs["LV_ViT_Medium"]
    return model


@register_model
def lvvit_l(pretrained=False, **kwargs):
    order = ["tr"] * 24  # this will override depth, can also be set as None
    model = LV_ViT(
        patch_size=16,
        embed_dim=768,
        depth=24,
        num_heads=12,
        mlp_ratio=3.0,
        p_emb="4_2_128",
        skip_lam=3.0,
        return_dense=True,
        mix_token=True,
        order=order,
        **kwargs,
    )
    model.default_cfg = default_cfgs["LV_ViT_Large"]
    return model
