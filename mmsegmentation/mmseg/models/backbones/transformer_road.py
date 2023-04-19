import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmcv.runner import BaseModule, _load_checkpoint
from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES
from torch.nn.init import trunc_normal_
import numpy as np


class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def attention_pool(tensor, pool, hw_shape, norm=None):
    if pool is None:
        return tensor, hw_shape
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")


    B, N, L, C = tensor.shape
    H, W = hw_shape
    tensor = tensor.reshape(B * N, H, W, C).permute(0, 3, 1, 2).contiguous()

    tensor = pool(tensor)

    hw_shape = [tensor.shape[2], tensor.shape[3]]
    L_pooled = tensor.shape[2] * tensor.shape[3]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)

    if norm is not None:
        tensor = norm(tensor)

    if tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor, hw_shape


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=12,
                 sr_ratio=1,
                 kernel_q=(3, 3),
                 kernel_kv=(3, 3),
                 stride_q=(1, 1),
                 stride_kv=(2, 2),
                 norm_layer=nn.LayerNorm,
                 has_cls_embed=False,
                 ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio + 1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim)
            self.sr_norm = nn.LayerNorm(dim, eps=1e-6)

        self.up = nn.Sequential(
            nn.Conv2d(dim, sr_ratio * sr_ratio * dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.PixelShuffle(upscale_factor=sr_ratio)
        )
        self.up_norm = nn.LayerNorm(dim, eps=1e-6)

        self.proj = nn.Linear(dim, dim)

        # mvit v2
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]
        self.has_cls_embed = has_cls_embed
        dim_conv = dim // num_heads
        self.pool_q = (
            nn.Conv2d(
                dim_conv,
                dim_conv,
                kernel_q,
                stride=stride_q,
                padding=padding_q,
                groups=dim_conv,
                bias=False,
            )
            if len(kernel_q) > 0
            else None
        )
        self.norm_q = norm_layer(dim_conv) if len(kernel_q) > 0 else None
        self.pool_k = (
            nn.Conv2d(
                dim_conv,
                dim_conv,
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
                groups=dim_conv,
                bias=False,
            )
            if len(kernel_kv) > 0
            else None
        )
        self.norm_k = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
        self.pool_v = (
            nn.Conv2d(
                dim_conv,
                dim_conv,
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
                groups=dim_conv,
                bias=False,
            )
            if len(kernel_kv) > 0
            else None
        )
        self.norm_v = norm_layer(dim_conv) if len(kernel_kv) > 0 else None

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        hw_shape = [H, W]
        q, q_shape = attention_pool(
            q,
            self.pool_q,
            hw_shape,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
        )
        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.sr_norm(x)
            hw_shape = [H // self.sr_ratio, W // self.sr_ratio]

        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        k, k_shape = attention_pool(
            k,
            self.pool_k,
            hw_shape,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
        )
        v_ = v
        v, v_shape = attention_pool(
            v,
            self.pool_v,
            hw_shape,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        identity = v_.transpose(-1, -2).reshape(B, C, H // self.sr_ratio, W // self.sr_ratio)
        identity = self.up(identity).flatten(2).transpose(1, 2)
        x = self.proj(x + self.up_norm(identity))
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio=1, drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, sr_ratio)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # mvitv2
        self.stride_q = (1,1)
        if len(self.stride_q) > 0 and np.prod(self.stride_q) > 1:
            kernel_skip = [s + 1 if s > 1 else s for s in self.stride_q]
            stride_skip = self.stride_q
            padding_skip = [int(skip // 2) for skip in kernel_skip]
            self.pool_skip = nn.MaxPool2d(kernel_skip, stride_skip, padding_skip, ceil_mode=False)
            self.pool_norm = nn.Linear(dim, dim * 4, bias=True)
        else:
            stride_q = self.stride_q[0]
            kernel_skip = stride_q + 2
            padding_skip = int(kernel_skip // 2)
            self.pool_skip = nn.MaxPool2d(kernel_skip, stride_q, padding_skip, ceil_mode=False)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x_res = x.permute(0, 2, 1).reshape(B, C, H, W)
        x_res = self.pool_skip(x_res).reshape(B, C, -1).permute(0, 2, 1)

        x = x_res + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))


class ConvStem(nn.Module):
    def __init__(self, in_ch=3, out_ch=96, patch_size=2, with_pos=True):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        stem = []
        in_dim, out_dim = in_ch, out_ch // 2
        for i in range(2):
            stem.append(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(out_dim))
            stem.append(nn.ReLU(inplace=True))
            in_dim, out_dim = out_dim, out_dim * 2

        stem.append(nn.Conv2d(in_dim, out_ch, kernel_size=1, stride=1))
        self.proj = nn.Sequential(*stem)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

        self.norm = nn.LayerNorm(out_ch, eps=1e-6)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.with_pos:
            x = self.pos(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, out_ch=96, patch_size=2, with_pos=True):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=patch_size + 1, stride=patch_size, padding=patch_size // 2)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

        self.norm = nn.LayerNorm(out_ch, eps=1e-6)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.with_pos:
            x = self.pos(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


@BACKBONES.register_module()
class ResMCS(BaseModule):
    def __init__(self, in_chans=3, embed_dims=[96, 192, 384, 768],
                 num_heads=[1, 2, 4, 8], drop_path_rate=0.,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], out_indices=(0, 1, 2, 3)):
        super().__init__()
        """
        The code is written based on ResTv2(https://github.com/wofmanaf/ResT)
        and Mvit(https://github.com/facebookresearch/mvit).
        """
        self.depths = depths
        self.out_indices = out_indices

        self.stem = ConvStem(in_chans, embed_dims[0], patch_size=4)
        self.patch_2 = PatchEmbed(embed_dims[0], embed_dims[1], patch_size=2)
        self.patch_3 = PatchEmbed(embed_dims[1], embed_dims[2], patch_size=2)
        self.patch_4 = PatchEmbed(embed_dims[2], embed_dims[3], patch_size=2)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stage1 = nn.ModuleList([
            Block(embed_dims[0], num_heads[0], sr_ratios[0], dpr[cur + i])
            for i in range(depths[0])
        ])

        cur += depths[0]
        self.stage2 = nn.ModuleList([
            Block(embed_dims[1], num_heads[1], sr_ratios[1], dpr[cur + i])
            for i in range(depths[1])
        ])

        cur += depths[1]
        self.stage3 = nn.ModuleList([
            Block(embed_dims[2], num_heads[2], sr_ratios[2], dpr[cur + i])
            for i in range(depths[2])
        ])

        cur += depths[2]
        self.stage4 = nn.ModuleList([
            Block(embed_dims[3], num_heads[3], sr_ratios[3], dpr[cur + i])
            for i in range(depths[3])
        ])

        for idx in out_indices:
            out_ch = embed_dims[idx]
            layer = LayerNorm(out_ch, eps=1e-6, data_format="channels_first")
            layer_name = f"norm_{idx + 1}"
            self.add_module(layer_name, layer)

    def forward(self, x):
        outs = []
        B, _, H, W = x.shape
        x, (H, W) = self.stem(x)
        # stage 1
        for blk in self.stage1:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        if 0 in self.out_indices:
            outs.append(self.norm_1(x))

        # stage 2
        x, (H, W) = self.patch_2(x)
        for blk in self.stage2:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        if 1 in self.out_indices:
            outs.append(self.norm_2(x))

        # stage 3
        x, (H, W) = self.patch_3(x)
        for blk in self.stage3:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        if 2 in self.out_indices:
            outs.append(self.norm_3(x))

        # stage 4
        x, (H, W) = self.patch_4(x)
        for blk in self.stage4:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        if 3 in self.out_indices:
            outs.append(self.norm_4(x))

        return tuple(outs)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.dim = (dim,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.dim, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
