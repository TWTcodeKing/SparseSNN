"""
QKFormer: Hierarchical Spiking Transformer using Q-K Attention (NeurIPS 2024)
Source: https://github.com/zhouchenlin2096/QKFormer

Key innovations:
- Token-level Q-K attention: sum over Q channels then multiply with K (linear complexity)
- Hierarchical 3-stage architecture: Q-K Attention (stage 1-2) + SSA (stage 3)
- Residual patch embedding with MaxPool downsampling

Original uses spikingjelly - replaced with standalone neurons.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .neurons import MultiStepLIFNeuron

__all__ = ['QKFormer', 'qkformer_10_384', 'qkformer_10_512', 'qkformer_10_768']


def _trunc_normal_(tensor, mean=0., std=.02):
    with torch.no_grad():
        tensor.normal_(mean, std)
        tensor.clamp_(-2 * std, 2 * std)
    return tensor


def _to_2tuple(x):
    return (x, x) if not isinstance(x, (list, tuple)) else x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)
        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        self.fc2_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)
        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()
        x = self.fc1_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()
        x = self.fc2_lif(x)
        return x


class TokenQKAttention(nn.Module):
    """Token-level Q-K Attention (linear complexity).

    Instead of full Q@K^T, sums Q over channel dimension then
    element-wise multiplies with K. O(N) vs O(N^2).
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)

        self.attn_lif = MultiStepLIFNeuron(tau=2.0, v_threshold=0.5, detach_reset=True)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        # Token Q-K attention: sum Q over channel dim, then multiply with K
        q = torch.sum(q, dim=3, keepdim=True)
        attn = self.attn_lif(q)
        x = torch.mul(attn, k)

        x = x.flatten(2, 3)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        x = self.proj_lif(x)
        return x


class SpikingSelfAttention(nn.Module):
    """Standard Spiking Self-Attention (SSA) for deeper stages."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)

        self.attn_lif = MultiStepLIFNeuron(tau=2.0, v_threshold=0.5, detach_reset=True)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous()

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale
        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T, B, C, H, W)
        return x


class TokenSpikingBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.tssa = TokenQKAttention(dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.tssa(x)
        x = x + self.mlp(x)
        return x


class SpikingBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.attn = SpikingSelfAttention(dim, num_heads=num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class PatchEmbedInit(nn.Module):
    """Initial patch embedding with residual downsampling."""

    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4,
                 in_channels=2, embed_dims=256):
        super().__init__()
        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 2)
        self.proj_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.proj_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)

        self.proj1_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims)
        self.proj1_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.proj1_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)

        self.proj2_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm2d(embed_dims)
        self.proj2_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)

        self.proj_res_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x)
        x = self.proj_maxpool(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj_lif(x).flatten(0, 1).contiguous()

        x_feat = x
        x = self.proj1_conv(x)
        x = self.proj1_bn(x)
        x = self.proj1_maxpool(x).reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = self.proj1_lif(x).flatten(0, 1).contiguous()

        x = self.proj2_conv(x)
        x = self.proj2_bn(x).reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = self.proj2_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H // 4, W // 4).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat  # residual shortcut
        return x


class PatchEmbedStage(nn.Module):
    """Stage-level patch embedding with residual downsampling."""

    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4,
                 in_channels=2, embed_dims=256):
        super().__init__()
        self.proj3_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.proj3_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.proj3_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)

        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        self.proj4_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)

        self.proj_res_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = x.flatten(0, 1).contiguous()
        x_feat = x

        x = self.proj3_conv(x)
        x = self.proj3_bn(x)
        x = self.proj3_maxpool(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj3_lif(x).flatten(0, 1).contiguous()

        x = self.proj4_conv(x)
        x = self.proj4_bn(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj4_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat  # residual shortcut
        return x


class QKFormer(nn.Module):
    """Hierarchical Spiking Transformer with Q-K Attention.

    3-stage architecture:
    - Stage 1 (low-res): 1x TokenQKAttention block
    - Stage 2 (mid-res): 2x TokenQKAttention blocks
    - Stage 3 (high-res): (depths-3)x SSA blocks
    """

    def __init__(self, T=4, img_size_h=128, img_size_w=128, patch_size=16,
                 in_channels=2, num_classes=11, embed_dims=64, num_heads=8,
                 mlp_ratios=4, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, depths=10, sr_ratios=1):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        self.patch_embed1 = PatchEmbedInit(
            img_size_h=img_size_h, img_size_w=img_size_w,
            patch_size=patch_size, in_channels=in_channels,
            embed_dims=embed_dims // 4
        )
        self.stage1 = nn.ModuleList([
            TokenSpikingBlock(dim=embed_dims // 4, num_heads=num_heads, mlp_ratio=mlp_ratios)
            for _ in range(1)
        ])

        self.patch_embed2 = PatchEmbedStage(
            img_size_h=img_size_h, img_size_w=img_size_w,
            patch_size=patch_size, in_channels=in_channels,
            embed_dims=embed_dims // 2
        )
        self.stage2 = nn.ModuleList([
            TokenSpikingBlock(dim=embed_dims // 2, num_heads=num_heads, mlp_ratio=mlp_ratios)
            for _ in range(2)
        ])

        self.patch_embed3 = PatchEmbedStage(
            img_size_h=img_size_h, img_size_w=img_size_w,
            patch_size=patch_size, in_channels=in_channels,
            embed_dims=embed_dims
        )
        self.stage3 = nn.ModuleList([
            SpikingBlock(dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios)
            for _ in range(depths - 3)
        ])

        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed1(x)
        for blk in self.stage1:
            x = blk(x)

        x = self.patch_embed2(x)
        for blk in self.stage2:
            x = blk(x)

        x = self.patch_embed3(x)
        for blk in self.stage3:
            x = blk(x)

        return x.flatten(3).mean(3)

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x


def qkformer_10_384(num_classes=1000, T=4, **kwargs):
    return QKFormer(
        T=T, img_size_h=224, img_size_w=224, patch_size=16,
        embed_dims=384, num_heads=6, mlp_ratios=4,
        in_channels=3, num_classes=num_classes,
        depths=10, sr_ratios=1, **kwargs
    )

def qkformer_10_512(num_classes=1000, T=4, **kwargs):
    return QKFormer(
        T=T, img_size_h=224, img_size_w=224, patch_size=16,
        embed_dims=512, num_heads=8, mlp_ratios=4,
        in_channels=3, num_classes=num_classes,
        depths=10, sr_ratios=1, **kwargs
    )

def qkformer_10_768(num_classes=1000, T=4, **kwargs):
    return QKFormer(
        T=T, img_size_h=224, img_size_w=224, patch_size=16,
        embed_dims=768, num_heads=12, mlp_ratios=4,
        in_channels=3, num_classes=num_classes,
        depths=10, sr_ratios=1, **kwargs
    )
