"""
MaxFormer: Max-Former (NeurIPS 2025)
Source: https://github.com/bic-L/MaxFormer

Key innovations:
- Hierarchical 3-stage architecture with various mixer types
- MaxPool-based token mixer in early stages (simple but effective)
- DWConv (depthwise conv) mixers of various kernel sizes
- SSA (Spiking Self-Attention) with optional DWConv enhancement in later stages
- Modular embedding hub with Max/Avg/Original embedding variants

Original uses spikingjelly - replaced with standalone neurons.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models import MultiStepLIFNeuron
from models import SeqToANNContainerT, SeqToANNContainer

__all__ = ['MaxFormer', 'maxformer_10_384', 'maxformer_10_512', 'maxformer_10_768']


def _trunc_normal_(tensor, mean=0., std=.02):
    with torch.no_grad():
        tensor.normal_(mean, std)
        tensor.clamp_(-2 * std, 2 * std)
    return tensor


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class S_MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.res = in_features == hidden_features
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_lif = MultiStepLIFNeuron(detach_reset=True)
        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        self.fc2_lif = MultiStepLIFNeuron(detach_reset=True)
        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        identity = x
        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()
        if self.res:
            x = identity + x
            identity = x
        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()
        x = x + identity
        return x


# ---------------------------------------------------------------------------
# Mixer Blocks
# ---------------------------------------------------------------------------

class Block_Max(nn.Module):
    """MaxPool mixer block."""
    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = S_MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.pool(x.flatten(0, 1).contiguous()).reshape(T, B, -1, H, W).contiguous()
        x = self.mlp(x)
        return x


class Block_DWC(nn.Module):
    """Depthwise convolution mixer block."""
    def __init__(self, dim, kernel_size=5, mlp_ratio=4.):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                              padding=kernel_size // 2, groups=dim)
        self.conv_bn = nn.BatchNorm2d(dim)
        self.conv_neuron = MultiStepLIFNeuron(tau=2.0, detach_reset=True)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = S_MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        identity = x
        x = self.conv_neuron(x).reshape(T * B, -1, H, W).contiguous()
        x = self.conv(x)
        x = self.conv_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = x + identity
        x = self.mlp(x)
        return x


class SSA(nn.Module):
    """Spiking Self-Attention."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.x_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)

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

    def forward(self, x):
        T, B, C, H, W = x.shape
        identity = x
        x = self.x_lif(x)
        x = x.flatten(3).contiguous()
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1).contiguous()

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
        x = self.proj_bn(self.proj_conv(x)).reshape(T, B, C, H, W)

        x = x + identity
        return x


class Block_SSA(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.):
        super().__init__()
        self.attn = SSA(dim, num_heads=num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = S_MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x


# ---------------------------------------------------------------------------
# Embedding Hub
# ---------------------------------------------------------------------------

class Embed(nn.Module):
    def __init__(self, in_channels=2, out_channels=256, kernel_size=3,
                 stride=1, padding=1, shortcut=False):
        super().__init__()
        self.embed_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)
        self.embed_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, bias=False)
        self.embed_bn = nn.BatchNorm2d(out_channels)
        self.shortcut = shortcut

    def forward(self, x, dual=False):
        if not self.shortcut:
            x = self.embed_lif(x)
        x_feat = x
        x = self.embed_conv(x.flatten(0, 1).contiguous())
        x = self.embed_bn(x)
        if dual:
            return x, x_feat
        return x


class MaxEmbed(nn.Module):
    def __init__(self, in_channels=2, out_channels=256, kernel_size=3,
                 stride=1, padding=1, shortcut=False):
        super().__init__()
        self.embed_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)
        self.embed_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, bias=False)
        self.embed_bn = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.shortcut = shortcut

    def forward(self, x, dual=False):
        if not self.shortcut:
            x = self.embed_lif(x)
        x_feat = x
        x = self.embed_conv(x.flatten(0, 1).contiguous())
        x = self.embed_bn(x)
        x = self.maxpool(x)
        if dual:
            return x, x_feat
        return x


class EmbedOrigImageNet(nn.Module):
    """Initial 4x downsampling embedding for ImageNet."""
    def __init__(self, in_channels=3, embed_dims=256):
        super().__init__()
        self.embed1 = Embed(in_channels=in_channels, out_channels=embed_dims // 2,
                            kernel_size=3, stride=2, padding=1, shortcut=True)
        self.embed2 = Embed(in_channels=embed_dims // 2, out_channels=embed_dims,
                            kernel_size=3, stride=2, padding=1)
        self.embed3 = Embed(in_channels=embed_dims, out_channels=embed_dims,
                            kernel_size=3, stride=1, padding=1)
        self.embed4 = Embed(in_channels=embed_dims // 2, out_channels=embed_dims,
                            kernel_size=1, stride=2, padding=0, shortcut=True)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.embed1(x)
        x = x.reshape(T, B, -1, H // 2, W // 2).contiguous()

        x, x_feat = self.embed2(x, dual=True)
        x = x.reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = self.embed3(x)

        x_feat = self.embed4(x_feat)  # shortcut
        x = (x + x_feat).reshape(T, B, -1, H // 4, W // 4).contiguous()
        return x


class EmbedMax(nn.Module):
    """MaxPool-based 2x downsampling embedding with residual."""
    def __init__(self, in_channels=2, embed_dims=256):
        super().__init__()
        self.max_embed1 = MaxEmbed(in_channels=in_channels, out_channels=embed_dims,
                                   kernel_size=3, stride=1, padding=1)
        self.embed1 = Embed(in_channels=embed_dims, out_channels=embed_dims,
                            kernel_size=3, stride=1, padding=1)
        self.max_embed2 = MaxEmbed(in_channels=in_channels, out_channels=embed_dims,
                                   kernel_size=1, stride=1, padding=0, shortcut=True)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x, x_feat = self.max_embed1(x, dual=True)
        x = x.reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.embed1(x)

        x_feat = self.max_embed2(x_feat)
        x = (x + x_feat).reshape(T, B, -1, H // 2, W // 2).contiguous()
        return x


# ---------------------------------------------------------------------------
# MaxFormer
# ---------------------------------------------------------------------------

class MaxFormer(nn.Module):
    """MaxFormer: hierarchical spiking architecture.

    Default configuration:
    - Stage 1: 1x DWC7 block (depthwise conv k=7)
    - Stage 2: 2x DWC5 blocks (depthwise conv k=5)
    - Stage 3: 7x SSA blocks (spiking self-attention)
    """

    def __init__(self, in_channels=3, num_classes=1000, embed_dims=512,
                 mlp_ratios=4, depths=10, T=4):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T

        self.patch_embed1 = EmbedOrigImageNet(in_channels=in_channels,
                                              embed_dims=embed_dims // 4)

        self.stage1 = nn.ModuleList([
            Block_DWC(dim=embed_dims // 4, kernel_size=7, mlp_ratio=mlp_ratios)
            for _ in range(1)
        ])

        self.patch_embed2 = EmbedMax(in_channels=embed_dims // 4,
                                     embed_dims=embed_dims // 2)

        self.stage2 = nn.ModuleList([
            Block_DWC(dim=embed_dims // 2, kernel_size=5, mlp_ratio=mlp_ratios)
            for _ in range(2)
        ])

        self.patch_embed3 = EmbedMax(in_channels=embed_dims // 2,
                                     embed_dims=embed_dims)

        self.stage3 = nn.ModuleList([
            Block_SSA(dim=embed_dims, mlp_ratio=mlp_ratios,
                      num_heads=embed_dims // 64)
            for _ in range(7)
        ])

        self.head_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
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
        if len(x.shape) < 5:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        else:
            x = x.transpose(0, 1).contiguous()

        x = self.forward_features(x)
        x = self.head_lif(x)
        x = self.head(x)
        x = x.mean(0)
        return x


def maxformer_10_384(num_classes=1000, T=4, **kwargs):
    return MaxFormer(T=T, embed_dims=384, mlp_ratios=4,
                     in_channels=3, num_classes=num_classes, depths=10, **kwargs)

def maxformer_10_512(num_classes=1000, T=4, **kwargs):
    return MaxFormer(T=T, embed_dims=512, mlp_ratios=4,
                     in_channels=3, num_classes=num_classes, depths=10, **kwargs)

def maxformer_10_768(num_classes=1000, T=4, **kwargs):
    return MaxFormer(T=T, embed_dims=768, mlp_ratios=4,
                     in_channels=3, num_classes=num_classes, depths=10, **kwargs)
