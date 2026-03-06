"""
Spikformer: When Spiking Neural Network Meets Transformer (ICLR 2023)
Source: https://github.com/ZK-Zhou/spikformer

Original uses spikingjelly.clock_driven.neuron.MultiStepLIFNode.
Replaced with standalone MultiStepLIFNeuron from neurons.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models import MultiStepLIFNeuron
from models import SeqToANNContainer

__all__ = ['Spikformer', 'spikformer_8_512', 'spikformer_8_384', 'spikformer_8_768']


def _to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return (x, x)


def _trunc_normal_(tensor, mean=0., std=.02):
    with torch.no_grad():
        tensor.normal_(mean, std)
        tensor.clamp_(-2 * std, 2 * std)
    return tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (1,) * (x.ndim - 1) + (x.shape[-1],)  # simplified
        random_tensor = torch.rand(x.shape[0], 1, *([1] * (x.ndim - 2)), device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


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


class SSA(nn.Module):
    """Spiking Self-Attention."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
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

    def forward(self, x, res_attn):
        T, B, C, H, W = x.shape
        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x)).reshape(T, B, C, H, W))
        return x, v


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, res_attn):
        x_attn, attn = self.attn(x, res_attn)
        x = x + x_attn
        x = x + self.mlp(x)
        return x, attn


class SPS(nn.Module):
    """Spiking Patch Splitting."""

    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = _to_2tuple(patch_size)
        self.patch_size = patch_size
        self.H, self.W = img_size_h // patch_size[0], img_size_w // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.proj_conv1 = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = MultiStepLIFNeuron(tau=2.0, detach_reset=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.proj_conv2 = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = MultiStepLIFNeuron(tau=2.0, detach_reset=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.proj_conv3 = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MultiStepLIFNeuron(tau=2.0, detach_reset=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif(x).flatten(0, 1).contiguous()
        x = self.maxpool(x)

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj_lif1(x).flatten(0, 1).contiguous()
        x = self.maxpool1(x)

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = self.proj_lif2(x).flatten(0, 1).contiguous()
        x = self.maxpool2(x)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x).reshape(T, B, -1, H // 8, W // 8).contiguous()
        x = self.proj_lif3(x).flatten(0, 1).contiguous()
        x = self.maxpool3(x)

        x_feat = x.reshape(T, B, -1, H // 16, W // 16).contiguous()
        x = self.rpe_conv(x)
        x = self.rpe_bn(x).reshape(T, B, -1, H // 16, W // 16).contiguous()
        x = self.rpe_lif(x)
        x = x + x_feat

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class Spikformer(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=16, in_channels=2,
                 num_classes=11, embed_dims=64, num_heads=8, mlp_ratios=4,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=8, sr_ratios=1, T=4):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        self.patch_embed = SPS(
            img_size_h=img_size_h, img_size_w=img_size_w,
            patch_size=patch_size, in_channels=in_channels,
            embed_dims=embed_dims
        )

        self.block = nn.ModuleList([
            Block(dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[j],
                  norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)
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
        x, (H, W) = self.patch_embed(x)
        attn = None
        for blk in self.block:
            x, attn = blk(x, attn)
        return x.flatten(3).mean(3)

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x


def spikformer_8_384(num_classes=1000, T=4, **kwargs):
    return Spikformer(
        img_size_h=224, img_size_w=224, patch_size=16,
        embed_dims=384, num_heads=6, mlp_ratios=4,
        in_channels=3, num_classes=num_classes, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=8, sr_ratios=1, T=T, **kwargs
    )

def spikformer_8_512(num_classes=1000, T=4, **kwargs):
    return Spikformer(
        img_size_h=224, img_size_w=224, patch_size=16,
        embed_dims=512, num_heads=8, mlp_ratios=4,
        in_channels=3, num_classes=num_classes, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=8, sr_ratios=1, T=T, **kwargs
    )

def spikformer_8_768(num_classes=1000, T=4, **kwargs):
    return Spikformer(
        img_size_h=224, img_size_w=224, patch_size=16,
        embed_dims=768, num_heads=12, mlp_ratios=4,
        in_channels=3, num_classes=num_classes, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=8, sr_ratios=1, T=T, **kwargs
    )
