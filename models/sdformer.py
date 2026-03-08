"""
Spike-Driven Transformer V1 (NeurIPS 2023)
Source: https://github.com/BICLab/Spike-Driven-Transformer

Original uses spikingjelly - replaced with standalone neurons.
Key innovation: spike-driven self-attention where Q*K is replaced by
element-wise multiplication, reducing computation from O(N^2) to O(N).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .neurons import MultiStepLIFNeuron

__all__ = ['SpikeDrivenTransformerV1', 'build_sdformer']


def _trunc_normal_(tensor, mean=0., std=.02):
    with torch.no_grad():
        tensor.normal_(mean, std)
        tensor.clamp_(-2 * std, 2 * std)
    return tensor


class MS_MLP_Conv(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., layer=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.res = in_features == hidden_features

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


class MS_SSA_Conv(nn.Module):
    """Spike-driven self-attention with linear complexity.

    Key: Instead of Q@K^T (O(N^2)), uses element-wise K*V then sums over N,
    then multiplies with Q -> O(N) complexity.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1, layer=0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm2d(dim)
        self.q_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)

        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm2d(dim)
        self.k_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)

        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm2d(dim)
        self.v_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)

        self.attn_lif = MultiStepLIFNeuron(tau=2.0, v_threshold=0.5, detach_reset=True)

        self.talking_heads = nn.Conv1d(num_heads, num_heads, kernel_size=1, stride=1, bias=False)
        self.talking_heads_lif = MultiStepLIFNeuron(tau=2.0, v_threshold=0.5, detach_reset=True)

        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm2d(dim)

        self.shortcut_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)

    def forward(self, x):
        T, B, C, H, W = x.shape
        identity = x
        N = H * W

        x = self.shortcut_lif(x)
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, H, W).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.flatten(3).transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, H, W).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.flatten(3).transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, H, W).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.flatten(3).transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous()

        # Spike-driven linear attention: K*V -> sum(N) -> Q * result
        kv = k.mul(v)
        kv = kv.sum(dim=-2, keepdim=True)
        kv = self.talking_heads_lif(kv)
        x = q.mul(kv)

        x = x.transpose(3, 4).reshape(T, B, C, H, W).contiguous()
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W).contiguous()

        x = x + identity
        return x


class MS_Block_Conv(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 sr_ratio=1, layer=0):
        super().__init__()
        self.attn = MS_SSA_Conv(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, layer=layer
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP_Conv(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, layer=layer)

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x


class MS_DownSampling(nn.Module):
    def __init__(self, in_channels=2, embed_dims=256, kernel_size=3, stride=2,
                 padding=1, first_layer=True):
        super().__init__()
        self.encode_conv = nn.Conv2d(in_channels, embed_dims, kernel_size=kernel_size,
                                     stride=stride, padding=padding)
        self.encode_bn = nn.BatchNorm2d(embed_dims)
        if not first_layer:
            self.encode_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)

    def forward(self, x):
        T, B, _, _, _ = x.shape
        if hasattr(self, 'encode_lif'):
            x = self.encode_lif(x)
        x = self.encode_conv(x.flatten(0, 1))
        _, _, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()
        return x


class SpikeDrivenTransformerV1(nn.Module):
    """Spike-Driven Transformer V1 with hierarchical structure."""

    def __init__(self, img_size_h=224, img_size_w=224, patch_size=16,
                 in_channels=3, num_classes=1000,
                 embed_dim=[64, 128, 256, 512], num_heads=8, mlp_ratios=4,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=8, sr_ratios=1, T=4):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        self.downsample1_1 = MS_DownSampling(
            in_channels=in_channels, embed_dims=embed_dim[0] // 2,
            kernel_size=7, stride=2, padding=3, first_layer=True
        )
        self.downsample1_2 = MS_DownSampling(
            in_channels=embed_dim[0] // 2, embed_dims=embed_dim[0],
            kernel_size=3, stride=2, padding=1, first_layer=False
        )
        self.downsample2 = MS_DownSampling(
            in_channels=embed_dim[0], embed_dims=embed_dim[1],
            kernel_size=3, stride=2, padding=1, first_layer=False
        )
        self.downsample3 = MS_DownSampling(
            in_channels=embed_dim[1], embed_dims=embed_dim[2],
            kernel_size=3, stride=2, padding=1, first_layer=False
        )

        self.block3 = nn.ModuleList([
            MS_Block_Conv(
                dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratios,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[j],
                norm_layer=norm_layer, sr_ratio=sr_ratios, layer=j
            ) for j in range(depths)
        ])

        self.lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)
        self.head = nn.Linear(embed_dim[2], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.downsample1_1(x)
        x = self.downsample1_2(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        for blk in self.block3:
            x = blk(x)
        return x

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = x.flatten(3).mean(3)
        x = self.lif(x)
        x = self.head(x).mean(0)
        return x


def build_sdformer(config):
    """Build a Spike-Driven Transformer V1 from a config dict.

    Config keys (from YAML):
        embed_dim (list), num_heads, mlp_ratios, depths, sr_ratios,
        qkv_bias, drop_rate, attn_drop_rate, drop_path_rate
    Runtime keys (merged by training script):
        num_classes, T, img_size, in_channels
    """
    img_size = config.get('img_size', 224)
    return SpikeDrivenTransformerV1(
        img_size_h=img_size, img_size_w=img_size,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        mlp_ratios=config.get('mlp_ratios', 4),
        in_channels=config.get('in_channels', 3),
        num_classes=config['num_classes'],
        qkv_bias=config.get('qkv_bias', False),
        qk_scale=config.get('qk_scale', None),
        norm_layer=nn.LayerNorm,
        depths=config.get('depths', 8),
        sr_ratios=config.get('sr_ratios', 1),
        T=config['T'],
        drop_rate=config.get('drop_rate', 0.0),
        attn_drop_rate=config.get('attn_drop_rate', 0.0),
        drop_path_rate=config.get('drop_path_rate', 0.0),
    )
