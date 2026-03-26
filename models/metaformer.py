"""
Spike-Driven Transformer V2: Meta Spiking Neural Network Architecture (ICLR 2024)
Source: https://github.com/BICLab/Spike-Driven-Transformer-V2

Key innovations over V1:
- Meta architecture: ConvBlocks in early stages + Transformer blocks in later stages
- RepConv for Q/K/V projection (structural reparameterization)
- SepConv (inverted separable convolution) as token mixer in ConvBlocks

Original uses spikingjelly - replaced with standalone neurons.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models import MultiStepLIFNeuron

__all__ = ['SpikeDrivenTransformerV2', 'build_metaformer']


def _trunc_normal_(tensor, mean=0., std=.02):
    with torch.no_grad():
        tensor.normal_(mean, std)
        tensor.clamp_(-2 * std, 2 * std)
    return tensor


class BNAndPadLayer(nn.Module):
    def __init__(self, pad_pixels, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                    self.bn.bias.detach()
                    - self.bn.running_mean * self.bn.weight.detach()
                    / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0:self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0:self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output


class RepConv(nn.Module):
    def __init__(self, in_channel, out_channel, bias=False):
        super().__init__()
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=1)
        bn = BNAndPadLayer(pad_pixels=1, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 0, groups=in_channel, bias=False),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)


class SepConv(nn.Module):
    """Inverted separable convolution from MobileNetV2."""

    def __init__(self, dim, expansion_ratio=2, bias=False, kernel_size=7, padding=3):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.lif1 = MultiStepLIFNeuron(tau=2.0, detach_reset=True)
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(med_channels)
        self.lif2 = MultiStepLIFNeuron(tau=2.0, detach_reset=True)
        self.dwconv = nn.Conv2d(med_channels, med_channels, kernel_size=kernel_size,
                                padding=padding, groups=med_channels, bias=bias)
        self.pwconv2 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.lif1(x)
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif2(x)
        x = self.dwconv(x.flatten(0, 1))
        x = self.bn2(self.pwconv2(x)).reshape(T, B, -1, H, W)
        return x


class MS_ConvBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.Conv = SepConv(dim=dim)
        self.lif1 = MultiStepLIFNeuron(tau=2.0, detach_reset=True)
        self.conv1 = nn.Conv2d(dim, int(dim * mlp_ratio), kernel_size=3, padding=1, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(dim * mlp_ratio))
        self.lif2 = MultiStepLIFNeuron(tau=2.0, detach_reset=True)
        self.conv2 = nn.Conv2d(int(dim * mlp_ratio), dim, kernel_size=3, padding=1, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.Conv(x) + x
        x_feat = x
        x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, 4 * C, H, W)
        x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
        x = x_feat + x
        return x


class MS_MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        x = x.flatten(3)
        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()

        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()
        return x


class MS_Attention_RepConv(nn.Module):
    """Spike-driven attention with RepConv for Q/K/V projection."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.head_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)

        self.q_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.k_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.v_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.q_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)
        self.k_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)
        self.v_lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)
        self.attn_lif = MultiStepLIFNeuron(tau=2.0, v_threshold=0.5, detach_reset=True)

        self.proj_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W

        x = self.head_lif(x)
        q = self.q_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        k = self.k_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        v = self.v_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)

        q = self.q_lif(q).flatten(3)
        q = q.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k = self.k_lif(k).flatten(3)
        k = k.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v = self.v_lif(v).flatten(3)
        v = v.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x).reshape(T, B, C, H, W)
        x = self.proj_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        return x


class MS_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.attn = MS_Attention_RepConv(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
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


class SpikeDrivenTransformerV2(nn.Module):
    """Meta Spiking Architecture: ConvBlocks (early) + Transformer blocks (late)."""

    def __init__(self, img_size_h=224, img_size_w=224, patch_size=16,
                 in_channels=3, num_classes=1000,
                 embed_dim=[64, 128, 256, 512], num_heads=8, mlp_ratios=4,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=8, sr_ratios=1, T=1):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        # Stage 1: ConvBlocks
        self.downsample1_1 = MS_DownSampling(
            in_channels=in_channels, embed_dims=embed_dim[0] // 2,
            kernel_size=7, stride=2, padding=3, first_layer=True
        )
        self.ConvBlock1_1 = nn.ModuleList([MS_ConvBlock(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios)])

        self.downsample1_2 = MS_DownSampling(
            in_channels=embed_dim[0] // 2, embed_dims=embed_dim[0],
            kernel_size=3, stride=2, padding=1, first_layer=False
        )
        self.ConvBlock1_2 = nn.ModuleList([MS_ConvBlock(dim=embed_dim[0], mlp_ratio=mlp_ratios)])

        # Stage 2: ConvBlocks
        self.downsample2 = MS_DownSampling(
            in_channels=embed_dim[0], embed_dims=embed_dim[1],
            kernel_size=3, stride=2, padding=1, first_layer=False
        )
        self.ConvBlock2_1 = nn.ModuleList([MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)])
        self.ConvBlock2_2 = nn.ModuleList([MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)])

        # Stage 3: Transformer blocks
        self.downsample3 = MS_DownSampling(
            in_channels=embed_dim[1], embed_dims=embed_dim[2],
            kernel_size=3, stride=2, padding=1, first_layer=False
        )
        self.block3 = nn.ModuleList([
            MS_Block(
                dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratios,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[j],
                norm_layer=norm_layer, sr_ratio=sr_ratios
            ) for j in range(6)
        ])

        # Stage 4: Transformer blocks
        self.downsample4 = MS_DownSampling(
            in_channels=embed_dim[2], embed_dims=embed_dim[3],
            kernel_size=3, stride=1, padding=1, first_layer=False
        )
        self.block4 = nn.ModuleList([
            MS_Block(
                dim=embed_dim[3], num_heads=num_heads, mlp_ratio=mlp_ratios,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[j],
                norm_layer=norm_layer, sr_ratio=sr_ratios
            ) for j in range(2)
        ])

        self.lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)
        self.head = nn.Linear(embed_dim[3], num_classes) if num_classes > 0 else nn.Identity()
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
        for blk in self.ConvBlock1_1:
            x = blk(x)
        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2:
            x = blk(x)

        x = self.downsample2(x)
        for blk in self.ConvBlock2_1:
            x = blk(x)
        for blk in self.ConvBlock2_2:
            x = blk(x)

        x = self.downsample3(x)
        for blk in self.block3:
            x = blk(x)

        x = self.downsample4(x)
        for blk in self.block4:
            x = blk(x)
        return x

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = x.flatten(3).mean(3)
        x_lif = self.lif(x)
        x = self.head(x_lif).mean(0)
        return x


class SpikeDrivenTransformerV2Cifar(nn.Module):
    """Meta Spiking Architecture adapted for CIFAR (32x32).

    Reduces downsampling to preserve spatial resolution for attention:
    32→32(stage1.1)→16(stage1.2)→8(stage2)→8(stage3)→8(stage4)
    = 64 tokens at transformer stages (vs 4 tokens in ImageNet variant).
    """

    def __init__(self, in_channels=3, num_classes=100,
                 embed_dim=[64, 128, 256, 512], num_heads=8, mlp_ratios=4,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=8, sr_ratios=1, T=4):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        # Stage 1: ConvBlocks — stride=1 then stride=2 (32→32→16)
        self.downsample1_1 = MS_DownSampling(
            in_channels=in_channels, embed_dims=embed_dim[0] // 2,
            kernel_size=3, stride=1, padding=1, first_layer=True
        )
        self.ConvBlock1_1 = nn.ModuleList([MS_ConvBlock(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios)])

        self.downsample1_2 = MS_DownSampling(
            in_channels=embed_dim[0] // 2, embed_dims=embed_dim[0],
            kernel_size=3, stride=2, padding=1, first_layer=False
        )
        self.ConvBlock1_2 = nn.ModuleList([MS_ConvBlock(dim=embed_dim[0], mlp_ratio=mlp_ratios)])

        # Stage 2: ConvBlocks — stride=2 (16→8)
        self.downsample2 = MS_DownSampling(
            in_channels=embed_dim[0], embed_dims=embed_dim[1],
            kernel_size=3, stride=2, padding=1, first_layer=False
        )
        self.ConvBlock2_1 = nn.ModuleList([MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)])
        self.ConvBlock2_2 = nn.ModuleList([MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)])

        # Stage 3: Transformer blocks — stride=1 (keep 8x8 = 64 tokens)
        self.downsample3 = MS_DownSampling(
            in_channels=embed_dim[1], embed_dims=embed_dim[2],
            kernel_size=3, stride=1, padding=1, first_layer=False
        )
        self.block3 = nn.ModuleList([
            MS_Block(
                dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratios,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[j],
                norm_layer=norm_layer, sr_ratio=sr_ratios
            ) for j in range(6)
        ])

        # Stage 4: Transformer blocks — stride=1 (keep 8x8)
        self.downsample4 = MS_DownSampling(
            in_channels=embed_dim[2], embed_dims=embed_dim[3],
            kernel_size=3, stride=1, padding=1, first_layer=False
        )
        self.block4 = nn.ModuleList([
            MS_Block(
                dim=embed_dim[3], num_heads=num_heads, mlp_ratio=mlp_ratios,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[j],
                norm_layer=norm_layer, sr_ratio=sr_ratios
            ) for j in range(2)
        ])

        self.lif = MultiStepLIFNeuron(tau=2.0, detach_reset=True)
        self.head = nn.Linear(embed_dim[3], num_classes) if num_classes > 0 else nn.Identity()
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
        for blk in self.ConvBlock1_1:
            x = blk(x)
        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2:
            x = blk(x)

        x = self.downsample2(x)
        for blk in self.ConvBlock2_1:
            x = blk(x)
        for blk in self.ConvBlock2_2:
            x = blk(x)

        x = self.downsample3(x)
        for blk in self.block3:
            x = blk(x)

        x = self.downsample4(x)
        for blk in self.block4:
            x = blk(x)
        return x

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = x.flatten(3).mean(3)
        x_lif = self.lif(x)
        x = self.head(x_lif).mean(0)
        return x


def build_metaformer(config):
    """Build a Spike-Driven Transformer V2 (Meta Spikformer) from a config dict.

    Config keys (from YAML):
        embed_dim (list), num_heads, mlp_ratios, depths, sr_ratios,
        qkv_bias, drop_rate, attn_drop_rate, drop_path_rate
    Runtime keys (merged by training script):
        num_classes, T, img_size, in_channels
    """
    img_size = config.get('img_size', 224)
    variant = config.get('variant', 'imagenet')
    common = dict(
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
    if variant == 'cifar':
        return SpikeDrivenTransformerV2Cifar(**common)
    return SpikeDrivenTransformerV2(
        img_size_h=img_size, img_size_w=img_size, **common
    )
