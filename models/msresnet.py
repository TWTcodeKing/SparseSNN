"""
MS-ResNet: Advancing Spiking Neural Networks towards Deep Residual Learning
Paper: "Advancing Spiking Neural Networks towards Deep Residual Learning"
Source: https://github.com/Ariande1/MS-ResNet

Uses original-compatible components:
- MSNeuron: mem = mem * decay * (1-spike) + x (matching original mem_update)
- TDBNContainer: Conv2d (T*B flattened) + BatchNorm3d (temporal-aware)
  to match the original Snn_Conv2d + batch_norm_2d pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neurons import heaviside

__all__ = ['ms_resnet18', 'ms_resnet34', 'ms_resnet50', 'ms_resnet104']

_TIME_WINDOW = 6
_DECAY = 0.25
_THRESH = 0.5


# ============================================================================
# Original-compatible components
# ============================================================================

class MSNeuron(nn.Module):
    """Multi-step spiking neuron matching the original MS-ResNet mem_update.

    Dynamics per timestep:
        mem = mem * decay * (1 - spike_prev) + x[t]
        spike = Heaviside(mem - thresh)

    This differs from standard LIF: decay is applied multiplicatively with
    integrated reset (mem zeroed when spike=1), and input x is added at full
    scale (not divided by tau).
    """

    def __init__(self, decay=_DECAY, thresh=_THRESH):
        super().__init__()
        self.decay = decay
        self.thresh = thresh

    def reset(self):
        pass  # stateless — resets happen inline in forward

    def forward(self, x):
        """x: (T, B, C, H, W) → (T, B, C, H, W) binary spikes."""
        T = x.shape[0]
        mem = torch.zeros_like(x[0])
        spike = torch.zeros_like(x[0])
        output = torch.zeros_like(x)
        for t in range(T):
            mem = mem * self.decay * (1.0 - spike) + x[t]
            spike = heaviside(mem - self.thresh, 'gate')
            output[t] = spike
        return output


class TDBNContainer(nn.Module):
    """Container wrapping Conv2d + BatchNorm3d (temporal-aware BN).

    Applies Conv2d on flattened (T*B, C, H, W), then BatchNorm3d on the
    5D (B, C, T, H, W) tensor. This matches the original Snn_Conv2d +
    batch_norm_2d pipeline.

    State dict keys match SeqToANNContainer layout:
        module.0.weight — Conv2d
        module.1.weight — BatchNorm3d (same shape [C] as BatchNorm2d)
    """

    def __init__(self, conv, num_features, bn_init_thresh=True):
        super().__init__()
        self.module = nn.Sequential(conv, nn.BatchNorm3d(num_features))
        if bn_init_thresh:
            nn.init.constant_(self.module[1].weight, _THRESH)

    def forward(self, x):
        """x: (T, B, C, H, W) → (T, B, C', H', W')"""
        T, B = x.shape[:2]
        # Conv2d on flattened (T*B, C, H, W)
        conv_out = self.module[0](x.flatten(0, 1))
        conv_out = conv_out.view(T, B, *conv_out.shape[1:])
        # BN3d on (B, C, T, H, W)
        bn_out = self.module[1](conv_out.permute(1, 2, 0, 3, 4))
        return bn_out.permute(2, 0, 1, 3, 4)  # back to (T, B, C, H, W)


# ============================================================================
# ResNet-18/34 blocks
# ============================================================================

class BasicBlock18(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super().__init__()
        self.sn1 = MSNeuron()
        self.conv_bn1 = TDBNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            out_channels, bn_init_thresh=True,
        )
        self.sn2 = MSNeuron()
        self.conv_bn2 = TDBNContainer(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            out_channels, bn_init_thresh=False,  # zero-init for residual
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = TDBNContainer(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                out_channels, bn_init_thresh=True,
            )

    def forward(self, x):
        out = self.conv_bn1(self.sn1(x))
        out = self.conv_bn2(self.sn2(out))
        sc = self.shortcut(x) if isinstance(self.shortcut, TDBNContainer) else x
        return out + sc


class MSResNet18(nn.Module):
    """MS-ResNet for ResNet-18/34/50 configuration."""

    def __init__(self, block, num_block, in_channels, num_classes=1000, T=None,
                 **kwargs):
        super().__init__()
        self.time_window = T or _TIME_WINDOW
        self.in_channels = 64

        self.conv1 = TDBNContainer(
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=3, bias=False, stride=2),
            64, bn_init_thresh=True,
        )
        self.sn_out = MSNeuron()
        self.conv2_x = self._make_layer(block, 64, num_block[0], 2)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Accept both (B, C, H, W) static images and (T, B, C, H, W) DVS sequences
        if x.dim() == 4:
            T = self.time_window
            input_seq = x.unsqueeze(0).repeat(T, 1, 1, 1, 1)
        else:
            input_seq = x  # (T, B, C, H, W)
        output = self.conv1(input_seq)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.sn_out(output)
        # (T, B, C, H, W) -> temporal average then spatial pool
        output = output.mean(dim=0)
        output = F.adaptive_avg_pool2d(output, 1).flatten(1)
        output = self.fc(output)
        return output


# ============================================================================
# ResNet-50 Bottleneck block (1x1 → 3x3 → 1x1, expansion=4)
# ============================================================================

class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super().__init__()
        self.sn1 = MSNeuron()
        self.conv_bn1 = TDBNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            out_channels, bn_init_thresh=True,
        )
        self.sn2 = MSNeuron()
        self.conv_bn2 = TDBNContainer(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            out_channels, bn_init_thresh=True,
        )
        self.sn3 = MSNeuron()
        self.conv_bn3 = TDBNContainer(
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False),
            out_channels * self.expansion, bn_init_thresh=False,  # zero-init
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = TDBNContainer(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                out_channels * self.expansion, bn_init_thresh=True,
            )

    def forward(self, x):
        out = self.conv_bn1(self.sn1(x))
        out = self.conv_bn2(self.sn2(out))
        out = self.conv_bn3(self.sn3(out))
        sc = self.shortcut(x) if isinstance(self.shortcut, TDBNContainer) else x
        return out + sc


# ============================================================================
# ResNet-104 blocks (deeper, with AvgPool shortcut)
# ============================================================================

class BasicBlock104(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super().__init__()
        self.sn1 = MSNeuron()
        self.conv_bn1 = TDBNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            out_channels, bn_init_thresh=True,
        )
        self.sn2 = MSNeuron()
        self.conv_bn2 = TDBNContainer(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            out_channels, bn_init_thresh=False,  # zero-init
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2)),
                TDBNContainer(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              stride=1, bias=False),
                    out_channels, bn_init_thresh=True,
                ),
            )

    def forward(self, x):
        out = self.conv_bn1(self.sn1(x))
        out = self.conv_bn2(self.sn2(out))
        if isinstance(self.shortcut, nn.Sequential) and len(self.shortcut) > 0:
            # AvgPool3d expects (B,C,T,H,W)
            sc = x.permute(1, 2, 0, 3, 4)  # (B,C,T,H,W)
            sc = self.shortcut[0](sc)  # AvgPool3d
            sc = sc.permute(2, 0, 1, 3, 4)  # back to (T,B,C,H,W)
            sc = self.shortcut[1](sc)  # TDBNContainer
        else:
            sc = x
        return out + sc


class MSResNet104(nn.Module):
    """MS-ResNet-104: deeper variant with 3-conv stem."""

    def __init__(self, block, num_block, in_channels, num_classes=1000, T=None,
                 **kwargs):
        super().__init__()
        self.time_window = T or _TIME_WINDOW
        self.in_channels = 64

        # 3-conv stem stored as conv1.module = Sequential(Conv, Conv, Conv, BN3d)
        # to match converted checkpoint key structure: conv1.module.{0,1,2,3}.*
        self.conv1 = nn.Module()
        self.conv1.module = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(64),
        )
        nn.init.constant_(self.conv1.module[3].weight, _THRESH)

        self.sn_out = MSNeuron()
        self.conv2_x = self._make_layer(block, 64, num_block[0], 2)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def _stem_forward(self, x):
        """Apply 3-conv stem with TDBN."""
        T, B = x.shape[:2]
        out = x.flatten(0, 1)  # (T*B, C, H, W)
        out = self.conv1.module[0](out)
        out = self.conv1.module[1](out)
        out = self.conv1.module[2](out)
        out = out.view(T, B, *out.shape[1:])
        # BN3d on (B, C, T, H, W)
        out = self.conv1.module[3](out.permute(1, 2, 0, 3, 4))
        return out.permute(2, 0, 1, 3, 4)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 4:
            T = self.time_window
            input_seq = x.unsqueeze(0).repeat(T, 1, 1, 1, 1)
        else:
            input_seq = x
        output = self._stem_forward(input_seq)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.sn_out(output)
        output = output.mean(dim=0)
        output = F.adaptive_avg_pool2d(output, 1).flatten(1)
        output = self.dropout(output)
        output = self.fc(output)
        return output


# ============================================================================
# Factory functions
# ============================================================================

def ms_resnet18(num_classes=1000, **kwargs):
    return MSResNet18(BasicBlock18, [2, 2, 2, 2], num_classes=num_classes, **kwargs)

def ms_resnet34(num_classes=1000, **kwargs):
    return MSResNet18(BasicBlock18, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def ms_resnet50(num_classes=1000, **kwargs):
    return MSResNet18(BottleneckBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def ms_resnet104(num_classes=1000, **kwargs):
    return MSResNet104(BasicBlock104, [3, 8, 32, 8], num_classes=num_classes, **kwargs)


# ============================================================================
# CIFAR-specific MS-ResNet (depths 20/32/44/56/110)
# ============================================================================

class BasicBlockCifar(nn.Module):
    """Basic block for CIFAR MS-ResNet.

    Same pre-activation pattern as BasicBlock18 (sn → conv_bn),
    but shortcut uses AvgPool + Conv1x1 for spatial downsampling
    (following the paper's CIFAR variant, same as BasicBlock104).
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super().__init__()
        self.sn1 = MSNeuron()
        self.conv_bn1 = TDBNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            out_channels, bn_init_thresh=True,
        )
        self.sn2 = MSNeuron()
        self.conv_bn2 = TDBNContainer(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            out_channels, bn_init_thresh=False,  # zero-init
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool3d((1, stride, stride), stride=(1, stride, stride)),
                TDBNContainer(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              stride=1, bias=False),
                    out_channels, bn_init_thresh=True,
                ),
            )

    def forward(self, x):
        out = self.conv_bn1(self.sn1(x))
        out = self.conv_bn2(self.sn2(out))
        if len(self.shortcut) > 0:
            # AvgPool3d expects (B,C,T,H,W)
            sc = x.permute(1, 2, 0, 3, 4)
            sc = self.shortcut[0](sc)
            sc = sc.permute(2, 0, 1, 3, 4)
            sc = self.shortcut[1](sc)
        else:
            sc = x
        return out + sc


class MSResNetCifar(nn.Module):
    """CIFAR-specific MS-ResNet (paper Table VI).

    Architecture: 3x3 stem (stride=1, no downsampling) → 3 stages [16, 32, 64]
    with strides [1, 2, 2]. Total depth = 6*n + 2 where n = blocks per stage.

    Paper reported accuracy on CIFAR-100:
        depth 32  (n=5):  61.35%
        depth 44  (n=7):  63.84%
        depth 56  (n=9):  65.24%
        depth 110 (n=18): 66.83%
    """

    def __init__(self, n, in_channels=3, num_classes=100, T=None,
                 stem_stride=1, first_stage_stride=1, **kwargs):
        super().__init__()
        self.time_window = T or _TIME_WINDOW

        # Stem: 3x3 conv (stride=1 for CIFAR 32x32, stride=2 for DVS 128x128)
        self.conv1 = TDBNContainer(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=stem_stride,
                      padding=1, bias=False),
            16, bn_init_thresh=True,
        )

        self.in_channels = 16
        self.layer1 = self._make_layer(16, n, stride=first_stage_stride)
        self.layer2 = self._make_layer(32, n, stride=2)
        self.layer3 = self._make_layer(64, n, stride=2)

        self.sn_out = MSNeuron()
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlockCifar(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 4:
            T = self.time_window
            x = x.unsqueeze(0).repeat(T, 1, 1, 1, 1)
        else:
            x = x.permute(1, 0, 2, 3, 4)  # (B,C,H,W) → (T,B,C,H,W) with T=1
        # only 4 dim static images or 5 dim dvs images
        output = self.conv1(x)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.sn_out(output)
        output = output.mean(dim=0)
        output = F.adaptive_avg_pool2d(output, 1).flatten(1)
        output = self.fc(output)
        return output


def ms_resnet_cifar20(num_classes=100, **kwargs):
    """CIFAR MS-ResNet-20 (depth=20, n=3)."""
    return MSResNetCifar(n=3, num_classes=num_classes, **kwargs)

def ms_resnet_cifar32(num_classes=100, **kwargs):
    """CIFAR MS-ResNet-32 (depth=32, n=5)."""
    return MSResNetCifar(n=5, num_classes=num_classes, **kwargs)

def ms_resnet_cifar44(num_classes=100, **kwargs):
    """CIFAR MS-ResNet-44 (depth=44, n=7)."""
    return MSResNetCifar(n=7, num_classes=num_classes, **kwargs)

def ms_resnet_cifar56(num_classes=100, **kwargs):
    """CIFAR MS-ResNet-56 (depth=56, n=9)."""
    return MSResNetCifar(n=9, num_classes=num_classes, **kwargs)

def ms_resnet_cifar110(num_classes=100, **kwargs):
    """CIFAR MS-ResNet-110 (depth=110, n=18)."""
    return MSResNetCifar(n=18, num_classes=num_classes, **kwargs)


def ms_resnet_dvs20(num_classes=10, **kwargs):
    """DVS MS-ResNet-20 (128x128 input, stem_stride=2, first_stage_stride=2).
    Paper: 75.56% on CIFAR10-DVS with 0.27M params.
    Spatial: 128→64(stem)→32(layer1)→16(layer2)→8(layer3)."""
    return MSResNetCifar(n=3, num_classes=num_classes,
                         stem_stride=2, first_stage_stride=2, **kwargs)
