"""
MS-ResNet: Advancing Spiking Neural Networks towards Deep Residual Learning
Paper: "Advancing Spiking Neural Networks towards Deep Residual Learning"
Source: https://github.com/Ariande1/MS-ResNet

Rewritten to use MultiStepLIFNeuron and SeqToANNContainer.
"""

import torch.nn as nn
import torch.nn.functional as F
from models import MultiStepLIFNeuron
from models import SeqToANNContainer

__all__ = ['ms_resnet18', 'ms_resnet34', 'ms_resnet104']

_TIME_WINDOW = 6


# ============================================================================
# ResNet-18/34 blocks
# ============================================================================

class BasicBlock18(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.sn1 = MultiStepLIFNeuron(tau=4, v_threshold=0.5, detach_reset=True)
        self.conv_bn1 = SeqToANNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.sn2 = MultiStepLIFNeuron(tau=4, v_threshold=0.5, detach_reset=True)
        self.conv_bn2 = SeqToANNContainer(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = SeqToANNContainer(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.conv_bn1(self.sn1(x))
        out = self.conv_bn2(self.sn2(out))
        return out + self.shortcut(x)


class MSResNet18(nn.Module):
    """MS-ResNet for ResNet-18/34 configuration."""

    def __init__(self, block, num_block, num_classes=1000, time_window=None):
        super().__init__()
        self.time_window = time_window or _TIME_WINDOW
        self.in_channels = 64

        self.conv1 = SeqToANNContainer(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, bias=False, stride=2),
            nn.BatchNorm2d(64),
        )
        self.sn_out = MultiStepLIFNeuron(tau=4, v_threshold=0.5, detach_reset=True)
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
        T = self.time_window
        input_seq = x.unsqueeze(0).repeat(T, 1, 1, 1, 1)
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
# ResNet-104 blocks (deeper, with AvgPool shortcut)
# ============================================================================

class BasicBlock104(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.sn1 = MultiStepLIFNeuron(tau=4, v_threshold=0.5, detach_reset=True)
        self.conv_bn1 = SeqToANNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.sn2 = MultiStepLIFNeuron(tau=4, v_threshold=0.5, detach_reset=True)
        self.conv_bn2 = SeqToANNContainer(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2)),
                SeqToANNContainer(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              stride=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                ),
            )

    def forward(self, x):
        out = self.conv_bn1(self.sn1(x))
        out = self.conv_bn2(self.sn2(out))
        return out + self.shortcut(x)


class MSResNet104(nn.Module):
    """MS-ResNet-104: deeper variant with 3-conv stem."""

    def __init__(self, block, num_block, num_classes=1000, time_window=None):
        super().__init__()
        self.time_window = time_window or _TIME_WINDOW
        self.in_channels = 64

        self.conv1 = SeqToANNContainer(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
        )
        self.sn_out = MultiStepLIFNeuron(tau=4, v_threshold=0.5, detach_reset=True)
        self.conv2_x = self._make_layer(block, 64, num_block[0], 2)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        T = self.time_window
        input_seq = x.unsqueeze(0).repeat(T, 1, 1, 1, 1)
        output = self.conv1(input_seq)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.sn_out(output)
        # (T, B, C, H, W) -> temporal average then spatial pool
        output = output.mean(dim=0)
        output = F.adaptive_avg_pool2d(output, 1).flatten(1)
        output = self.dropout(output)
        output = self.fc(output)
        return output


def ms_resnet18(num_classes=1000, time_window=6, **kwargs):
    return MSResNet18(BasicBlock18, [2, 2, 2, 2], num_classes=num_classes, time_window=time_window)

def ms_resnet34(num_classes=1000, time_window=6, **kwargs):
    return MSResNet18(BasicBlock18, [3, 4, 6, 3], num_classes=num_classes, time_window=time_window)

def ms_resnet104(num_classes=1000, time_window=6, **kwargs):
    return MSResNet104(BasicBlock104, [3, 8, 32, 8], num_classes=num_classes, time_window=time_window)
