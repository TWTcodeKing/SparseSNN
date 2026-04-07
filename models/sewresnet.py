"""
SEW-ResNet: Spike-Element-Wise ResNet
Paper: "Deep Residual Learning in Spiking Neural Networks" (NeurIPS 2021)
Source: https://github.com/fangwei123456/Spike-Element-Wise-ResNet

Original uses spikingjelly - replaced with standalone neurons/layers.
"""

import torch
import torch.nn as nn
from models import MultiStepIFNeuron, MultiStepLIFNeuron
from models import SeqToANNContainer, SeqToANNContainerT

__all__ = ['SEWResNet', 'sew_resnet18', 'sew_resnet34', 'sew_resnet50',
           'sew_resnet101', 'sew_resnet152']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, neuron_type="lif",stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, connect_f=None):
        super().__init__()
        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = SeqToANNContainer(
            conv3x3(inplanes, planes, stride),
            norm_layer(planes)
        )
        self.sn1 = MultiStepIFNeuron(detach_reset=True) if neuron_type == "if" else MultiStepLIFNeuron(detach_reset=True)

        self.conv2 = SeqToANNContainer(
            conv3x3(planes, planes),
            norm_layer(planes)
        )
        self.downsample = downsample
        self.sn2 = MultiStepIFNeuron(detach_reset=True) if neuron_type == "if" else MultiStepLIFNeuron(detach_reset=True)

    def forward(self, x):
        identity = x
        out = self.sn1(self.conv1(x))
        out = self.sn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.connect_f == 'ADD':
            out += identity
        elif self.connect_f == 'AND':
            out *= identity
        elif self.connect_f == 'IAND':
            out = identity * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, neuron_type="if", stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, connect_f=None):
        super().__init__()
        self.connect_f = connect_f
        self.neuron_type = neuron_type
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = SeqToANNContainer(conv1x1(inplanes, width), norm_layer(width))
        self.sn1 = MultiStepIFNeuron(detach_reset=True) if self.neuron_type == "if" else MultiStepLIFNeuron(detach_reset=True)

        self.conv2 = SeqToANNContainer(conv3x3(width, width, stride, groups, dilation), norm_layer(width))
        self.sn2 = MultiStepIFNeuron(detach_reset=True) if self.neuron_type == "if" else MultiStepLIFNeuron(detach_reset=True)

        self.conv3 = SeqToANNContainer(conv1x1(width, planes * self.expansion), norm_layer(planes * self.expansion))
        self.downsample = downsample
        self.sn3 = MultiStepIFNeuron(detach_reset=True) if self.neuron_type == "if" else MultiStepLIFNeuron(detach_reset=True)

    def forward(self, x):
        identity = x
        out = self.sn1(self.conv1(x))
        out = self.sn2(self.conv2(out))
        out = self.sn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.connect_f == 'ADD':
            out += identity
        elif self.connect_f == 'AND':
            out *= identity
        elif self.connect_f == 'IAND':
            out = identity * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)
        return out


def zero_init_blocks(net, connect_f):
    for m in net.modules():
        if isinstance(m, Bottleneck):
            nn.init.constant_(m.conv3.module[1].weight, 0)
            if connect_f == 'AND':
                nn.init.constant_(m.conv3.module[1].bias, 1)
        elif isinstance(m, BasicBlock):
            nn.init.constant_(m.conv2.module[1].weight, 0)
            if connect_f == 'AND':
                nn.init.constant_(m.conv2.module[1].bias, 1)


class SEWResNet(nn.Module):
    def __init__(self, block, layers, in_channels,neuron_type="if",num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, T=4, connect_f="ADD"):
        super().__init__()
        self.T = T
        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        self.neuron_type = neuron_type
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = MultiStepIFNeuron(detach_reset=True) if neuron_type == "if" else MultiStepLIFNeuron(detach_reset=True)
        self.maxpool = SeqToANNContainer(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, 64, layers[0], connect_f=connect_f)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], connect_f=connect_f)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], connect_f=connect_f)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], connect_f=connect_f)
        self.avgpool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            zero_init_blocks(self, connect_f)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, connect_f=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SeqToANNContainer(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                ),
                MultiStepIFNeuron(detach_reset=True) if self.neuron_type == "if" else MultiStepLIFNeuron(detach_reset=True)
            )
        layers = []
        layers.append(block(self.inplanes, planes, self.neuron_type, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, connect_f))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.neuron_type,groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, connect_f=connect_f))
        return nn.Sequential(*layers)

    def forward(self, x):
        if len(x.shape) == 5:
            B,T,C,H,W = x.shape
            x = x.transpose(0,1).contiguous() # B,T,C,H,W -> T,B,C,H,W
            x = x.flatten(0,1)
            x = self.conv1(x)
            x = self.bn1(x)
            x = x.view(T, B, -1, H//2, W//2)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x = self.sn1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        return self.fc(x.mean(dim=0))


def sew_resnet18(**kwargs):
    return SEWResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def sew_resnet34(**kwargs):
    return SEWResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def sew_resnet50(**kwargs):
    return SEWResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def sew_resnet101(**kwargs):
    return SEWResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def sew_resnet152(**kwargs):
    return SEWResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


# ============================================================================
# CIFAR-specific SEW-ResNet (depths 20/32/44/56/110)
# ============================================================================

class SEWResNetCifar(nn.Module):
    """CIFAR-specific SEW-ResNet.

    Architecture: 3x3 stem (stride=1) → 3 stages [16, 32, 64]
    with strides [1, 2, 2]. Total depth = 6*n + 2.

    Uses the same BasicBlock + element-wise residual as ImageNet SEW-ResNet,
    but adapted for 32x32 inputs (no 7x7 stem, no MaxPool).
    """

    def __init__(self, n, in_channels=3, num_classes=100, neuron_type="if",
                 T=4, connect_f="ADD", zero_init_residual=False, **kwargs):
        super().__init__()
        self.T = T
        self.connect_f = connect_f
        self.neuron_type = neuron_type
        self.inplanes = 16

        # Stem: single 3x3 conv, stride=1
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.sn1 = MultiStepIFNeuron(detach_reset=True) if neuron_type == "if" \
            else MultiStepLIFNeuron(detach_reset=True)

        self.layer1 = self._make_layer(16, n, stride=1)   # 32x32
        self.layer2 = self._make_layer(32, n, stride=2)   # 16x16
        self.layer3 = self._make_layer(64, n, stride=2)   # 8x8

        self.avgpool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            zero_init_blocks(self, connect_f)

    def _make_layer(self, planes, num_blocks, stride):
        norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                SeqToANNContainer(
                    conv1x1(self.inplanes, planes, stride),
                    norm_layer(planes),
                ),
                MultiStepIFNeuron(detach_reset=True) if self.neuron_type == "if"
                else MultiStepLIFNeuron(detach_reset=True)
            )
        layers = []
        layers.append(BasicBlock(self.inplanes, planes, self.neuron_type,
                                 stride=stride, downsample=downsample,
                                 norm_layer=norm_layer, connect_f=self.connect_f))
        self.inplanes = planes
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.inplanes, planes, self.neuron_type,
                                     norm_layer=norm_layer, connect_f=self.connect_f))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Static image: (B, C, H, W) → repeat T times
        if x.dim() == 4:
            x = self.conv1(x)
            x = self.bn1(x)
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        elif x.dim() == 5:
            # DVS: (T, B, C, H, W) or (B, T, C, H, W)
            if x.shape[0] != self.T and x.shape[1] == self.T:
                x = x.transpose(0, 1).contiguous()
            T, B = x.shape[:2]
            x = self.conv1(x.flatten(0, 1))
            x = self.bn1(x)
            x = x.view(T, B, *x.shape[1:])

        x = self.sn1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        return self.fc(x.mean(dim=0))


def sew_resnet_cifar20(**kwargs):
    """CIFAR SEW-ResNet-20 (depth=20, n=3)."""
    return SEWResNetCifar(n=3, **kwargs)

def sew_resnet_cifar32(**kwargs):
    """CIFAR SEW-ResNet-32 (depth=32, n=5)."""
    return SEWResNetCifar(n=5, **kwargs)

def sew_resnet_cifar44(**kwargs):
    """CIFAR SEW-ResNet-44 (depth=44, n=7)."""
    return SEWResNetCifar(n=7, **kwargs)

def sew_resnet_cifar56(**kwargs):
    """CIFAR SEW-ResNet-56 (depth=56, n=9)."""
    return SEWResNetCifar(n=9, **kwargs)

def sew_resnet_cifar110(**kwargs):
    """CIFAR SEW-ResNet-110 (depth=110, n=18)."""
    return SEWResNetCifar(n=18, **kwargs)
