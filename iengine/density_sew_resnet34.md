# Activation Sparsity Profile — SEW-ResNet34

Dataset: **cifar100**  |  Samples: **64**

## Summary

| Metric | Value |
|--------|-------|
| Overall mean density | 0.2001 |
| Overall sparsity | 0.7999 |
| Total profiled layers | 37 |
| Sparse layers (density < 50%) | 35 |
| Very sparse layers (density < 15%) | 16 |

### Density by module type

| Type | Count | Mean Density |
|------|-------|-------------|
| Conv2d | 36 | 0.1903 |
| Linear | 1 | 0.5523 |

## Per-Layer Density

| Layer | Type | Mean Density | Sparsity | Min | Max | Calls | Shape |
|-------|------|-------------|----------|-----|-----|-------|-------|
| layer3.2.conv2.module.0 | Conv2d | 0.0180 | 0.9820 | 0.0171 | 0.0193 | 4 | 64x256x2x2 |
| layer3.4.conv2.module.0 | Conv2d | 0.0180 | 0.9820 | 0.0170 | 0.0191 | 4 | 64x256x2x2 |
| layer3.1.conv2.module.0 | Conv2d | 0.0206 | 0.9794 | 0.0196 | 0.0215 | 4 | 64x256x2x2 |
| layer3.3.conv2.module.0 | Conv2d | 0.0211 | 0.9789 | 0.0201 | 0.0227 | 4 | 64x256x2x2 |
| layer3.5.conv2.module.0 | Conv2d | 0.0218 | 0.9782 | 0.0198 | 0.0237 | 4 | 64x256x2x2 |
| layer4.1.conv2.module.0 | Conv2d | 0.0281 | 0.9719 | 0.0271 | 0.0295 | 4 | 64x512x1x1 |
| layer4.2.conv2.module.0 | Conv2d | 0.0296 | 0.9704 | 0.0289 | 0.0312 | 4 | 64x512x1x1 |
| layer2.2.conv2.module.0 | Conv2d | 0.0334 | 0.9666 | 0.0320 | 0.0357 | 4 | 64x128x4x4 |
| layer2.3.conv2.module.0 | Conv2d | 0.0379 | 0.9621 | 0.0362 | 0.0396 | 4 | 64x128x4x4 |
| layer2.1.conv2.module.0 | Conv2d | 0.0379 | 0.9621 | 0.0363 | 0.0407 | 4 | 64x128x4x4 |
| layer4.0.conv2.module.0 | Conv2d | 0.0663 | 0.9337 | 0.0646 | 0.0674 | 4 | 64x512x1x1 |
| layer1.2.conv2.module.0 | Conv2d | 0.0746 | 0.9254 | 0.0733 | 0.0757 | 4 | 64x64x8x8 |
| layer1.1.conv2.module.0 | Conv2d | 0.0814 | 0.9186 | 0.0794 | 0.0836 | 4 | 64x64x8x8 |
| layer3.0.conv2.module.0 | Conv2d | 0.0943 | 0.9057 | 0.0924 | 0.0967 | 4 | 64x256x2x2 |
| layer1.0.conv2.module.0 | Conv2d | 0.1200 | 0.8800 | 0.1170 | 0.1250 | 4 | 64x64x8x8 |
| layer3.1.conv1.module.0 | Conv2d | 0.1498 | 0.8502 | 0.1484 | 0.1521 | 4 | 64x256x2x2 |
| layer3.2.conv1.module.0 | Conv2d | 0.1691 | 0.8309 | 0.1668 | 0.1732 | 4 | 64x256x2x2 |
| layer2.0.conv2.module.0 | Conv2d | 0.1778 | 0.8222 | 0.1735 | 0.1805 | 4 | 64x128x4x4 |
| layer3.3.conv1.module.0 | Conv2d | 0.1858 | 0.8142 | 0.1827 | 0.1904 | 4 | 64x256x2x2 |
| layer3.4.conv1.module.0 | Conv2d | 0.2104 | 0.7896 | 0.2064 | 0.2158 | 4 | 64x256x2x2 |
| layer4.1.conv1.module.0 | Conv2d | 0.2144 | 0.7856 | 0.2130 | 0.2171 | 4 | 64x512x1x1 |
| layer4.2.conv1.module.0 | Conv2d | 0.2224 | 0.7776 | 0.2208 | 0.2247 | 4 | 64x512x1x1 |
| layer1.0.conv1.module.0 | Conv2d | 0.2259 | 0.7741 | 0.1845 | 0.2748 | 4 | 64x64x8x8 |
| layer3.5.conv1.module.0 | Conv2d | 0.2326 | 0.7674 | 0.2277 | 0.2382 | 4 | 64x256x2x2 |
| layer2.1.conv1.module.0 | Conv2d | 0.2555 | 0.7445 | 0.2521 | 0.2585 | 4 | 64x128x4x4 |
| layer4.0.conv1.module.0 | Conv2d | 0.2646 | 0.7354 | 0.2574 | 0.2702 | 4 | 64x256x2x2 |
| layer4.0.downsample.0.module.0 | Conv2d | 0.2646 | 0.7354 | 0.2574 | 0.2702 | 4 | 64x256x2x2 |
| layer1.1.conv1.module.0 | Conv2d | 0.2786 | 0.7214 | 0.2419 | 0.3229 | 4 | 64x64x8x8 |
| layer2.2.conv1.module.0 | Conv2d | 0.2842 | 0.7158 | 0.2798 | 0.2869 | 4 | 64x128x4x4 |
| layer2.3.conv1.module.0 | Conv2d | 0.3118 | 0.6882 | 0.3071 | 0.3149 | 4 | 64x128x4x4 |
| layer1.2.conv1.module.0 | Conv2d | 0.3204 | 0.6796 | 0.2857 | 0.3619 | 4 | 64x64x8x8 |
| layer3.0.conv1.module.0 | Conv2d | 0.3353 | 0.6647 | 0.3302 | 0.3396 | 4 | 64x128x4x4 |
| layer3.0.downsample.0.module.0 | Conv2d | 0.3353 | 0.6647 | 0.3302 | 0.3396 | 4 | 64x128x4x4 |
| layer2.0.conv1.module.0 | Conv2d | 0.3553 | 0.6447 | 0.3208 | 0.3957 | 4 | 64x64x8x8 |
| layer2.0.downsample.0.module.0 | Conv2d | 0.3553 | 0.6447 | 0.3208 | 0.3957 | 4 | 64x64x8x8 |
| fc | Linear | 0.5523 | 0.4477 | 0.5463 | 0.5579 | 4 | 16x512 |
| conv1 | Conv2d | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 4 | 16x3x32x32 |
