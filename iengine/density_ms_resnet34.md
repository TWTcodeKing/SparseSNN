# Activation Sparsity Profile — MS-ResNet34

Dataset: **cifar100**  |  Samples: **64**

## Summary

| Metric | Value |
|--------|-------|
| Overall mean density | 0.2232 |
| Overall sparsity | 0.7768 |
| Total profiled layers | 38 |
| Sparse layers (density < 50%) | 33 |
| Very sparse layers (density < 15%) | 26 |

### Density by module type

| Type | Count | Mean Density |
|------|-------|-------------|
| Conv2d | 37 | 0.2184 |
| Linear | 1 | 0.4014 |

## Per-Layer Density

| Layer | Type | Mean Density | Sparsity | Min | Max | Calls | Shape |
|-------|------|-------------|----------|-----|-----|-------|-------|
| conv4_x.5.conv_bn2.module.0 | Conv2d | 0.0016 | 0.9984 | 0.0014 | 0.0020 | 4 | 64x256x2x2 |
| conv4_x.4.conv_bn2.module.0 | Conv2d | 0.0048 | 0.9952 | 0.0043 | 0.0052 | 4 | 64x256x2x2 |
| conv4_x.2.conv_bn2.module.0 | Conv2d | 0.0049 | 0.9951 | 0.0044 | 0.0052 | 4 | 64x256x2x2 |
| conv4_x.3.conv_bn2.module.0 | Conv2d | 0.0070 | 0.9930 | 0.0061 | 0.0073 | 4 | 64x256x2x2 |
| conv5_x.1.conv_bn2.module.0 | Conv2d | 0.0231 | 0.9769 | 0.0223 | 0.0238 | 4 | 64x512x1x1 |
| conv5_x.2.conv_bn2.module.0 | Conv2d | 0.0252 | 0.9748 | 0.0241 | 0.0271 | 4 | 64x512x1x1 |
| conv4_x.1.conv_bn2.module.0 | Conv2d | 0.0361 | 0.9639 | 0.0324 | 0.0402 | 4 | 64x256x2x2 |
| conv3_x.1.conv_bn2.module.0 | Conv2d | 0.0454 | 0.9546 | 0.0389 | 0.0563 | 4 | 64x128x4x4 |
| conv3_x.2.conv_bn2.module.0 | Conv2d | 0.0462 | 0.9538 | 0.0424 | 0.0482 | 4 | 64x128x4x4 |
| conv3_x.3.conv_bn2.module.0 | Conv2d | 0.0485 | 0.9515 | 0.0444 | 0.0510 | 4 | 64x128x4x4 |
| conv4_x.1.conv_bn1.module.0 | Conv2d | 0.0645 | 0.9355 | 0.0622 | 0.0674 | 4 | 64x256x2x2 |
| conv5_x.1.conv_bn1.module.0 | Conv2d | 0.0756 | 0.9244 | 0.0731 | 0.0769 | 4 | 64x512x1x1 |
| conv4_x.2.conv_bn1.module.0 | Conv2d | 0.0780 | 0.9220 | 0.0764 | 0.0795 | 4 | 64x256x2x2 |
| conv5_x.0.conv_bn2.module.0 | Conv2d | 0.0817 | 0.9183 | 0.0751 | 0.0853 | 4 | 64x512x1x1 |
| conv3_x.1.conv_bn1.module.0 | Conv2d | 0.0832 | 0.9168 | 0.0747 | 0.0939 | 4 | 64x128x4x4 |
| conv3_x.2.conv_bn1.module.0 | Conv2d | 0.0859 | 0.9141 | 0.0809 | 0.0915 | 4 | 64x128x4x4 |
| conv2_x.2.conv_bn2.module.0 | Conv2d | 0.0899 | 0.9101 | 0.0867 | 0.0926 | 4 | 64x64x8x8 |
| conv4_x.3.conv_bn1.module.0 | Conv2d | 0.0926 | 0.9074 | 0.0905 | 0.0942 | 4 | 64x256x2x2 |
| conv2_x.0.conv_bn1.module.0 | Conv2d | 0.0998 | 0.9002 | 0.0865 | 0.1185 | 4 | 64x64x16x16 |
| conv4_x.0.conv_bn2.module.0 | Conv2d | 0.1052 | 0.8948 | 0.1049 | 0.1053 | 4 | 64x256x2x2 |
| conv5_x.2.conv_bn1.module.0 | Conv2d | 0.1093 | 0.8907 | 0.1048 | 0.1117 | 4 | 64x512x1x1 |
| conv3_x.3.conv_bn1.module.0 | Conv2d | 0.1112 | 0.8888 | 0.1085 | 0.1156 | 4 | 64x128x4x4 |
| conv4_x.4.conv_bn1.module.0 | Conv2d | 0.1147 | 0.8853 | 0.1128 | 0.1167 | 4 | 64x256x2x2 |
| conv3_x.0.conv_bn2.module.0 | Conv2d | 0.1342 | 0.8658 | 0.1316 | 0.1373 | 4 | 64x128x4x4 |
| conv2_x.1.conv_bn2.module.0 | Conv2d | 0.1369 | 0.8631 | 0.1360 | 0.1375 | 4 | 64x64x8x8 |
| conv4_x.5.conv_bn1.module.0 | Conv2d | 0.1382 | 0.8618 | 0.1353 | 0.1410 | 4 | 64x256x2x2 |
| conv5_x.0.conv_bn1.module.0 | Conv2d | 0.1601 | 0.8399 | 0.1559 | 0.1638 | 4 | 64x256x2x2 |
| conv4_x.0.conv_bn1.module.0 | Conv2d | 0.1626 | 0.8374 | 0.1587 | 0.1647 | 4 | 64x128x4x4 |
| conv2_x.1.conv_bn1.module.0 | Conv2d | 0.2090 | 0.7910 | 0.1983 | 0.2231 | 4 | 64x64x8x8 |
| conv2_x.2.conv_bn1.module.0 | Conv2d | 0.2230 | 0.7770 | 0.2145 | 0.2347 | 4 | 64x64x8x8 |
| conv2_x.0.conv_bn2.module.0 | Conv2d | 0.2358 | 0.7642 | 0.2335 | 0.2381 | 4 | 64x64x8x8 |
| conv3_x.0.conv_bn1.module.0 | Conv2d | 0.2459 | 0.7541 | 0.2391 | 0.2566 | 4 | 64x64x8x8 |
| fc | Linear | 0.4014 | 0.5986 | 0.3958 | 0.4091 | 4 | 16x512 |
| conv1.module.0 | Conv2d | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 4 | 64x3x32x32 |
| conv2_x.0.shortcut.module.0 | Conv2d | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 4 | 64x64x16x16 |
| conv3_x.0.shortcut.module.0 | Conv2d | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 4 | 64x64x8x8 |
| conv4_x.0.shortcut.module.0 | Conv2d | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 4 | 64x128x4x4 |
| conv5_x.0.shortcut.module.0 | Conv2d | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 4 | 64x256x2x2 |
