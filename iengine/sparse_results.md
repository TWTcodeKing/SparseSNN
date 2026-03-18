# Sparse Acceleration Benchmark Results

Dataset: **cifar100**  |  Samples: **100**

## Latency & Accuracy Comparison

| Model | Backend | Dense (ms) | Sparse (ms) | Speedup | Dense Acc | Sparse Acc | Acc Delta |
|-------|---------|-----------|------------|---------|-----------|------------|-----------|
| Spikformer | torch_sparse | 1.935 | 54.475 | 0.04x | 67.86% | 67.86% | +0.00% |
| Spikformer | semi_structured | 1.935 | 2.584 | 0.75x | 67.86% | 61.61% | -6.25% |
| Spikformer | triton_sparse | 1.935 | 90.840 | 0.02x | 67.86% | 67.86% | +0.00% |
| SEW-ResNet34 | torch_sparse | 1.945 | 2.104 | 0.92x | 58.04% | 58.04% | +0.00% |
| SEW-ResNet34 | semi_structured | 1.945 | 2.158 | 0.90x | 58.04% | 58.04% | +0.00% |
| SEW-ResNet34 | triton_sparse | 1.945 | 3.257 | 0.60x | 58.04% | 58.04% | +0.00% |
| MS-ResNet34 | torch_sparse | 1.854 | 3.912 | 0.47x | 57.14% | 57.14% | +0.00% |
| MS-ResNet34 | semi_structured | 1.854 | 2.198 | 0.84x | 57.14% | 57.14% | +0.00% |
| MS-ResNet34 | triton_sparse | 1.854 | 3.647 | 0.51x | 57.14% | 58.93% | +1.79% |

## Backend Execution Stats

### Spikformer — torch_sparse

| Metric | Value |
|--------|-------|
| total_ops | 157593600 |
| effective_ops | 39546797 |
| density | 0.2509 |

<details><summary>Per-layer stats (29 layers)</summary>

| Layer | type | total_ops | effective_ops | density | total_calls | sparse_calls |
|-------| --- | --- | --- | --- | --- | --- |
| block.0.attn | attention | 25165824 | 1926870 | 0.0766 | 8 |  |
| block.0.attn.k_linear | linear | 1179648 | 151275 | 0.1282 | 8 | 8 |
| block.0.attn.proj_linear | linear | 1179648 | 1179648 | 1.0000 | 8 | 0 |
| block.0.attn.q_linear | linear | 1179648 | 151275 | 0.1282 | 8 | 8 |
| block.0.attn.v_linear | linear | 1179648 | 151275 | 0.1282 | 8 | 8 |
| block.0.mlp.fc1_linear | linear | 4718592 | 4718592 | 1.0000 | 8 | 0 |
| block.0.mlp.fc2_linear | linear | 4718592 | 98562 | 0.0209 | 8 | 8 |
| block.1.attn | attention | 25165824 | 1875652 | 0.0745 | 8 |  |
| block.1.attn.k_linear | linear | 1179648 | 1179648 | 1.0000 | 8 | 0 |
| block.1.attn.proj_linear | linear | 1179648 | 157463 | 0.1335 | 8 | 8 |
| block.1.attn.q_linear | linear | 1179648 | 1179648 | 1.0000 | 8 | 0 |
| block.1.attn.v_linear | linear | 1179648 | 1179648 | 1.0000 | 8 | 0 |
| block.1.mlp.fc1_linear | linear | 4718592 | 4718592 | 1.0000 | 8 | 0 |
| block.1.mlp.fc2_linear | linear | 4718592 | 93992 | 0.0199 | 8 | 8 |
| block.2.attn | attention | 25165824 | 1692276 | 0.0672 | 8 |  |
| block.2.attn.k_linear | linear | 1179648 | 1179648 | 1.0000 | 8 | 0 |
| block.2.attn.proj_linear | linear | 1179648 | 131934 | 0.1118 | 8 | 8 |
| block.2.attn.q_linear | linear | 1179648 | 1179648 | 1.0000 | 8 | 0 |
| block.2.attn.v_linear | linear | 1179648 | 1179648 | 1.0000 | 8 | 0 |
| block.2.mlp.fc1_linear | linear | 4718592 | 4718592 | 1.0000 | 8 | 0 |
| block.2.mlp.fc2_linear | linear | 4718592 | 89498 | 0.0190 | 8 | 8 |
| block.3.attn | attention | 25165824 | 1793874 | 0.0713 | 8 |  |
| block.3.attn.k_linear | linear | 1179648 | 1179648 | 1.0000 | 8 | 0 |
| block.3.attn.proj_linear | linear | 1179648 | 117866 | 0.0999 | 8 | 8 |
| block.3.attn.q_linear | linear | 1179648 | 1179648 | 1.0000 | 8 | 0 |
| block.3.attn.v_linear | linear | 1179648 | 1179648 | 1.0000 | 8 | 0 |
| block.3.mlp.fc1_linear | linear | 4718592 | 4718592 | 1.0000 | 8 | 0 |
| block.3.mlp.fc2_linear | linear | 4718592 | 136937 | 0.0290 | 8 | 8 |
| head | linear | 307200 | 307200 | 1.0000 | 8 | 0 |

</details>

### Spikformer — semi_structured

| Metric | Value |
|--------|-------|
| total_ops | 7116288 |
| effective_ops | 3577344 |
| density | 0.5027 |
| layers_converted | 24 |
| layers_skipped | 1 |

<details><summary>Per-layer stats (25 layers)</summary>

| Layer | total_ops | effective_ops | density | weight_density | activation_density | converted |
|-------| --- | --- | --- | --- | --- | --- |
| block.0.attn.k_linear | 147456 | 73728 | 0.5000 | 0.5000 | 1.0000 | True |
| block.0.attn.proj_linear | 147456 | 73728 | 0.5000 | 0.5000 | 1.0000 | True |
| block.0.attn.q_linear | 147456 | 73728 | 0.5000 | 0.5000 | 1.0000 | True |
| block.0.attn.v_linear | 147456 | 73728 | 0.5000 | 0.5000 | 1.0000 | True |
| block.0.mlp.fc1_linear | 589824 | 294912 | 0.5000 | 0.5000 | 1.0000 | True |
| block.0.mlp.fc2_linear | 589824 | 294912 | 0.5000 | 0.5000 | 1.0000 | True |
| block.1.attn.k_linear | 147456 | 73728 | 0.5000 | 0.5000 | 1.0000 | True |
| block.1.attn.proj_linear | 147456 | 73728 | 0.5000 | 0.5000 | 1.0000 | True |
| block.1.attn.q_linear | 147456 | 73728 | 0.5000 | 0.5000 | 1.0000 | True |
| block.1.attn.v_linear | 147456 | 73728 | 0.5000 | 0.5000 | 1.0000 | True |
| block.1.mlp.fc1_linear | 589824 | 294912 | 0.5000 | 0.5000 | 1.0000 | True |
| block.1.mlp.fc2_linear | 589824 | 294912 | 0.5000 | 0.5000 | 1.0000 | True |
| block.2.attn.k_linear | 147456 | 73728 | 0.5000 | 0.5000 | 1.0000 | True |
| block.2.attn.proj_linear | 147456 | 73728 | 0.5000 | 0.5000 | 1.0000 | True |
| block.2.attn.q_linear | 147456 | 73728 | 0.5000 | 0.5000 | 1.0000 | True |
| block.2.attn.v_linear | 147456 | 73728 | 0.5000 | 0.5000 | 1.0000 | True |
| block.2.mlp.fc1_linear | 589824 | 294912 | 0.5000 | 0.5000 | 1.0000 | True |
| block.2.mlp.fc2_linear | 589824 | 294912 | 0.5000 | 0.5000 | 1.0000 | True |
| block.3.attn.k_linear | 147456 | 73728 | 0.5000 | 0.5000 | 1.0000 | True |
| block.3.attn.proj_linear | 147456 | 73728 | 0.5000 | 0.5000 | 1.0000 | True |
| block.3.attn.q_linear | 147456 | 73728 | 0.5000 | 0.5000 | 1.0000 | True |
| block.3.attn.v_linear | 147456 | 73728 | 0.5000 | 0.5000 | 1.0000 | True |
| block.3.mlp.fc1_linear | 589824 | 294912 | 0.5000 | 0.5000 | 1.0000 | True |
| block.3.mlp.fc2_linear | 589824 | 294912 | 0.5000 | 0.5000 | 1.0000 | True |
| head | 38400 | 38400 | 1.0000 | 1.0000 | 1.0000 | False |

</details>

### Spikformer — triton_sparse

| Metric | Value |
|--------|-------|
| total_ops | 239855468544 |
| effective_ops | 239849177088 |
| density | 1.0000 |
| sparse_launches | 0 |
| dense_fallback | 40 |
| skipped_zero | 0 |
| attn_total_blocks | 393216 |
| attn_computed_blocks | 292374 |
| attn_block_density | 0.7435 |
| attn_sparse_calls | 32 |
| attn_dense_fallback | 0 |

### SEW-ResNet34 — torch_sparse

| Metric | Value |
|--------|-------|
| total_ops | 409600 |
| effective_ops | 409600 |
| density | 1.0000 |

<details><summary>Per-layer stats (1 layers)</summary>

| Layer | type | total_ops | effective_ops | density | total_calls | sparse_calls |
|-------| --- | --- | --- | --- | --- | --- |
| fc | linear | 409600 | 409600 | 1.0000 | 8 | 0 |

</details>

### SEW-ResNet34 — semi_structured

| Metric | Value |
|--------|-------|
| total_ops | 51200 |
| effective_ops | 51200 |
| density | 1.0000 |
| layers_converted | 0 |
| layers_skipped | 1 |

<details><summary>Per-layer stats (1 layers)</summary>

| Layer | total_ops | effective_ops | density | weight_density | activation_density | converted |
|-------| --- | --- | --- | --- | --- | --- |
| fc | 51200 | 51200 | 1.0000 | 1.0000 | 1.0000 | False |

</details>

### SEW-ResNet34 — triton_sparse

| Metric | Value |
|--------|-------|
| total_ops | 37352374272 |
| effective_ops | 30238179328 |
| density | 0.8095 |
| sparse_launches | 48 |
| dense_fallback | 240 |
| skipped_zero | 0 |
| attn_total_blocks | 0 |
| attn_computed_blocks | 0 |
| attn_block_density | 0.0000 |
| attn_sparse_calls | 0 |
| attn_dense_fallback | 0 |

### MS-ResNet34 — torch_sparse

| Metric | Value |
|--------|-------|
| total_ops | 409600 |
| effective_ops | 409600 |
| density | 1.0000 |

<details><summary>Per-layer stats (1 layers)</summary>

| Layer | type | total_ops | effective_ops | density | total_calls | sparse_calls |
|-------| --- | --- | --- | --- | --- | --- |
| fc | linear | 409600 | 409600 | 1.0000 | 8 | 0 |

</details>

### MS-ResNet34 — semi_structured

| Metric | Value |
|--------|-------|
| total_ops | 51200 |
| effective_ops | 51200 |
| density | 1.0000 |
| layers_converted | 0 |
| layers_skipped | 1 |

<details><summary>Per-layer stats (1 layers)</summary>

| Layer | total_ops | effective_ops | density | weight_density | activation_density | converted |
|-------| --- | --- | --- | --- | --- | --- |
| fc | 51200 | 51200 | 1.0000 | 1.0000 | 1.0000 | False |

</details>

### MS-ResNet34 — triton_sparse

| Metric | Value |
|--------|-------|
| total_ops | 38411436032 |
| effective_ops | 28479258624 |
| density | 0.7414 |
| sparse_launches | 80 |
| dense_fallback | 216 |
| skipped_zero | 0 |
| attn_total_blocks | 0 |
| attn_computed_blocks | 0 |
| attn_block_density | 0.0000 |
| attn_sparse_calls | 0 |
| attn_dense_fallback | 0 |

## Activation Sparsity Profiles

### Spikformer

Overall mean density: **0.2213** (sparsity: **0.7787**)

- Sparse layers (< 50%): 32 / 34
- Very sparse layers (< 15%): 15 / 34

| Type | Count | Mean Density |
|------|-------|-------------|
| Conv2d | 5 | 0.2600 |
| Linear | 25 | 0.2130 |
| SSA | 4 | 0.2247 |

<details><summary>Per-layer density (34 layers)</summary>

| Layer | Type | Density | Sparsity | Min | Max | Calls | Shape |
|-------|------|---------|----------|-----|-----|-------|-------|
| block.2.mlp.fc2_linear | Linear | 0.0189 | 0.9811 | 0.0187 | 0.0192 | 4 | 64x64x1536 |
| block.1.mlp.fc2_linear | Linear | 0.0202 | 0.9798 | 0.0196 | 0.0209 | 4 | 64x64x1536 |
| block.0.mlp.fc2_linear | Linear | 0.0211 | 0.9789 | 0.0205 | 0.0225 | 4 | 64x64x1536 |
| block.3.mlp.fc2_linear | Linear | 0.0289 | 0.9711 | 0.0261 | 0.0306 | 4 | 64x64x1536 |
| patch_embed.proj_conv2 | Conv2d | 0.0521 | 0.9479 | 0.0497 | 0.0549 | 4 | 64x96x32x32 |
| patch_embed.rpe_conv | Conv2d | 0.0734 | 0.9266 | 0.0714 | 0.0744 | 4 | 64x384x8x8 |
| patch_embed.proj_conv1 | Conv2d | 0.0743 | 0.9257 | 0.0573 | 0.0967 | 4 | 64x48x32x32 |
| block.3.attn.proj_linear | Linear | 0.0993 | 0.9007 | 0.0908 | 0.1061 | 4 | 64x64x384 |
| patch_embed.proj_conv3 | Conv2d | 0.1001 | 0.8999 | 0.0974 | 0.1032 | 4 | 64x192x16x16 |
| block.2.attn.proj_linear | Linear | 0.1110 | 0.8890 | 0.1080 | 0.1126 | 4 | 64x64x384 |
| block.0.attn | SSA | 0.1296 | 0.8704 | 0.1265 | 0.1320 | 4 | 4x16x64x384 |
| block.0.attn.k_linear | Linear | 0.1296 | 0.8704 | 0.1265 | 0.1320 | 4 | 64x64x384 |
| block.0.attn.q_linear | Linear | 0.1296 | 0.8704 | 0.1265 | 0.1320 | 4 | 64x64x384 |
| block.0.attn.v_linear | Linear | 0.1296 | 0.8704 | 0.1265 | 0.1320 | 4 | 64x64x384 |
| block.1.attn.proj_linear | Linear | 0.1329 | 0.8671 | 0.1302 | 0.1357 | 4 | 64x64x384 |
| block.0.mlp.fc1_linear | Linear | 0.1659 | 0.8341 | 0.1624 | 0.1702 | 4 | 64x64x384 |
| block.0.attn.proj_linear | Linear | 0.1681 | 0.8319 | 0.1623 | 0.1751 | 4 | 64x64x384 |
| block.1.attn | SSA | 0.1936 | 0.8064 | 0.1905 | 0.1983 | 4 | 4x16x64x384 |
| block.1.attn.k_linear | Linear | 0.1936 | 0.8064 | 0.1905 | 0.1983 | 4 | 64x64x384 |
| block.1.attn.q_linear | Linear | 0.1936 | 0.8064 | 0.1905 | 0.1983 | 4 | 64x64x384 |
| block.1.attn.v_linear | Linear | 0.1936 | 0.8064 | 0.1905 | 0.1983 | 4 | 64x64x384 |
| block.1.mlp.fc1_linear | Linear | 0.2259 | 0.7741 | 0.2230 | 0.2281 | 4 | 64x64x384 |
| block.2.attn | SSA | 0.2556 | 0.7444 | 0.2537 | 0.2573 | 4 | 4x16x64x384 |
| block.2.attn.k_linear | Linear | 0.2556 | 0.7444 | 0.2537 | 0.2573 | 4 | 64x64x384 |
| block.2.attn.q_linear | Linear | 0.2556 | 0.7444 | 0.2537 | 0.2573 | 4 | 64x64x384 |
| block.2.attn.v_linear | Linear | 0.2556 | 0.7444 | 0.2537 | 0.2573 | 4 | 64x64x384 |
| block.2.mlp.fc1_linear | Linear | 0.2863 | 0.7137 | 0.2843 | 0.2888 | 4 | 64x64x384 |
| block.3.attn | SSA | 0.3201 | 0.6799 | 0.3166 | 0.3253 | 4 | 4x16x64x384 |
| block.3.attn.k_linear | Linear | 0.3201 | 0.6799 | 0.3166 | 0.3253 | 4 | 64x64x384 |
| block.3.attn.q_linear | Linear | 0.3201 | 0.6799 | 0.3166 | 0.3253 | 4 | 64x64x384 |
| block.3.attn.v_linear | Linear | 0.3201 | 0.6799 | 0.3166 | 0.3253 | 4 | 64x64x384 |
| block.3.mlp.fc1_linear | Linear | 0.3509 | 0.6491 | 0.3441 | 0.3596 | 4 | 64x64x384 |
| head | Linear | 0.9999 | 0.0001 | 0.9998 | 1.0000 | 4 | 16x384 |
| patch_embed.proj_conv | Conv2d | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 4 | 64x3x32x32 |

</details>

### SEW-ResNet34

Overall mean density: **0.2001** (sparsity: **0.7999**)

- Sparse layers (< 50%): 35 / 37
- Very sparse layers (< 15%): 16 / 37

| Type | Count | Mean Density |
|------|-------|-------------|
| Conv2d | 36 | 0.1903 |
| Linear | 1 | 0.5523 |

<details><summary>Per-layer density (37 layers)</summary>

| Layer | Type | Density | Sparsity | Min | Max | Calls | Shape |
|-------|------|---------|----------|-----|-----|-------|-------|
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

</details>

### MS-ResNet34

Overall mean density: **0.2232** (sparsity: **0.7768**)

- Sparse layers (< 50%): 33 / 38
- Very sparse layers (< 15%): 26 / 38

| Type | Count | Mean Density |
|------|-------|-------------|
| Conv2d | 37 | 0.2184 |
| Linear | 1 | 0.4014 |

<details><summary>Per-layer density (38 layers)</summary>

| Layer | Type | Density | Sparsity | Min | Max | Calls | Shape |
|-------|------|---------|----------|-----|-----|-------|-------|
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

</details>
