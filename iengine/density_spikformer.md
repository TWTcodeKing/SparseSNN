# Activation Sparsity Profile — Spikformer

Dataset: **cifar100**  |  Samples: **64**

## Summary

| Metric | Value |
|--------|-------|
| Overall mean density | 0.2213 |
| Overall sparsity | 0.7787 |
| Total profiled layers | 34 |
| Sparse layers (density < 50%) | 32 |
| Very sparse layers (density < 15%) | 15 |

### Density by module type

| Type | Count | Mean Density |
|------|-------|-------------|
| Conv2d | 5 | 0.2600 |
| Linear | 25 | 0.2130 |
| SSA | 4 | 0.2247 |

## Per-Layer Density

| Layer | Type | Mean Density | Sparsity | Min | Max | Calls | Shape |
|-------|------|-------------|----------|-----|-----|-------|-------|
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
