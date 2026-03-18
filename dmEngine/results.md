# Weight-Sparse Inference Benchmark

## Configuration
- Input: ImageNet (3x224x224), batch_size=32
- Device: cuda:3
- Warmup: 10, Iterations: 100

## Results

| Model | Backend | Sparsity | Dense (ms) | Sparse (ms) | Speedup |
|-------|---------|----------|-----------|-------------|---------|
| resnet18 | torch_csr | 0.50 | 6.81 | 78.19 | 0.09x |
| resnet18 | torch_csr | 0.70 | 6.79 | 56.02 | 0.12x |
| resnet18 | torch_csr | 0.90 | 6.68 | 35.05 | 0.19x |
| resnet18 | torch_csr | 0.95 | 6.77 | 30.11 | 0.22x |
| resnet18 | semi_structured | 0.50 | 6.73 | 6.89 | 0.98x |
| resnet18 | triton_wsparse | 0.50 | 6.83 | 98.20 | 0.07x |
| resnet18 | triton_wsparse | 0.70 | 6.79 | 79.12 | 0.09x |
| resnet18 | triton_wsparse | 0.90 | 6.88 | 58.76 | 0.12x |
| resnet18 | triton_wsparse | 0.95 | 6.75 | 54.62 | 0.12x |
| resnet50 | torch_csr | 0.50 | 24.49 | 181.93 | 0.13x |
| resnet50 | torch_csr | 0.70 | 24.74 | 136.87 | 0.18x |
| resnet50 | torch_csr | 0.90 | 24.76 | 90.26 | 0.27x |
| resnet50 | torch_csr | 0.95 | 24.69 | 80.85 | 0.31x |
| resnet50 | semi_structured | 0.50 | 24.68 | 24.78 | 1.00x |
| resnet50 | triton_wsparse | 0.50 | 24.73 | 202.06 | 0.12x |
| resnet50 | triton_wsparse | 0.70 | 24.72 | 162.65 | 0.15x |
| resnet50 | triton_wsparse | 0.90 | 24.71 | 122.79 | 0.20x |
| resnet50 | triton_wsparse | 0.95 | 24.82 | 114.28 | 0.22x |

## Per-Backend Summary

| Backend | Avg Speedup | Num Runs |
|---------|-------------|----------|
| semi_structured | 0.99x | 2 |
| torch_csr | 0.19x | 8 |
| triton_wsparse | 0.14x | 8 |
