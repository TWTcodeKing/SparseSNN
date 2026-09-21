# Measured results

CSV files exported from the benchmark logs (`scripts/export_new_workload_results.py`).
All numbers were taken on an otherwise idle NVIDIA RTX 4090 (CUDA 12.8, T=4, FP16 unless
stated), September 2026.

| File | Content | Produced by |
|---|---|---|
| `latency_4090.csv` | sengine / TensorRT / torch.compile (Inductor) / ONNX Runtime latency and throughput for SNN-VGG16 on UT-HAR and SNN-VGG9 on UrbanSound8K, batch 4 to 32, FP16 and FP32 | `scripts/remeasure_new_workloads.sh`, `scripts/bench_extra_baselines.sh`, `scripts/analyze_new_workloads.sh` |
| `correctness_4090.csv` | engine-vs-PyTorch top-1 agreement, cosine similarity and test accuracy for sengine, sengine (Orin profile) and sengine_cpu | `scripts/verify_workloads.py --eval-acc` |
| `kernel_breakdown_4090.csv` | per-category TensorRT GPU time (nsys) and per-layer sengine kernel time | `scripts/analyze_kernel_latency.py` |

The GPU-utilization study (ncu metrics for sengine vs TensorRT on MaxFormer / SEW-ResNet-101 /
Spikformer) lives next to its code in `experiments/gpu_util/results/`.

Notes on the correctness file: the `strict_verdict` column applies the default thresholds
of `verify_workloads.py` (top-1 agreement >= 0.99 and mean cosine >= 0.99 against PyTorch FP16).
SNN-VGG9 on UrbanSound8K is a low-confidence model whose own FP16 run disagrees with FP32 on 3%
of samples (`torch_fp16_vs_fp32_agree`), so every engine "fails" the strict check there while
matching the FP16 test accuracy within 0.4 points. Use `--fp16-tolerance` to judge such models
relative to PyTorch's own FP16 noise.
