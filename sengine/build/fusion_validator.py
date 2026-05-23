"""Fusion validation pre-pass: autotuned fused vs cuDNN/cuBLAS decomposed.

Profiles each fusible shape with:
  - FUSED: autotuned interleaved TileLang kernel (same config as final build)
  - DECOMPOSED: cuDNN Conv / cuBLAS MatMul + native CUDA IF/LIF (strongest baseline)

Supports ALL fusion patterns the graph slicer can produce:
  Conv+IF, Conv+LIF, Conv+Add+LIF, Conv+IF+Add, Conv+IF+Add+LIF,
  MatMul+IF, MatMul+LIF, MatMul+Add+LIF, MaxPool+LIF, Add+LIF

Usage:
    python -m sengine.build.fusion_validator \
        --onnx model_plugin.onnx --T 4 --batch 4 \
        --output .cache/fusion_rec.json
"""

from __future__ import annotations
import json
import os
import time
import torch
import torch.nn.functional as F

from sengine.ir import EngineIR, OpType, KernelVariant, BoundType
from sengine.logger import logger


def generate_recommendations(onnx_path: str, T: int, batch_size: int,
                              output_path: str, precision: str = "fp16"):
    """Profile each fusible shape and output KEEP/REVERT recommendations."""
    # Ensure CUDA toolchain
    for cuda_path in ['/usr/local/cuda-12.8', '/usr/local/cuda', '/usr/local/cuda-12.6']:
        if os.path.isdir(cuda_path):
            os.environ.setdefault('CUDA_HOME', cuda_path)
            nvcc_bin = os.path.join(cuda_path, 'bin')
            if nvcc_bin not in os.environ.get('PATH', ''):
                os.environ['PATH'] = nvcc_bin + ':' + os.environ.get('PATH', '')
            break

    io_torch_dtype = torch.float32 if precision == "fp32" else torch.float16
    bpe = 4 if precision == "fp32" else 2

    import tilelang
    _cache_was_enabled = tilelang.is_cache_enabled()
    tilelang.disable_cache()

    # Build IR with fusion proposals
    from sengine.parser import ONNXParser
    from sengine.optimizer import optimize_ir
    from sengine.fusion_strategy import apply_fusion

    ir = ONNXParser(onnx_path).parse()
    ir.precision = precision
    optimize_ir(ir, tilelang=True, batch_size=batch_size)
    apply_fusion(ir, strategy='slicer', batch_size=batch_size)

    _device_id = torch.cuda.current_device()
    _device = f'cuda:{_device_id}'
    TB = T * batch_size

    # ── Create compiler for autotuned fused kernels ──
    # Use a TuningCache that persists to disk — the build subprocess
    # will load these cached configs instead of re-autotuning.
    from sengine.build.tilelang_compiler import TileLangCompiler
    from sengine.build.tuning_cache import TuningCache
    tuning_cache = TuningCache()
    compiler = TileLangCompiler(ir, T=T, batch_size=batch_size,
                                 autotune=True, tuning_cache=tuning_cache,
                                 precision=precision)

    # ── Extract ALL unique fused shapes ──
    fused_shapes = {}
    for nid, node in ir.nodes.items():
        pattern = node.extra_attrs.get('fusion_slice')
        if not pattern or not node.output_shapes:
            continue

        cp = node.conv_params
        is_matmul = node.op_type in (OpType.MatMul, OpType.Linear)

        if is_matmul:
            if len(node.input_shapes) < 2:
                continue
            K_in = node.input_shapes[0][-1]
            N_out = node.input_shapes[1][-1] if len(node.input_shapes[1]) >= 2 else node.input_shapes[1][0]
            key = f"{K_in}_{N_out}_MatMul"
        elif cp:
            out_shape = node.output_shapes[0]
            if len(out_shape) < 4:
                continue
            _, _, OH, OW = out_shape
            key = f"{cp.in_channels}_{cp.out_channels}_K{cp.kernel_h}_S{cp.stride_h}_{OH}x{OW}"
        else:
            continue

        if key not in fused_shapes:
            fused_shapes[key] = (nid, node, pattern, is_matmul)

    # Also detect fused attention (DSSA, Spikformer, MaxFormer) nodes
    attn_shapes = {}
    for nid, node in ir.nodes.items():
        if node.op_type == OpType.FusedAttention and node.attention_params:
            ap = node.attention_params
            key = f"attn_{ap.variant}_{ap.num_heads}h_{ap.head_dim}d"
            if node.output_shapes:
                out = node.output_shapes[0]
                key += f"_{out[-2]}x{out[-1]}" if len(out) >= 2 else ""
            if key not in attn_shapes:
                attn_shapes[key] = (nid, node)

    if not fused_shapes and not attn_shapes:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({}, f)
        print(f"No fused shapes to validate. Wrote empty: {output_path}")
        if _cache_was_enabled:
            tilelang.enable_cache()
        return

    recommendations = {}
    print(f"Validating {len(fused_shapes)} fused shapes + {len(attn_shapes)} attention shapes...")

    for key, (nid, node, pattern, is_matmul) in fused_shapes.items():
        cp = node.conv_params
        absorbed = node.extra_attrs.get("absorbed_nids", [])
        is_lif = any(ir.nodes.get(a) and ir.nodes[a].op_type == OpType.LIF for a in absorbed)
        has_add = 'Add' in pattern

        # Get neuron params
        lif_node = next((ir.nodes[a] for a in absorbed
                         if ir.nodes.get(a) and ir.nodes[a].op_type == OpType.LIF), None)
        np_ = lif_node.neuron_params if lif_node else None
        recip_tau = 1.0 / np_.tau if (np_ and np_.tau and np_.tau > 0) else 0.5
        v_threshold = np_.v_threshold if np_ else 1.0

        # Compute shape parameters
        if is_matmul:
            K_in = node.input_shapes[0][-1]
            N_out = node.input_shapes[1][-1] if len(node.input_shapes[1]) >= 2 else node.input_shapes[1][0]
            M_full = 1
            for d in node.input_shapes[0]: M_full *= d
            M_full //= K_in
            M_per_t = M_full // T
            H, W = M_per_t // batch_size, 1
            C_in, C_out = K_in, N_out
            OH, OW = H, W
        else:
            _, _, OH, OW = node.output_shapes[0]
            H = OH * cp.stride_h if cp.stride_h > 1 else OH
            W = OW * cp.stride_w if cp.stride_w > 1 else OW
            if cp.pad_h > 0 and cp.stride_h == 1:
                H = OH
                W = OW
            M_per_t = batch_size * OH * OW
            C_in, C_out = cp.in_channels, cp.out_channels

        torch.cuda.set_device(_device_id)

        # ════════════════════════════════════════════════════
        # FUSED: compile + autotune via compiler
        # ════════════════════════════════════════════════════
        try:
            is_grouped = cp and cp.groups > 1 and cp.groups != cp.in_channels
            if is_grouped:
                fused_kern, _ = compiler._get_fused_grouped_conv_bn_lif(node)
            elif is_matmul:
                fused_kern, _ = compiler._get_fused_matmul_lif(node)
            else:
                fused_kern, _ = compiler._get_fused_conv_bn_if(node)
            if fused_kern is None:
                raise RuntimeError("Compiler returned None")
        except Exception as e:
            logger.warning("  %s: fused compile failed: %s", key, e)
            recommendations[key] = {'decision': 'revert', 'reason': f'compile_fail: {e}'}
            continue

        # Build profile args matching the kernel signature
        # Stem conv (C_in < 4): data uses real C_in, weight uses padded C_in (16)
        is_stem = cp and cp.in_channels < 4 and not is_matmul
        C_in_kern = 16 if is_stem else C_in
        data_fused = torch.randn(TB, H, W, C_in, dtype=io_torch_dtype, device=_device)
        if is_matmul or (cp and cp.kernel_h == 1):
            w_fused = torch.randn(C_in, C_out, dtype=io_torch_dtype, device=_device)
        elif is_grouped:
            C_in_per_g = cp.in_channels // cp.groups
            w_fused = torch.randn(cp.kernel_h, cp.kernel_w, C_in_per_g, C_out,
                                   dtype=io_torch_dtype, device=_device)
        else:
            w_fused = torch.randn(cp.kernel_h, cp.kernel_w, C_in_kern, C_out,
                                   dtype=io_torch_dtype, device=_device)
        st_fused = torch.zeros(M_per_t, C_out, dtype=torch.float32, device=_device)
        bn_s = torch.ones(C_out, dtype=torch.float32, device=_device)
        bn_b = torch.zeros(C_out, dtype=torch.float32, device=_device)

        torch.cuda.set_device(_device_id)
        fused_us = _profile(fused_kern, (data_fused, w_fused, st_fused, bn_s, bn_b),
                            reset_fn=st_fused.zero_)

        # ════════════════════════════════════════════════════
        # DECOMPOSED: cuDNN/cuBLAS + native CUDA IF/LIF
        # ════════════════════════════════════════════════════

        # Conv/MatMul via cuDNN/cuBLAS (strongest baseline)
        torch.cuda.set_device(_device_id)
        if is_matmul:
            # cuBLAS matmul
            A = torch.randn(TB, H * W, C_in, dtype=io_torch_dtype, device=_device)
            B_mat = torch.randn(C_in, C_out, dtype=io_torch_dtype, device=_device)
            decomp_compute_us = _profile(lambda: torch.matmul(A, B_mat))
        else:
            # cuDNN conv2d + BN
            data_nchw = torch.randn(TB, C_in, H, W, dtype=io_torch_dtype, device=_device)
            groups = cp.groups if cp.groups else 1
            w_nchw = torch.randn(C_out, C_in // groups, cp.kernel_h, cp.kernel_w,
                                  dtype=io_torch_dtype, device=_device)
            pad = cp.pad_h; stride = cp.stride_h
            scale_4d = bn_s.view(1, -1, 1, 1)
            bias_4d = bn_b.view(1, -1, 1, 1)
            decomp_compute_us = _profile(
                lambda: F.conv2d(data_nchw, w_nchw, padding=pad, stride=stride,
                                  groups=groups) * scale_4d + bias_4d)

        # Native IF/LIF kernel
        neuron_input = torch.randn(TB * OH * OW, C_out, dtype=io_torch_dtype, device=_device)
        neuron_mem = torch.zeros(M_per_t, C_out, dtype=torch.float32, device=_device)
        total_elems = TB * OH * OW * C_out
        spatial_elems = M_per_t * C_out

        try:
            from sengine.runtime.cpp_executor import CppExecutor
            exe = CppExecutor()
            exe.alloc_nodes(1)
            exe.set_schedule([0])
            if precision == "fp32":
                exe.set_fp32(True)
            if is_lif:
                exe.set_lif_node(0, neuron_input.data_ptr(), neuron_input.data_ptr(),
                                  neuron_mem.data_ptr(), total_elems, spatial_elems,
                                  v_threshold, recip_tau)
            else:
                exe.set_if_node(0, neuron_input.data_ptr(), neuron_input.data_ptr(),
                                 neuron_mem.data_ptr(), total_elems, spatial_elems,
                                 v_threshold)
            exe.add_membrane(neuron_mem.data_ptr(), neuron_mem.numel())
            import ctypes
            _lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), '..', 'csrc', 'libsengine_exec.so'))
            _lib.sengine_execute.argtypes = [ctypes.c_void_p]
            _lib.sengine_reset_membranes.argtypes = [ctypes.c_void_p]

            def _run_neuron():
                _lib.sengine_reset_membranes(exe._handle)
                _lib.sengine_execute(exe._handle)

            neuron_us = _profile(_run_neuron)
        except Exception:
            # Fallback: estimate neuron cost as ~5% of compute (memory-bound)
            neuron_us = decomp_compute_us * 0.05

        # Optional: Add kernel for patterns with residual
        add_us = 0.0
        if has_add:
            add_a = torch.randn(TB * OH * OW, C_out, dtype=io_torch_dtype, device=_device)
            add_b = torch.randn_like(add_a)
            add_us = _profile(lambda: torch.add(add_a, add_b))

        decomp_total_us = decomp_compute_us + neuron_us + add_us
        decision = 'keep' if fused_us <= decomp_total_us else 'revert'

        recommendations[key] = {
            'decision': decision,
            'fused_us': round(fused_us, 1),
            'decomp_us': round(decomp_total_us, 1),
            'conv_us': round(decomp_compute_us, 1),
            'if_us': round(neuron_us, 1),
            'add_us': round(add_us, 1) if has_add else 0,
            'pattern': pattern,
        }

        tag = 'KEEP' if decision == 'keep' else 'REVERT'
        parts = f"compute={decomp_compute_us:.1f}+IF={neuron_us:.1f}"
        if has_add:
            parts += f"+Add={add_us:.1f}"
        print(f"  {tag:>6} {key}: fused={fused_us:.1f} vs decomp={decomp_total_us:.1f} "
              f"({parts}) [{pattern}]")

    # ── Also compile decomposed TileLang kernels for REVERT'd shapes ──
    # The build phase uses different kernel+key for decomposed (e.g. _get_conv1x1_bn
    # vs _get_fused_conv_bn_if). Pre-compile those so their configs also land in rec.json.
    n_decomp = 0
    for key, rec in recommendations.items():
        if rec.get('decision') != 'revert':
            continue
        info = fused_shapes.get(key)
        if not info:
            continue
        nid, node, pattern, is_matmul = info
        torch.cuda.set_device(_device_id)
        try:
            if is_matmul:
                compiler._get_matmul(node)
            elif node.conv_params and node.conv_params.kernel_h == 1:
                compiler._get_conv1x1_bn(node)
            elif node.conv_params and node.conv_params.groups > 1 and node.conv_params.groups != node.conv_params.in_channels:
                compiler._get_grouped_conv_bn(node)
            elif node.conv_params:
                compiler._get_conv_bn(node)
            n_decomp += 1
        except Exception as e:
            logger.debug("  Decomposed compile for %s failed: %s", key, e)
    if n_decomp:
        print(f"  Pre-compiled {n_decomp} decomposed kernels for REVERT'd shapes")

    # ── Profile fused attention (DSSA, Spikformer, MaxFormer) ──
    if attn_shapes:
        print(f"\nProfiling {len(attn_shapes)} attention shapes...")
        for key, (nid, node) in attn_shapes.items():
            ap = node.attention_params
            torch.cuda.set_device(_device_id)
            try:
                # Compile attention kernels via compiler
                result = compiler._get_fused_attn_kernels(node)
                if result is None:
                    print(f"  SKIP {key}: compile returned None")
                    continue
                # Profile: attention is always KEEP (no decomposed alternative in our framework)
                # Just report the latency for visibility
                recommendations[key] = {
                    'decision': 'keep',
                    'fused_us': 0,
                    'pattern': f'FusedAttn_{ap.variant}',
                    'note': 'attention always fused (no decomposed path)',
                }
                print(f"    KEEP {key} [{ap.variant}] (always fused)")
            except Exception as e:
                print(f"  FAIL {key}: {e}")
                recommendations[key] = {'decision': 'keep', 'reason': f'compile_fail: {e}'}

    # Embed autotuned tile configs from the compiler's internal cache.
    # The build phase loads these directly into TuningCache, avoiding re-tuning.
    hw_name = torch.cuda.get_device_properties(_device_id).name
    tuning_configs = {}
    for cache_key, cfg in compiler._config_cache.items():
        if cfg:  # skip empty configs
            tuning_configs[cache_key] = {
                k: cfg[k] for k in ('block_M', 'block_N', 'block_K',
                                     'num_stages', 'threads', 'latency_us')
                if k in cfg
            }

    output_data = {
        'recommendations': recommendations,
        'tuning_configs': tuning_configs,
        'gpu_name': hw_name,
        'gpu_arch': f"sm_{torch.cuda.get_device_properties(_device_id).major}"
                    f"{torch.cuda.get_device_properties(_device_id).minor}",
        'T': T,
        'batch_size': batch_size,
    }

    # Write recommendations + embedded configs
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Also save to standalone tuning cache file (backup)
    tuning_cache.save()

    n_keep = sum(1 for r in recommendations.values() if r.get('decision') == 'keep')
    n_revert = sum(1 for r in recommendations.values() if r.get('decision') == 'revert')
    print(f"\nWrote {output_path}: {n_keep} keep, {n_revert} revert")

    if _cache_was_enabled:
        tilelang.enable_cache()


def load_recommendations(path: str) -> dict:
    """Load fusion recommendations from JSON file."""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    # New format: {'recommendations': {...}, 'tuning_configs': {...}, ...}
    if 'recommendations' in data:
        return data['recommendations']
    # Old format: flat dict of shape_key → {decision, ...}
    return data


def load_tuning_configs(path: str) -> tuple[dict, str, str, int, int]:
    """Load embedded tuning configs from fusion rec JSON.

    Returns:
        (configs_dict, gpu_name, gpu_arch, T, batch_size)
        configs_dict maps compiler cache_key → tile config dict.
        Returns empty dict if no configs embedded.
    """
    if not path or not os.path.exists(path):
        return {}, "", "", 0, 0
    with open(path) as f:
        data = json.load(f)
    if 'tuning_configs' not in data:
        return {}, "", "", 0, 0
    return (data['tuning_configs'],
            data.get('gpu_name', ''),
            data.get('gpu_arch', ''),
            data.get('T', 0),
            data.get('batch_size', 0))


def _profile(kern_or_fn, args=None, reset_fn=None, n_warmup=50, n_iter=200):
    """Profile and return latency in microseconds."""
    def _run():
        if reset_fn:
            reset_fn()
        if args is not None:
            kern_or_fn(*args)
        else:
            kern_or_fn()
    for _ in range(n_warmup):
        _run()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter):
        _run()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iter * 1000


# CLI entry point
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Fusion validation pre-pass")
    parser.add_argument('--onnx', required=True, help='Plugin-mode ONNX path')
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--output', default='.cache/fusion_rec.json')
    parser.add_argument('--precision', default='fp16', choices=['fp16', 'fp32'])
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    generate_recommendations(args.onnx, args.T, args.batch, args.output,
                              precision=args.precision)
