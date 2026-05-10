"""Fusion validation pre-pass: profile fused vs decomposed per shape.

Runs as a SEPARATE process before the main build. Outputs a JSON
recommendation file that the fusion strategy reads to decide which
shapes to fuse.

Usage:
    # Pre-pass: generate recommendations
    python -m sengine.build.fusion_validator \
        --onnx model_plugin.onnx --T 4 --batch 4 \
        --output .cache/fusion_rec.json

    # Main build: reads recommendations
    sengine.build(onnx, fusion='slicer', autotune=True,
                  fusion_rec='.cache/fusion_rec.json')
"""

from __future__ import annotations
import json
import os
import sys
import torch
import ctypes

from sengine.ir import EngineIR, OpType, KernelVariant, BoundType
from sengine.logger import logger


def generate_recommendations(onnx_path: str, T: int, batch_size: int,
                              output_path: str):
    """Profile each fusible shape and output recommendations.

    For each unique fused shape: compile both interleaved and decomposed,
    profile on GPU, decide which is faster.

    Writes JSON: {shape_key: "keep" | "revert", ...}
    """
    # Ensure CUDA 12.8 nvcc is on PATH (required for sm_89 / Ada Lovelace)
    for cuda_path in ['/usr/local/cuda-12.8', '/usr/local/cuda', '/usr/local/cuda-12.6']:
        if os.path.isdir(cuda_path):
            os.environ.setdefault('CUDA_HOME', cuda_path)
            nvcc_bin = os.path.join(cuda_path, 'bin')
            if nvcc_bin not in os.environ.get('PATH', ''):
                os.environ['PATH'] = nvcc_bin + ':' + os.environ.get('PATH', '')
            break

    # Disable TileLang disk cache during validation — the validator profiles
    # many tile configs per shape. Caching them would pollute the engine
    # builder's cache (which expects specific configs per shape key).
    import tilelang
    _cache_was_enabled = tilelang.is_cache_enabled()
    tilelang.disable_cache()

    # Build IR with all fusions proposed
    from sengine.parser import ONNXParser
    from sengine.optimizer import optimize_ir
    from sengine.fusion_strategy import apply_fusion

    ir = ONNXParser(onnx_path).parse()
    optimize_ir(ir, tilelang=True, batch_size=batch_size)
    apply_fusion(ir, strategy='slicer', batch_size=batch_size)

    # Pin CUDA device — TileLang/TVM compilation can reset current device
    _device_id = torch.cuda.current_device()
    _device = f'cuda:{_device_id}'

    TB = T * batch_size
    recommendations = {}

    # Collect unique fused shapes
    fused_shapes = {}
    for nid, node in ir.nodes.items():
        if node.assigned_kernel not in (KernelVariant.TileLangFusedConv1x1BNIF,
                                         KernelVariant.TileLangFusedConvBNIF,
                                         KernelVariant.TileLangFusedMatMulLIF):
            continue
        if not node.output_shapes:
            continue
        cp = node.conv_params
        is_matmul = node.op_type in (OpType.MatMul, OpType.Linear)

        if is_matmul:
            # MatMul: shapes from input_shapes
            if len(node.input_shapes) < 2:
                continue
            K_in = node.input_shapes[0][-1]
            N_out = node.input_shapes[1][-1] if len(node.input_shapes[1]) == 2 else node.input_shapes[1][0]
            key = f"{K_in}_{N_out}_MatMul"
        elif cp:
            _, _, OH, OW = node.output_shapes[0]
            is_1x1 = (cp.kernel_h == 1)
            key = f"{cp.in_channels}_{cp.out_channels}_K{cp.kernel_h}_S{cp.stride_h}_{OH}x{OW}"
        else:
            continue

        if key not in fused_shapes:
            fused_shapes[key] = (nid, node, is_matmul)

    if not fused_shapes:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({}, f)
        print(f"No fused shapes to validate. Wrote empty: {output_path}")
        return

    # Load native IF kernel
    _lib = ctypes.CDLL('sengine/csrc/libsengine_exec.so')
    from sengine.runtime.cpp_executor import CppExecutor
    from sengine.tuning.roofline import select_config_roofline

    print(f"Validating {len(fused_shapes)} fused shapes...")

    for key, (nid, node, is_matmul) in fused_shapes.items():
        cp = node.conv_params
        absorbed = node.extra_attrs.get("absorbed_nids", [])
        is_lif = any(ir.nodes.get(a) and ir.nodes[a].op_type == OpType.LIF for a in absorbed)

        # Get neuron params
        lif_node = next((ir.nodes[a] for a in absorbed
                         if ir.nodes.get(a) and ir.nodes[a].op_type == OpType.LIF), None)
        np_ = lif_node.neuron_params if lif_node else None
        recip_tau = 1.0 / np_.tau if (np_ and np_.tau and np_.tau > 0) else 0.5

        if is_matmul:
            # MatMul: (TB, spatial, K) @ (K, N)
            K_in = node.input_shapes[0][-1]
            N_out = node.input_shapes[1][-1] if len(node.input_shapes[1]) == 2 else node.input_shapes[1][0]
            M_full = 1
            for d in node.input_shapes[0]: M_full *= d
            M_full = M_full // K_in
            M_per_t = M_full // T
            spatial = M_per_t // batch_size
            H, W = spatial, 1
            C_in, C_out = K_in, N_out
            K_red = K_in
            is_1x1 = True       # 2D weight shape (C_in, C_out)
            use_1x1_tmpl = True  # 1x1 interleaved template (no stride)
        else:
            _, _, OH, OW = node.output_shapes[0]
            H = OH * cp.stride_h; W = OW * cp.stride_w
            M_per_t = batch_size * OH * OW
            C_in, C_out = cp.in_channels, cp.out_channels
            K_red = cp.kernel_h * cp.kernel_w * cp.in_channels
            is_1x1 = (cp.kernel_h == 1)

        # Compute output spatial dims
        if is_matmul or is_1x1:
            OH = (H + cp.stride_h - 1) // cp.stride_h if cp else H
            OW = (W + cp.stride_w - 1) // cp.stride_w if cp else W
        else:
            OH = (H + 2 * cp.pad_h - cp.dilation_h * (cp.kernel_h - 1) - 1) // cp.stride_h + 1
            OW = (W + 2 * cp.pad_w - cp.dilation_w * (cp.kernel_w - 1) - 1) // cp.stride_w + 1

        # Allocate all tensors on the pinned device
        torch.cuda.set_device(_device_id)
        data = torch.randn(TB, H, W, C_in, dtype=torch.float16, device=_device)
        if is_1x1 or is_matmul:
            w = torch.randn(C_in, C_out, dtype=torch.float16, device=_device)
        else:
            w = torch.randn(cp.kernel_h, cp.kernel_w, C_in, C_out,
                            dtype=torch.float16, device=_device)
        bn_s = torch.ones(C_out, dtype=torch.float32, device=_device)
        bn_b = torch.zeros(C_out, dtype=torch.float32, device=_device)
        st = torch.zeros(M_per_t, C_out, dtype=torch.float32, device=_device)

        # Compile + profile FUSED (interleaved)
        fused_S = cp.stride_h if cp else 1
        if is_1x1 or is_matmul:
            if is_lif:
                from sengine.kernels.interleaved_templates import conv1x1_bn_lif
                def _compile_fused(cfg):
                    return conv1x1_bn_lif(B=batch_size, C_in=C_in, H=H, W=W,
                                          F=C_out, T_steps=T, S=fused_S, recip_tau=recip_tau,
                                          **{k: cfg[k] for k in ('block_M','block_N','block_K','num_stages','threads')})
            else:
                from sengine.kernels.interleaved_templates import conv1x1_bn_if
                def _compile_fused(cfg):
                    return conv1x1_bn_if(B=batch_size, C_in=C_in, H=H, W=W,
                                          F=C_out, T_steps=T, S=fused_S,
                                          **{k: cfg[k] for k in ('block_M','block_N','block_K','num_stages','threads')})
        else:
            from sengine.kernels.conv2d_bn_if_t4 import conv2d_bn_if_interleaved_kernel
            def _compile_fused(cfg):
                return conv2d_bn_if_interleaved_kernel(
                    B=batch_size, C_in=C_in, H=H, W=W, F=C_out,
                    T_steps=T, K=cp.kernel_h, S=cp.stride_h, D=cp.dilation_h, P=cp.pad_h,
                    **{k: cfg[k] for k in ('block_M','block_N','block_K','num_stages','threads')})

        fused_cfg = select_config_roofline(M_per_t, K_red, C_out, T,
                                           compile_fn=_compile_fused,
                                           profile_args=(data, w, st, bn_s, bn_b),
                                           top_k=5, n_profile=100)
        torch.cuda.set_device(_device_id)  # restore after TileLang compilation
        fused_kern = _compile_fused(fused_cfg)
        fused_us = _profile(fused_kern, (data, w, st, bn_s, bn_b), reset_fn=st.zero_)

        # Compile + profile DECOMPOSED
        decomp_S = cp.stride_h if cp else 1
        if is_matmul or is_1x1:
            from sengine.kernels.conv2d_bn_if_t4 import conv1x1_bn_t4_kernel
            def _compile_decomp(cfg):
                return conv1x1_bn_t4_kernel(TB=TB, C_in=C_in, H=H, W=W,
                                             F=C_out, S=decomp_S,
                                             **{k: cfg[k] for k in ('block_M','block_N','block_K','num_stages','threads')})
        else:
            from sengine.kernels.conv2d_bn_if_t4 import conv2d_bn_t4_kernel
            def _compile_decomp(cfg):
                return conv2d_bn_t4_kernel(TB=TB, C_in=C_in, H=H, W=W,
                                            F=C_out, K=cp.kernel_h, S=cp.stride_h,
                                            D=cp.dilation_h, P=cp.pad_h,
                                            **{k: cfg[k] for k in ('block_M','block_N','block_K','num_stages','threads')})

        M_decomp = TB * OH * OW
        decomp_cfg = select_config_roofline(M_decomp, K_red, C_out, 1,
                                             compile_fn=_compile_decomp,
                                             profile_args=(data, w, bn_s, bn_b),
                                             top_k=3, n_profile=100)
        torch.cuda.set_device(_device_id)  # restore after TileLang compilation
        decomp_kern = _compile_decomp(decomp_cfg)
        decomp_conv_us = _profile(decomp_kern, (data, w, bn_s, bn_b))

        # Profile native C++ IF/LIF
        total_elems = TB * OH * OW * C_out
        spatial_elems = M_per_t * C_out
        if_in = torch.randn(TB, OH, OW, C_out, dtype=torch.float16, device=_device)
        if_out = torch.zeros_like(if_in)
        if_mem = torch.zeros(M_per_t, C_out, dtype=torch.float32, device=_device)
        exe_if = CppExecutor()
        exe_if.alloc_nodes(1); exe_if.set_schedule([0])
        exe_if.set_if_node(0, if_in.data_ptr(), if_out.data_ptr(),
                           if_mem.data_ptr(), total_elems, spatial_elems, 1.0)
        def _run_if():
            if_mem.zero_()
            _lib.sengine_execute(exe_if._handle)
        if_us = _profile(_run_if)

        decomp_total_us = decomp_conv_us + if_us
        decision = 'keep' if fused_us <= decomp_total_us else 'revert'

        recommendations[key] = {
            'decision': decision,
            'fused_us': round(fused_us, 1),
            'decomp_us': round(decomp_total_us, 1),
            'conv_us': round(decomp_conv_us, 1),
            'if_us': round(if_us, 1),
        }

        tag = 'KEEP' if decision == 'keep' else 'REVERT'
        print(f"  {tag:>6} {key}: fused={fused_us:.1f} vs decomp={decomp_total_us:.1f} "
              f"(conv={decomp_conv_us:.1f}+IF={if_us:.1f})")

    # Write recommendations
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(recommendations, f, indent=2)

    n_keep = sum(1 for r in recommendations.values() if r['decision'] == 'keep')
    n_revert = sum(1 for r in recommendations.values() if r['decision'] == 'revert')
    print(f"\nWrote {output_path}: {n_keep} keep, {n_revert} revert")

    # Restore TileLang cache state
    if _cache_was_enabled:
        tilelang.enable_cache()


def load_recommendations(path: str) -> dict:
    """Load fusion recommendations from JSON file."""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _profile(kern_or_fn, args=None, reset_fn=None, n_warmup=50, n_iter=200):
    """Profile and return latency in microseconds."""
    def _run():
        if reset_fn: reset_fn()
        if args is not None: kern_or_fn(*args)
        else: kern_or_fn()
    for _ in range(n_warmup): _run()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter): _run()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iter * 1000


# CLI entry point
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Fusion validation pre-pass")
    parser.add_argument('--onnx', required=True, help='Plugin-mode ONNX path')
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--output', default='.cache/fusion_rec.json')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.8')
    os.environ['PATH'] = '/usr/local/cuda-12.8/bin:' + os.environ.get('PATH', '')
    torch.cuda.set_device(args.gpu)

    generate_recommendations(args.onnx, args.T, args.batch, args.output)
