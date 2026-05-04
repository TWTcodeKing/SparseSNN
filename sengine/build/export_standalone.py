"""Export TileLang kernels as standalone .so files for C++ executor.

Each exported .so has:
  - init()  — sets dynamic shared memory attributes
  - call()  — launches the CUDA kernel with raw pointers + stream

Zero TVM dependency at runtime. The .so is self-contained.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from typing import Optional

from sengine.logger import logger


# TileLang include paths (resolved once)
_TL_INCLUDE = None
_CUTLASS_INCLUDE = None


def _get_include_paths():
    global _TL_INCLUDE, _CUTLASS_INCLUDE
    if _TL_INCLUDE is not None:
        return _TL_INCLUDE, _CUTLASS_INCLUDE
    import tilelang
    tl_path = os.path.dirname(tilelang.__file__)
    _TL_INCLUDE = os.path.join(tl_path, 'src')
    _CUTLASS_INCLUDE = os.path.join(tl_path, '3rdparty', 'cutlass', 'include')
    return _TL_INCLUDE, _CUTLASS_INCLUDE


def export_kernel_so(kern, output_path: str,
                     nvcc: str | None = None,
                     arch: str = 'sm_89') -> str:
    """Export a TileLang JITKernel as a standalone .so.

    Args:
        kern: TileLang JITKernel object (from @tilelang.jit)
        output_path: Where to write the .so file
        nvcc: Path to nvcc compiler
        arch: CUDA architecture (e.g., 'sm_89' for RTX 4090)

    Returns:
        Path to the compiled .so file
    """
    from tilelang import tvm
    from tilelang.jit.adapter.wrapper import TLCUDASourceWrapper

    adapter = kern.adapter
    device_src = adapter.get_device_source()
    target = adapter.target

    with tvm.target.Target(target):
        wrapper = TLCUDASourceWrapper(
            scheduled_ir_module=adapter.ir_module,
            source=device_src,
            target=target,
            pass_configs=adapter.pass_configs,
        )

    lib_code = wrapper.lib_code

    # Add common header for half_t type if not present
    if '#include <cuda_fp16.h>' not in lib_code:
        lib_code = '#include <cuda_fp16.h>\n' + lib_code

    # Write to temp .cu file and compile
    tl_inc, cutlass_inc = _get_include_paths()

    if nvcc is None:
        for p in ['/usr/local/cuda/bin/nvcc', '/usr/local/cuda-12.8/bin/nvcc']:
            if os.path.exists(p):
                nvcc = p
                break
        else:
            nvcc = 'nvcc'

    cu_path = output_path.replace('.so', '.cu')
    with open(cu_path, 'w') as f:
        f.write(lib_code)

    cmd = [
        nvcc, '-O3', '--use_fast_math',
        '-shared', '-Xcompiler', '-fPIC',
        f'-gencode=arch=compute_{arch.replace("sm_", "")},code={arch}',
        f'-I{tl_inc}', f'-I{cutlass_inc}',
        '-o', output_path, cu_path,
        '-lcudart',
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("nvcc failed for %s:\n%s", output_path, result.stderr)
        raise RuntimeError(f"nvcc compilation failed: {result.stderr[:500]}")

    # Clean up .cu source
    os.remove(cu_path)
    return output_path


def export_all_kernels(kernels: dict, ir, build_dir: str,
                       nvcc: str | None = None,
                       arch: str = 'sm_89') -> dict[int, str]:
    """Export all TileLang kernels in a schedule to standalone .so files.

    Args:
        kernels: dict[node_id → JITKernel] from TileLangCompiler.compile_all()
        ir: EngineIR (for node info)
        build_dir: Directory to write .so files
        nvcc: Path to nvcc
        arch: CUDA architecture

    Returns:
        dict[node_id → .so path] for all TileLang kernel nodes
    """
    from sengine.ir import KernelVariant

    os.makedirs(build_dir, exist_ok=True)

    # Group kernels by shape key to avoid duplicate compilation
    # Kernels with the same object identity share the same compiled kernel
    kern_to_so: dict[int, str] = {}  # id(kern_obj) → .so path
    nid_to_so: dict[int, str] = {}   # node_id → .so path

    tilelang_variants = {
        KernelVariant.TileLangConvBN,
        KernelVariant.TileLangConv1x1BN,
        KernelVariant.TileLangStemConvBN,
        KernelVariant.TileLangFusedConvBNIF,
        KernelVariant.TileLangFusedConv1x1BNIF,
        KernelVariant.TileLangLinearBN,
        KernelVariant.TileLangLinearBNLIF,
        KernelVariant.TileLangDWConvBN,
        KernelVariant.TileLangFusedDWConvBNIF,
        KernelVariant.TileLangGroupedConvBN,
        KernelVariant.TileLangMatMulScale,
        KernelVariant.TileLangFusedMatMulLIF,
    }

    fused_attn_variants = {
        KernelVariant.FusedSpikformerAttn,
        KernelVariant.FusedMaxformerAttn,
        KernelVariant.FusedDSSAAttn,
        KernelVariant.FusedTokenQKAttn,
    }

    def _export_one(kern_obj, so_path):
        kid = id(kern_obj)
        if kid in kern_to_so:
            return kern_to_so[kid]
        if os.path.exists(so_path):
            kern_to_so[kid] = so_path
            return so_path
        export_kernel_so(kern_obj, so_path, nvcc=nvcc, arch=arch)
        kern_to_so[kid] = so_path
        return so_path

    count = 0
    for nid, kern in kernels.items():
        if kern is None:
            continue
        node = ir.nodes.get(nid)
        if node is None:
            continue

        # Fused attention: kern is (gemm1, gemm2) tuple
        if node.assigned_kernel in fused_attn_variants and isinstance(kern, tuple):
            gemm1, gemm2 = kern
            try:
                so1 = _export_one(gemm1, os.path.join(build_dir, f'kern_{nid}_g1.so'))
                so2 = _export_one(gemm2, os.path.join(build_dir, f'kern_{nid}_g2.so'))
                nid_to_so[nid] = (so1, so2)  # tuple of paths
                count += 2
            except Exception as e:
                logger.error("Failed to export attention kernel for node %d: %s", nid, e)
            continue

        if node.assigned_kernel not in tilelang_variants:
            continue

        kid = id(kern)
        if kid in kern_to_so:
            nid_to_so[nid] = kern_to_so[kid]
            continue

        so_path = os.path.join(build_dir, f'kern_{nid}.so')
        if os.path.exists(so_path):
            kern_to_so[kid] = so_path
            nid_to_so[nid] = so_path
            continue
        try:
            export_kernel_so(kern, so_path, nvcc=nvcc, arch=arch)
            kern_to_so[kid] = so_path
            nid_to_so[nid] = so_path
            count += 1
        except Exception as e:
            logger.error("Failed to export kernel for node %d: %s", nid, e)

    cached = len(nid_to_so) - count
    logger.phase("EXPORT", "Exported %d .so files (%d cached) to %s", count, cached, build_dir)
    return nid_to_so


def get_call_signature(kern) -> list[tuple[str, str]]:
    """Get the call() function parameter types and names.

    Returns list of (c_type, name) pairs, e.g.:
    [('half_t*', 'data'), ('half_t*', 'weight'), ('float*', 'scale'), ('float*', 'bias'), ('half_t*', 'output')]
    """
    pf = kern.adapter.prim_func
    result = []
    for p in pf.params:
        if p in pf.buffer_map:
            buf = pf.buffer_map[p]
            dtype = str(buf.dtype)
            c_type = {'float16': 'half_t*', 'float32': 'float*'}[dtype]
            result.append((c_type, buf.data.name))
    return result
