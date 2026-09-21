"""TVM kernel compiler for sengine_cpu (optional backend).

Compiles TVM TE kernels to standalone .so files. Handles shape-key
deduplication: identical layer shapes share one compiled .so.

Compilation runs in a separate TVM interpreter (see sengine_cpu/tvm_env.py:
$SENGINE_CPU_TVM_PYTHON or sengine_cpu/.tvm-env). Compiled .so files are
standalone — no TVM dependency at runtime.

Self-contained — no imports from sengine/.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import json
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from sengine_cpu.ir import (
    EngineIR, Node, OpType, NeuronType, CPUKernelVariant, BoundType,
)
from sengine_cpu.logger import log
from sengine_cpu.tvm_env import tvm_python, require_tvm_python

# Default build directory for compiled .so files
DEFAULT_BUILD_DIR = ".sengine_cpu.cache"

# Fused kernel variants that need TVM compilation
_TVM_FUSED_VARIANTS = {
    CPUKernelVariant.TVMConvBNIF,
    CPUKernelVariant.TVMConvBNLIF,
    CPUKernelVariant.TVMConv1x1BNIF,
    CPUKernelVariant.TVMConv1x1BNLIF,
    CPUKernelVariant.TVMConvBNAddLIF,
    CPUKernelVariant.TVMMatMulBNIF,
    CPUKernelVariant.TVMMatMulBNLIF,
    CPUKernelVariant.TVMAddLIF,
    CPUKernelVariant.TVMPoolLIF,
}

_TVM_DECOMPOSED_VARIANTS = {
    CPUKernelVariant.TVMConvBN,
    CPUKernelVariant.TVMConv1x1BN,
    CPUKernelVariant.TVMLinearBN,
    CPUKernelVariant.TVMMatMul,
}


def _shape_key(node: Node, ir: EngineIR) -> str:
    """Generate a unique key for this node's kernel shape."""
    parts = [node.assigned_kernel.name]
    if node.conv_params:
        cp = node.conv_params
        parts.append(f"C{cp.in_channels}_F{cp.out_channels}"
                     f"_K{cp.kernel_h}x{cp.kernel_w}"
                     f"_S{cp.stride_h}_G{cp.groups}")
    if node.output_shapes:
        parts.append(f"out{'x'.join(str(d) for d in node.output_shapes[0])}")
    if node.input_shapes:
        parts.append(f"in{'x'.join(str(d) for d in node.input_shapes[0])}")
    if node.neuron_params:
        np_ = node.neuron_params
        parts.append(f"n{np_.neuron_type.name}_tau{np_.tau:.2f}_th{np_.v_threshold:.2f}")
    return "_".join(parts)


def _so_path(build_dir: str, shape_key: str) -> str:
    """Deterministic .so path from shape key."""
    h = hashlib.md5(shape_key.encode()).hexdigest()[:12]
    return os.path.join(build_dir, f"kern_{h}.so")


class TVMCompiler:
    """Compile TVM kernels for all nodes in the IR.

    Deduplicates: nodes with identical shapes share one .so.
    Compiles via subprocess to avoid TVM/tilelang conflicts.
    """

    def __init__(self, ir: EngineIR, T: int = 4, batch_size: int = 1,
                 build_dir: str = DEFAULT_BUILD_DIR,
                 target: str | dict | None = None):
        self.ir = ir
        self.T = T
        self.batch_size = batch_size
        self.build_dir = os.path.abspath(build_dir)
        self.target = target or {"kind": "llvm"}
        os.makedirs(self.build_dir, exist_ok=True)

    def compile_all(self) -> dict[int, str]:
        """Compile all TVM kernels. Returns {node_id: so_path}.

        Nodes with non-TVM kernels (Native*, ZeroCost, Skip) are not compiled.
        Raises RuntimeError if TVM kernels are needed but no TVM interpreter
        is configured.
        """
        # Group nodes by shape key for deduplication
        shape_groups: dict[str, list[int]] = {}
        for nid in self.ir.topo_order:
            node = self.ir.nodes.get(nid)
            if node is None:
                continue
            if node.assigned_kernel not in (_TVM_FUSED_VARIANTS | _TVM_DECOMPOSED_VARIANTS):
                continue
            key = _shape_key(node, self.ir)
            shape_groups.setdefault(key, []).append(nid)

        log.info("TVM compiler: %d unique shapes from %d TVM nodes",
                 len(shape_groups),
                 sum(len(nids) for nids in shape_groups.values()))
        if shape_groups:
            require_tvm_python()

        # Compile each unique shape
        kernel_map: dict[int, str] = {}
        compiled = 0
        cached = 0

        for key, nids in shape_groups.items():
            so_file = _so_path(self.build_dir, key)

            if os.path.exists(so_file):
                cached += 1
            else:
                node = self.ir.nodes[nids[0]]
                success = self._compile_kernel(node, so_file)
                if not success:
                    log.warning("Failed to compile kernel for shape %s", key)
                    continue
                compiled += 1

            for nid in nids:
                kernel_map[nid] = so_file

        log.info("TVM compiler: %d compiled, %d cached, %d total .so files",
                 compiled, cached, compiled + cached)
        return kernel_map

    def _compile_kernel(self, node: Node, so_path: str) -> bool:
        """Compile a single kernel via TVM subprocess."""
        kv = node.assigned_kernel

        # Determine kernel builder function and args
        if kv in (CPUKernelVariant.TVMConvBNIF, CPUKernelVariant.TVMConv1x1BNIF,
                  CPUKernelVariant.TVMConvBN, CPUKernelVariant.TVMConv1x1BN):
            return self._compile_conv_bn_if(node, so_path)
        elif kv in (CPUKernelVariant.TVMConvBNLIF, CPUKernelVariant.TVMConv1x1BNLIF):
            return self._compile_conv_bn_lif(node, so_path)
        elif kv == CPUKernelVariant.TVMAddLIF:
            return self._compile_add_lif(node, so_path)
        else:
            log.warning("No compiler for kernel variant %s", kv.name)
            return False

    def _get_conv_dims(self, node: Node) -> tuple[int, int, int]:
        """Extract (M, C_in, F) from a Conv2d node."""
        cp = node.conv_params
        if not cp or not node.output_shapes:
            return 0, 0, 0
        out_shape = node.output_shapes[0]
        # out_shape is NCHW: (TB, C_out, OH, OW)
        TB = out_shape[0] if len(out_shape) >= 1 else self.T * self.batch_size
        C_out = cp.out_channels
        OH = out_shape[2] if len(out_shape) >= 3 else 1
        OW = out_shape[3] if len(out_shape) >= 4 else 1
        M = (TB // self.T) * OH * OW  # spatial per timestep
        return M, cp.in_channels, C_out

    def _get_neuron_params(self, node: Node) -> dict:
        """Get neuron params from the fused neuron (via fusion group)."""
        fg_id = node.fusion_group_id
        if fg_id < 0 or fg_id >= len(self.ir.fusion_groups):
            return {"v_threshold": 1.0, "v_reset": 0.0, "tau": 2.0}
        fg = self.ir.fusion_groups[fg_id]
        neuron = self.ir.nodes.get(fg.neuron_node_id)
        if neuron and neuron.neuron_params:
            np_ = neuron.neuron_params
            return {"v_threshold": np_.v_threshold, "v_reset": np_.v_reset,
                    "tau": np_.tau}
        return {"v_threshold": 1.0, "v_reset": 0.0, "tau": 2.0}

    def _compile_conv_bn_if(self, node: Node, so_path: str) -> bool:
        """Compile Conv1x1+BN+IF kernel."""
        M, C_in, F = self._get_conv_dims(node)
        if M == 0:
            return False
        nparams = self._get_neuron_params(node)
        return self._run_tvm_subprocess(
            "conv_bn_if", "make_conv1x1_bn_if",
            {"M": M, "C_in": C_in, "F": F,
             "v_threshold": nparams["v_threshold"],
             "v_reset": nparams["v_reset"]},
            so_path)

    def _compile_conv_bn_lif(self, node: Node, so_path: str) -> bool:
        M, C_in, F = self._get_conv_dims(node)
        if M == 0:
            return False
        nparams = self._get_neuron_params(node)
        return self._run_tvm_subprocess(
            "conv_bn_if", "make_conv1x1_bn_lif",
            {"M": M, "C_in": C_in, "F": F,
             "v_threshold": nparams["v_threshold"],
             "v_reset": nparams["v_reset"],
             "tau": nparams["tau"]},
            so_path)

    def _compile_add_lif(self, node: Node, so_path: str) -> bool:
        if not node.output_shapes:
            return False
        shape = node.output_shapes[0]
        total = 1
        for d in shape:
            total *= d
        M = total // self.T  # per-timestep size
        F_dim = shape[-1] if len(shape) >= 2 else total
        M_dim = M // F_dim if F_dim > 0 else M
        nparams = self._get_neuron_params(node)
        return self._run_tvm_subprocess(
            "add_lif", "make_add_lif",
            {"M": M_dim, "F": F_dim,
             "v_threshold": nparams["v_threshold"],
             "v_reset": nparams["v_reset"],
             "tau": nparams["tau"]},
            so_path)

    def _run_tvm_subprocess(self, module_name: str, func_name: str,
                            kwargs: dict, so_path: str) -> bool:
        """Run TVM compilation in a subprocess (avoids tilelang conflicts)."""
        # Generate a small script that imports the kernel builder and compiles.
        # The TVM interpreter imports sengine_cpu.kernels.<module> from the
        # repo root (put on PYTHONPATH below).
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        repo_root = os.path.dirname(pkg_dir)

        script = f"""
import importlib
kmod = importlib.import_module("sengine_cpu.kernels.{module_name}")
target = {json.dumps(self.target) if isinstance(self.target, dict) else repr(self.target)}
lib = kmod.{func_name}(**{repr(kwargs)}, target_str=target)
kmod.export_kernel(lib, "{so_path}")
print("OK")
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            result = subprocess.run(
                [tvm_python(), script_path],
                capture_output=True, text=True, timeout=120,
                env={**os.environ, "PYTHONPATH": repo_root},
            )
            if result.returncode != 0:
                log.warning("TVM compile failed for %s: %s",
                            func_name, result.stderr[-500:] if result.stderr else "unknown")
                return False
            return "OK" in result.stdout
        except subprocess.TimeoutExpired:
            log.warning("TVM compile timed out for %s", func_name)
            return False
        finally:
            os.unlink(script_path)
