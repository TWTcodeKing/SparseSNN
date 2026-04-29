"""Build orchestrator: ONNX → optimized engine → .sengine file.

Single entry point for building and loading SNN inference engines.

Usage:
    # Build from ONNX (slow: parse + compile + autotune)
    engine = EngineBuilder("model.onnx", T=4, batch_size=16).build()
    engine.benchmark()
    engine.save("model.sengine")

    # Load from .sengine (fast: recompile with cached configs)
    engine = EngineBuilder.load("model.sengine")
    engine.benchmark()
"""

from __future__ import annotations

import time
from typing import Optional

from sengine.ir import EngineIR
from sengine.parser import ONNXParser
from sengine.optimizer import optimize_ir
from sengine.memory import plan_memory
from sengine.build.tilelang_compiler import TileLangCompiler
from sengine.build.tuning_cache import TuningCache
from sengine.build.schedule_builder import build_schedule, export_schedule_md
from sengine.build.sengine_io import save_sengine, load_sengine
from sengine.cuda_graph_runtime import CUDAGraphEngine
from sengine.logger import logger


class EngineBuilder:
    """Orchestrates the full build pipeline: ONNX → CUDAGraphEngine."""

    def __init__(self, onnx_path: str, T: int = 4, batch_size: int = 1):
        self.onnx_path = onnx_path
        self.T = T
        self.batch_size = batch_size
        self._ir: Optional[EngineIR] = None
        self._schedule: Optional[list[int]] = None
        self._engine: Optional[CUDAGraphEngine] = None

    def build(self, autotune: bool = False, capture_graph: bool = True) -> CUDAGraphEngine:
        """Full build pipeline.

        Args:
            autotune: If True, run autotuning sweep for tile configs.
                      If False, use heuristic defaults (fast, ~5% slower).
            capture_graph: If True, capture CUDA Graph after building.

        Returns:
            Ready-to-execute CUDAGraphEngine.
        """
        t0 = time.time()

        # 1. Parse ONNX
        logger.phase("BUILD", "Parsing ONNX: %s", self.onnx_path)
        parser = ONNXParser(self.onnx_path)
        self._ir = parser.parse()
        logger.phase("BUILD", "Parsed: %d nodes, T=%d", len(self._ir.nodes), self._ir.T)

        # 2. Optimize IR (BN fold + fusion + sparsity + shapes + bound classification)
        logger.phase("BUILD", "Optimizing IR (TileLang mode)")
        optimize_ir(self._ir, tilelang=True, batch_size=self.batch_size)

        # 3. Compile TileLang kernels
        logger.phase("BUILD", "Compiling TileLang kernels (TB=%d)", self.T * self.batch_size)
        tuning_cache = TuningCache() if autotune else None
        compiler = TileLangCompiler(
            self._ir, T=self.T, batch_size=self.batch_size,
            autotune=autotune, tuning_cache=tuning_cache,
        )
        kernels = compiler.compile_all()

        if tuning_cache:
            tuning_cache.save()

        # 4. Build BA-MTTS schedule
        logger.phase("BUILD", "Building BA-MTTS schedule")
        self._schedule = build_schedule(self._ir)

        # 5. Plan memory
        logger.phase("BUILD", "Planning memory")
        plan_memory(self._ir, execution_order=self._schedule)

        # 6. Build CUDA Graph engine
        logger.phase("BUILD", "Building CUDA Graph engine")
        self._engine = CUDAGraphEngine()
        self._engine.build(self._ir, kernels, self._schedule,
                          T=self.T, batch_size=self.batch_size)

        # 7. Capture CUDA Graph
        if capture_graph:
            logger.phase("BUILD", "Capturing CUDA Graph")
            self._engine.capture_graph()

        elapsed = time.time() - t0
        logger.phase("BUILD", "Done in %.1fs", elapsed)
        return self._engine

    def save(self, path: str):
        """Save the built engine to a .sengine file."""
        if self._ir is None or self._schedule is None:
            raise RuntimeError("Must call build() before save()")
        save_sengine(path, self._ir, self._schedule, self.T, self.batch_size)

    def export_schedule(self, path: str, model_name: str = ""):
        """Export the BA-MTTS execution order to a Markdown file."""
        if self._ir is None or self._schedule is None:
            raise RuntimeError("Must call build() before export_schedule()")
        export_schedule_md(self._ir, self._schedule, path, model_name)

    @staticmethod
    def load(path: str, capture_graph: bool = True) -> CUDAGraphEngine:
        """Load a .sengine file and reconstruct the engine.

        This is fast: no ONNX parsing, no autotuning. Just recompile
        kernels from cached tile configs and capture the CUDA Graph.
        """
        t0 = time.time()

        # 1. Load IR + schedule + configs from .sengine
        ir, schedule, T, batch_size = load_sengine(path)

        # 2. Recompile TileLang kernels with cached configs
        logger.phase("LOAD", "Recompiling kernels from cached configs")
        compiler = TileLangCompiler(ir, T=T, batch_size=batch_size, autotune=False)
        kernels = compiler.compile_all()

        # 3. Build engine
        engine = CUDAGraphEngine()
        engine.build(ir, kernels, schedule, T=T, batch_size=batch_size)

        # 4. Capture CUDA Graph
        if capture_graph:
            logger.phase("LOAD", "Capturing CUDA Graph")
            engine.capture_graph()

        elapsed = time.time() - t0
        logger.phase("LOAD", "Loaded in %.1fs from %s", elapsed, path)
        return engine
