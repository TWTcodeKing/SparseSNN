"""
SEngine async logger — TensorRT-style build logging.

Format: [sengine] [LEVEL] [phase] message
Color:  DEBUG=grey, INFO=green, WARNING=yellow, ERROR=red

Uses QueueHandler + QueueListener for non-blocking async logging
so that nvcc compilation and CUDA ops are not stalled by I/O.

Usage:
    from sengine.logger import logger, set_log_level

    logger.info("Parsed %d nodes", n)
    logger.phase("STAKG", "Found %d interleaved groups", n)
    logger.conv("layer3.0.conv1", C_in=256, C_out=512, K=3, sparse=True)

    set_log_level("DEBUG")   # show everything
    set_log_level("WARNING") # quiet mode
"""

from __future__ import annotations

import logging
import logging.handlers
import queue
import sys
import time
import atexit
from typing import Any

import colorlog


# ──────────────────────────────────────────────────────────────
# Async queue setup
# ──────────────────────────────────────────────────────────────

_log_queue: queue.Queue = queue.Queue(-1)  # unbounded
_listener: logging.handlers.QueueListener | None = None


def _make_console_handler() -> logging.Handler:
    """Create a colorlog StreamHandler with TRT-style formatting."""
    fmt = (
        "%(log_color)s[sengine] [%(levelname)1.1s] "
        "%(message)s%(reset)s"
    )
    handler = colorlog.StreamHandler(sys.stderr)
    handler.setFormatter(colorlog.ColoredFormatter(
        fmt,
        log_colors={
            "DEBUG": "light_black",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    ))
    return handler


def _start_listener():
    """Start the async QueueListener (idempotent)."""
    global _listener
    if _listener is not None:
        return
    console = _make_console_handler()
    _listener = logging.handlers.QueueListener(
        _log_queue, console, respect_handler_level=True)
    _listener.start()
    atexit.register(_stop_listener)


def _stop_listener():
    """Flush and stop the async listener."""
    global _listener
    if _listener is not None:
        _listener.stop()
        _listener = None


# ──────────────────────────────────────────────────────────────
# SEngineLogger: extends stdlib Logger with domain helpers
# ──────────────────────────────────────────────────────────────

class SEngineLogger(logging.Logger):
    """Logger with SNN engine build domain methods."""

    def phase(self, phase_name: str, msg: str, *args, **kwargs):
        """Log a phase marker: [sengine] [I] [PHASE] message"""
        self.info("[%s] %s", phase_name, msg % args if args else msg, **kwargs)

    def conv(self, name: str, **params):
        """Log Conv layer details."""
        parts = [f"{name}:"]
        if "C_in" in params and "C_out" in params:
            parts.append(f"Conv({params['C_in']}→{params['C_out']},")
            K = params.get("K", params.get("kernel", "?"))
            parts.append(f"{K}x{K})")
        if params.get("sparse"):
            parts.append("[2:4 sparse]")
        if params.get("fused"):
            parts.append("[fused]")
        if "kernel_variant" in params:
            parts.append(f"→ {params['kernel_variant']}")
        self.info("  %s", " ".join(parts))

    def neuron(self, name: str, neuron_type: str, **params):
        """Log neuron layer details."""
        parts = [f"{name}: {neuron_type}"]
        if "tau" in params:
            parts.append(f"tau={params['tau']}")
        if "v_threshold" in params:
            parts.append(f"thr={params['v_threshold']}")
        self.info("  %s", " ".join(parts))

    def group(self, group_id: int, pattern: str, compute: str,
              memory: str = "", benefit_us: float = 0.0):
        """Log STAKG group decision."""
        parts = [f"Group {group_id}: {pattern}"]
        parts.append(f"compute=[{compute}]")
        if memory:
            parts.append(f"memory=[{memory}]")
        if benefit_us > 0:
            parts.append(f"benefit={benefit_us:.1f}µs")
        self.info("  %s", " ".join(parts))

    def mem(self, label: str, bytes_val: int | float):
        """Log memory allocation."""
        if bytes_val >= 1024 * 1024:
            self.info("  %s: %.1f MB", label, bytes_val / (1024 * 1024))
        elif bytes_val >= 1024:
            self.info("  %s: %.1f KB", label, bytes_val / 1024)
        else:
            self.info("  %s: %d B", label, int(bytes_val))

    def latency(self, label: str, ms: float):
        """Log latency measurement."""
        if ms < 1.0:
            self.info("  %s: %.1f µs", label, ms * 1000)
        else:
            self.info("  %s: %.3f ms", label, ms)

    def table(self, header: list[str], rows: list[list[Any]], phase: str = ""):
        """Log a formatted table (for layer summaries)."""
        if phase:
            self.info("[%s]", phase)
        # Compute column widths
        widths = [len(h) for h in header]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))
        # Format
        hdr = "  " + " │ ".join(h.ljust(widths[i]) for i, h in enumerate(header))
        sep = "  " + "─┼─".join("─" * w for w in widths)
        self.info(hdr)
        self.info(sep)
        for row in rows:
            line = "  " + " │ ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))
            self.info(line)


# ──────────────────────────────────────────────────────────────
# Module-level logger instance
# ──────────────────────────────────────────────────────────────

logging.setLoggerClass(SEngineLogger)
logger: SEngineLogger = logging.getLogger("sengine")  # type: ignore
logger.setLevel(logging.INFO)
logger.propagate = False

# Install async queue handler
_queue_handler = logging.handlers.QueueHandler(_log_queue)
logger.addHandler(_queue_handler)

# Start the listener (writes to stderr asynchronously)
_start_listener()


def set_log_level(level: str | int):
    """Set the sengine log level. Accepts 'DEBUG', 'INFO', 'WARNING', 'ERROR'."""
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)
