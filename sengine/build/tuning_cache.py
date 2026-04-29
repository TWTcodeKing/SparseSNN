"""JSON-backed autotuning cache for TileLang kernel tile configs.

Stores per-layer optimal configs so rebuilds skip the autotuning sweep.
Cache is keyed by (shape_key, gpu_name, gpu_arch, T, B) for correct
matching across different hardware, temporal steps, and batch sizes.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


_DEFAULT_CACHE_PATH = os.path.expanduser("~/.cache/sengine/tuning_cache.json")


@dataclass
class TuningEntry:
    shape_key: str
    gpu_name: str
    gpu_arch: str = ""       # e.g., "sm_89"
    T: int = 0
    B: int = 0
    block_M: int = 0
    block_N: int = 0
    block_K: int = 0
    num_stages: int = 0
    threads: int = 0
    latency_us: float = 0.0


def _detect_gpu_arch() -> str:
    """Detect GPU compute capability string."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return f"sm_{props.major}{props.minor}"
    except Exception:
        pass
    return ""


class TuningCache:
    """JSON-backed cache of autotuned kernel configs."""

    def __init__(self, cache_path: str = _DEFAULT_CACHE_PATH):
        self.path = Path(cache_path)
        self._entries: dict[str, TuningEntry] = {}
        self._load()

    def _cache_key(self, shape_key: str, gpu_name: str,
                   gpu_arch: str = "", T: int = 0, B: int = 0) -> str:
        return f"{shape_key}|{gpu_name}|{gpu_arch}|T{T}|B{B}"

    def _load(self):
        if not self.path.exists():
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            for item in data:
                # Handle both old and new format
                entry = TuningEntry(
                    shape_key=item['shape_key'],
                    gpu_name=item['gpu_name'],
                    gpu_arch=item.get('gpu_arch', ''),
                    T=item.get('T', 0),
                    B=item.get('B', 0),
                    block_M=item['block_M'],
                    block_N=item['block_N'],
                    block_K=item['block_K'],
                    num_stages=item['num_stages'],
                    threads=item['threads'],
                    latency_us=item['latency_us'],
                )
                key = self._cache_key(entry.shape_key, entry.gpu_name,
                                      entry.gpu_arch, entry.T, entry.B)
                self._entries[key] = entry
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    def get(self, shape_key: str, gpu_name: str,
            gpu_arch: str = "", T: int = 0, B: int = 0) -> Optional[dict]:
        """Look up cached config. Falls back to less-specific keys."""
        # Try exact match first
        key = self._cache_key(shape_key, gpu_name, gpu_arch, T, B)
        entry = self._entries.get(key)
        if entry is not None:
            return self._entry_to_dict(entry)

        # Fallback: match without T/B (config is often stable across T/B)
        key = self._cache_key(shape_key, gpu_name, gpu_arch, 0, 0)
        entry = self._entries.get(key)
        if entry is not None:
            return self._entry_to_dict(entry)

        # Fallback: match without arch (same GPU name is usually same arch)
        key = self._cache_key(shape_key, gpu_name, "", 0, 0)
        entry = self._entries.get(key)
        if entry is not None:
            return self._entry_to_dict(entry)

        return None

    def put(self, shape_key: str, gpu_name: str, config: dict,
            gpu_arch: str = "", T: int = 0, B: int = 0):
        """Store a tuned config."""
        entry = TuningEntry(
            shape_key=shape_key, gpu_name=gpu_name,
            gpu_arch=gpu_arch, T=T, B=B,
            block_M=config['block_M'], block_N=config['block_N'],
            block_K=config['block_K'], num_stages=config['num_stages'],
            threads=config['threads'], latency_us=config.get('latency_us', 0.0),
        )
        key = self._cache_key(shape_key, gpu_name, gpu_arch, T, B)
        self._entries[key] = entry

    def save(self):
        """Persist cache to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(e) for e in self._entries.values()]
        with open(self.path, 'w') as f:
            json.dump(data, f, indent=2)

    def _entry_to_dict(self, entry: TuningEntry) -> dict:
        return dict(
            block_M=entry.block_M, block_N=entry.block_N, block_K=entry.block_K,
            num_stages=entry.num_stages, threads=entry.threads,
            latency_us=entry.latency_us,
        )

    def __len__(self) -> int:
        return len(self._entries)
