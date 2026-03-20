"""Unified per-layer statistics tracking for all sparse inference backends.

Replaces the duplicated SparseLinearStats, SputnikLinearStats,
SparseAttentionStats, and SputnikAttentionStats classes with a single
reusable pair: LinearStats and AttentionStats.
"""


class LinearStats:
    """Tracks per-layer statistics for sparse linear acceleration.

    Used by all backends (torch_sparse, sputnik, semi_structured, triton).
    """

    def __init__(self):
        self.total_calls = 0
        self.sparse_calls = 0
        self.dense_calls = 0
        self.total_ops = 0
        self.effective_ops = 0
        self.per_layer = {}

    def record(self, layer_name: str, ops: int, density: float,
               used_sparse: bool):
        self.total_calls += 1
        self.total_ops += ops
        if used_sparse:
            self.sparse_calls += 1
            self.effective_ops += int(ops * density)
        else:
            self.dense_calls += 1
            self.effective_ops += ops

        if layer_name not in self.per_layer:
            self.per_layer[layer_name] = {
                'total_ops': 0, 'effective_ops': 0,
                'sparse_calls': 0, 'dense_calls': 0,
                'densities': [],
            }
        s = self.per_layer[layer_name]
        s['total_ops'] += ops
        s['sparse_calls' if used_sparse else 'dense_calls'] += 1
        s['densities'].append(density)
        if used_sparse:
            s['effective_ops'] += int(ops * density)
        else:
            s['effective_ops'] += ops

    def reset(self):
        self.__init__()

    @property
    def density(self):
        return self.effective_ops / max(self.total_ops, 1)

    def to_dict(self) -> dict:
        per_layer = {}
        for name, s in self.per_layer.items():
            densities = s['densities']
            per_layer[name] = {
                'total_ops': s['total_ops'],
                'effective_ops': s['effective_ops'],
                'density': sum(densities) / len(densities) if densities else 1.0,
                'sparse_calls': s['sparse_calls'],
                'dense_calls': s['dense_calls'],
            }
        return {
            'total_ops': self.total_ops,
            'effective_ops': self.effective_ops,
            'density': self.density,
            'per_layer': per_layer,
            'total_calls': self.total_calls,
            'sparse_calls': self.sparse_calls,
            'dense_calls': self.dense_calls,
        }


class AttentionStats:
    """Tracks statistics for sparse attention acceleration (Q@K^T and A@V).

    Used by all backends that accelerate SSA attention matmuls.
    """

    def __init__(self):
        self.total_calls = 0
        self.sparse_qk_calls = 0
        self.sparse_av_calls = 0
        self.total_ops = 0
        self.effective_ops = 0
        self.per_layer = {}

    def record_qk(self, layer_name: str, ops: int, density: float,
                   used_sparse: bool):
        self.total_calls += 1
        self.total_ops += ops
        if used_sparse:
            self.sparse_qk_calls += 1
            self.effective_ops += int(ops * density)
        else:
            self.effective_ops += ops
        self._update_layer(layer_name, 'qk', ops, density, used_sparse)

    def record_av(self, layer_name: str, ops: int, density: float,
                  used_sparse: bool):
        self.total_ops += ops
        if used_sparse:
            self.sparse_av_calls += 1
            self.effective_ops += int(ops * density)
        else:
            self.effective_ops += ops
        self._update_layer(layer_name, 'av', ops, density, used_sparse)

    def _update_layer(self, name, op_type, ops, density, used_sparse):
        if name not in self.per_layer:
            self.per_layer[name] = {
                'qk_ops': 0, 'av_ops': 0,
                'qk_sparse': 0, 'av_sparse': 0,
                'qk_densities': [], 'av_densities': [],
            }
        s = self.per_layer[name]
        s[f'{op_type}_ops'] += ops
        if used_sparse:
            s[f'{op_type}_sparse'] += 1
        s[f'{op_type}_densities'].append(density)

    def reset(self):
        self.__init__()

    @property
    def density(self):
        return self.effective_ops / max(self.total_ops, 1)

    def to_dict(self) -> dict:
        per_layer = {}
        for name, s in self.per_layer.items():
            per_layer[name] = {
                'qk_density': (sum(s['qk_densities']) / len(s['qk_densities'])
                               if s['qk_densities'] else 1.0),
                'av_density': (sum(s['av_densities']) / len(s['av_densities'])
                               if s['av_densities'] else 1.0),
                'qk_sparse_calls': s['qk_sparse'],
                'av_sparse_calls': s['av_sparse'],
            }
        return {
            'total_ops': self.total_ops,
            'effective_ops': self.effective_ops,
            'density': self.density,
            'per_layer': per_layer,
            'total_calls': self.total_calls,
            'sparse_qk_calls': self.sparse_qk_calls,
            'sparse_av_calls': self.sparse_av_calls,
        }
