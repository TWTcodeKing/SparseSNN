"""Backward-compatible re-export shim.

All N:M pruning primitives now live in sparse.pruning.
This module re-exports them so existing imports continue to work.
"""

from sparse.pruning import (  # noqa: F401
    prune_n_m,
    prune_2_4,
    verify_n_m,
    verify_2_4,
    prune_model_linear,
)

# Legacy aliases used by existing callers
prune_2_4 = prune_2_4
verify_2_4 = verify_2_4
