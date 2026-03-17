"""SR-STE N:M weight projection for structured sparse training of SNN Linear layers.

SR-STE (Sparse-Refined Straight-Through Estimator) regularizes training toward an
N:M sparsity pattern so that post-training hard projection incurs minimal accuracy
loss. See "Efficient GPU Kernels for N:M Sparse Weights" (Zhou et al., 2021).

Usage:
    from sparse.st_train import (
        sr_ste_regularizer,
        ProgressiveSparsityScheduler,
        apply_hard_n_m_projection,
    )

    scheduler = ProgressiveSparsityScheduler(start_epoch=50, end_epoch=150,
                                              target_lambda=0.01)
    # in train_one_epoch, per step:
    sr_loss = sr_ste_regularizer(model, scheduler.get_lambda(epoch))
    loss = task_loss + sr_loss
    loss.backward()

    # after training finishes:
    apply_hard_n_m_projection(model)
"""

import torch
import torch.nn as nn

from sparse.pruning import prune_n_m


# ---------------------------------------------------------------------------
# SR-STE regularizer
# ---------------------------------------------------------------------------

def sr_ste_regularizer(
    model: nn.Module,
    lambda_sr: float,
    n: int = 2,
    m: int = 4,
) -> torch.Tensor:
    """Compute the SR-STE regularization loss: lambda_sr * ||W - project_N_M(W)||^2.

    Only applied to nn.Linear weight parameters. Biases and non-Linear
    parameters are excluded. The sum is taken over all eligible layers.

    Args:
        model: The model being trained.
        lambda_sr: Regularization strength (0.0 disables the term).
        n: Non-zeros to keep per group (default 2).
        m: Group size (default 4).

    Returns:
        Scalar tensor (differentiable w.r.t. model parameters).
        Returns a zero tensor on the correct device if lambda_sr == 0
        or no eligible layers are found.
    """
    if lambda_sr == 0.0:
        device = next(
            (p.device for p in model.parameters() if p.requires_grad), 'cpu'
        )
        return torch.tensor(0.0, device=device)

    reg = None
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        w = module.weight
        if not w.requires_grad:
            continue
        # project is non-differentiable at zero crossings, but in practice we
        # treat it as a constant (straight-through estimator): we detach the
        # projection target so gradients only flow through w.
        with torch.no_grad():
            w_proj = prune_n_m(w.detach(), n=n, m=m)
        diff = w - w_proj
        layer_reg = (diff * diff).sum()
        reg = layer_reg if reg is None else reg + layer_reg

    if reg is None:
        device = next(
            (p.device for p in model.parameters() if p.requires_grad), 'cpu'
        )
        return torch.tensor(0.0, device=device)

    return lambda_sr * reg


# ---------------------------------------------------------------------------
# Progressive sparsity scheduler
# ---------------------------------------------------------------------------

class ProgressiveSparsityScheduler:
    """Linearly anneals SR-STE lambda from 0 to target_lambda.

    Lambda is 0 for epochs < start_epoch, increases linearly during
    [start_epoch, end_epoch], and stays at target_lambda for later epochs.

    Args:
        start_epoch: Epoch at which regularization begins (0-indexed).
        end_epoch: Epoch at which lambda reaches its target value.
        target_lambda: Final regularization coefficient.

    Example:
        sched = ProgressiveSparsityScheduler(50, 150, 0.01)
        for epoch in range(200):
            lam = sched.get_lambda(epoch)
            sr_loss = sr_ste_regularizer(model, lam)
    """

    def __init__(self, start_epoch: int, end_epoch: int, target_lambda: float):
        if end_epoch <= start_epoch:
            raise ValueError(
                f"end_epoch ({end_epoch}) must be > start_epoch ({start_epoch})"
            )
        if target_lambda < 0:
            raise ValueError(f"target_lambda must be >= 0, got {target_lambda}")

        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.target_lambda = target_lambda

    def get_lambda(self, epoch: int) -> float:
        """Return the SR-STE lambda for the given epoch (0-indexed)."""
        if epoch < self.start_epoch:
            return 0.0
        if epoch >= self.end_epoch:
            return self.target_lambda
        progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        return self.target_lambda * progress

    def __repr__(self) -> str:
        return (
            f"ProgressiveSparsityScheduler("
            f"start={self.start_epoch}, end={self.end_epoch}, "
            f"target_lambda={self.target_lambda})"
        )


# ---------------------------------------------------------------------------
# Hard N:M projection (post-training, in-place)
# ---------------------------------------------------------------------------

def apply_hard_n_m_projection(
    model: nn.Module,
    n: int = 2,
    m: int = 4,
) -> dict:
    """Hard-project all nn.Linear weights to N:M sparsity in-place.

    Called once after SR-STE-regularized training completes, before
    converting to SparseSemiStructuredTensor.

    Args:
        model: The trained model. Its nn.Linear weight tensors are modified
            in-place. This is intentional and destructive — call it on a
            copy if you need the original dense weights.
        n: Non-zeros to keep per group (default 2).
        m: Group size (default 4).

    Returns:
        Dict mapping layer name to sparsity info:
            {name: {'shape': tuple, 'density': float, 'total_params': int,
                    'nonzero': int}}
    """
    stats = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        with torch.no_grad():
            w_proj = prune_n_m(module.weight.data, n=n, m=m)
            module.weight.data.copy_(w_proj)

        total = module.weight.numel()
        nonzero = module.weight.data.count_nonzero().item()
        stats[name] = {
            'shape': tuple(module.weight.shape),
            'total_params': total,
            'nonzero': nonzero,
            'density': nonzero / max(total, 1),
        }

    return stats


# Backward-compatible alias
apply_hard_2_4_projection = apply_hard_n_m_projection
