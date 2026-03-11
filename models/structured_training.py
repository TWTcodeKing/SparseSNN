"""SR-STE 2:4 weight projection for structured sparse training of SNN Linear layers.

SR-STE (Sparse-Refined Straight-Through Estimator) regularizes training toward a
2:4 sparsity pattern so that post-training hard projection incurs minimal accuracy
loss. See "Efficient GPU Kernels for N:M Sparse Weights" (Zhou et al., 2021).

Usage:
    from models.structured_training import (
        project_to_2_4,
        sr_ste_regularizer,
        ProgressiveSparsityScheduler,
        apply_hard_2_4_projection,
    )

    scheduler = ProgressiveSparsityScheduler(start_epoch=50, end_epoch=150,
                                              target_lambda=0.01)
    # in train_one_epoch, per step:
    sr_loss = sr_ste_regularizer(model, scheduler.get_lambda(epoch))
    loss = task_loss + sr_loss
    loss.backward()

    # after training finishes:
    apply_hard_2_4_projection(model)
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Core projection: 2:4 magnitude projection
# ---------------------------------------------------------------------------

def project_to_2_4(weight: torch.Tensor) -> torch.Tensor:
    """Project a weight tensor to 2:4 sparsity (keep top-2 per group of 4).

    For each group of 4 contiguous elements along dim=-1, zeros the 2 elements
    with the smallest magnitude. This is a straight-through-compatible operation:
    the forward result is the 2:4-projected weight; gradients flow through
    unchanged (the caller is responsible for using this in a loss, not replacing
    the parameter in-place during the forward pass for autograd).

    Args:
        weight: Tensor with at least 1 dimension. Projection is applied along
            the last dimension (dim=-1). Typically shape (out, in) for Linear.

    Returns:
        Projected tensor with same shape and dtype as input.
    """
    orig_shape = weight.shape
    orig_dtype = weight.dtype

    # Flatten all leading dims into one so we always work on a 2D view
    # shape: (rows, cols)
    w2d = weight.reshape(-1, orig_shape[-1])
    rows, cols = w2d.shape

    # Pad to multiple of 4 along cols
    pad = (4 - cols % 4) % 4
    if pad > 0:
        w2d = torch.cat(
            [w2d, torch.zeros(rows, pad, dtype=orig_dtype, device=w2d.device)],
            dim=1,
        )

    padded_cols = w2d.shape[1]
    # Reshape to (rows, num_groups, 4)
    grouped = w2d.reshape(rows, padded_cols // 4, 4)

    # Find the 2 smallest by magnitude and build a binary mask
    _, small_idx = grouped.abs().topk(2, dim=2, largest=False)
    mask = torch.ones_like(grouped)
    mask.scatter_(2, small_idx, 0.0)

    projected = (grouped * mask).reshape(rows, padded_cols)

    # Remove padding
    if pad > 0:
        projected = projected[:, :cols]

    return projected.reshape(orig_shape)


# ---------------------------------------------------------------------------
# SR-STE regularizer
# ---------------------------------------------------------------------------

def sr_ste_regularizer(model: nn.Module, lambda_sr: float) -> torch.Tensor:
    """Compute the SR-STE regularization loss: lambda_sr * ||W - project_2_4(W)||^2.

    Only applied to nn.Linear weight parameters. Biases and non-Linear
    parameters are excluded. The sum is taken over all eligible layers.

    Args:
        model: The model being trained.
        lambda_sr: Regularization strength (0.0 disables the term).

    Returns:
        Scalar tensor (differentiable w.r.t. model parameters).
        Returns a zero tensor on the correct device if lambda_sr == 0
        or no eligible layers are found.
    """
    if lambda_sr == 0.0:
        # Avoid graph construction overhead when disabled
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
        # project_to_2_4 is non-differentiable at zero crossings, but
        # in practice we treat it as a constant (straight-through estimator):
        # we detach the projection target so gradients only flow through w.
        with torch.no_grad():
            w_proj = project_to_2_4(w.detach())
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
# Hard 2:4 projection (post-training, in-place)
# ---------------------------------------------------------------------------

def apply_hard_2_4_projection(model: nn.Module) -> dict:
    """Hard-project all nn.Linear weights to 2:4 sparsity in-place.

    Called once after SR-STE-regularized training completes, before
    converting to SparseSemiStructuredTensor.

    Args:
        model: The trained model. Its nn.Linear weight tensors are modified
            in-place. This is intentional and destructive — call it on a
            copy if you need the original dense weights.

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
            w_proj = project_to_2_4(module.weight.data)
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
