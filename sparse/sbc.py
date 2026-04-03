"""Spiking Brain Compression (SBC) — second-order post-training pruning for SNNs.

Key idea: replace the standard OBC Hessian H = X^TX with the Surrogate
Membrane Potential (SMP) Hessian H_SMP = 2·(MX)^T·(MX), where M is the
Van Rossum Distance convolution matrix encoding LIF temporal dynamics.

Pruning mode:
  - N:M structured: ExactOBS-style element-wise greedy with per-row H⁻¹
    and N:M capacity constraint (global greedy ordering).

Reference:
    Anonymous, "Spiking Brain Compression", under review at ICLR 2026.
"""

import torch
import torch.nn as nn
from typing import Optional


# ---------------------------------------------------------------------------
# VRD matrix construction
# ---------------------------------------------------------------------------

def build_vrd_matrix(T: int, tau: float, device: torch.device) -> torch.Tensor:
    """Build the Van Rossum Distance convolution matrix M.

    M[i,j] = (1 - 1/τ)^(i-j) / τ  for i >= j, 0 otherwise.
    For IF neurons (τ=∞): M = lower-triangular ones (cumulative sum).

    Args:
        T:   Number of timesteps.
        tau: Membrane time constant. float('inf') for IF.
        device: Compute device.
    Returns:
        (T, T) lower-triangular matrix.
    """
    if tau == float('inf') or tau > 1e6:
        return torch.tril(torch.ones(T, T, device=device))
    decay = 1.0 - 1.0 / tau
    scale = 1.0 / tau
    powers = torch.arange(T, device=device).float()
    decay_vec = decay ** powers
    M = scale * decay_vec.unsqueeze(1) / decay_vec.unsqueeze(0)
    return torch.tril(M)


# ---------------------------------------------------------------------------
# ExactOBS-style N:M pruning: element-wise greedy with per-row H⁻¹
# ---------------------------------------------------------------------------

def sbc_prune_layer_nm_global(
    W: torch.Tensor,
    H: torch.Tensor,
    n: int = 2,
    m: int = 4,
    rel_damp: float = 0.01,
    parallel: int = 0,
) -> tuple[torch.Tensor, float]:
    """ExactOBS-style N:M pruning: element-wise greedy with per-row H⁻¹.

    N:M-constrained element-wise greedy OBS (following ExactOBS/OBC):
      Prune one weight at a time across all rows. Each row maintains its
      own H⁻¹ (they diverge as different columns are pruned per row).
      At each step, every row prunes its cheapest surviving weight with
      OBS compensation and rank-1 H⁻¹ update.

      Each M-group has a capacity of (M-N) zeros. Once a group reaches
      capacity, its remaining weights become ineligible. This directly
      produces a valid N:M mask — no post-hoc adjustment needed.

      Per-row H⁻¹ already encodes the exact pruning history for each
      row, so no Phase 2 re-compensation is needed.

    Args:
        W:        (rows, cols) weight matrix.
        H:        (cols, cols) SMP Hessian.
        n:        Non-zeros to KEEP per M-group.
        m:        Group size.
        rel_damp: Relative Hessian damping.
        parallel: Rows to process simultaneously (0 = all).

    Returns:
        (W_pruned, total_loss)
    """
    rows, cols = W.shape
    n_prune = m - n
    assert cols % m == 0
    n_groups = cols // m
    target_zeros_per_row = n_prune * n_groups

    W = W.clone().float()
    H = H.clone().double()

    # Dead columns: H_jj=0 means input column is always zero, so weight
    # value is irrelevant. Do NOT zero them — keep original values to avoid
    # creating extra zeros that break N:M group structure. Instead, make
    # them unprunable (high saliency) by fixing H diagonal.
    dead = H.diag() == 0
    n_dead = dead.sum().item()
    H[dead, dead] = 1.0
    # Do NOT zero W[:, dead] — keep original weights for N:M integrity

    damp = rel_damp * H.diag().mean()
    diag_idx = torch.arange(cols, device=H.device)
    H[diag_idx, diag_idx] += damp

    try:
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L).float()
    except RuntimeError:
        H[diag_idx, diag_idx] += 10 * damp
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L).float()

    total_loss = 0.0
    col_to_group = torch.arange(cols, device=W.device) // m

    # Pre-count existing zeros per group (from dead columns or original
    # weights). These consume N:M group capacity before pruning begins.
    existing_zeros = (W == 0)  # (rows, cols)

    # Process rows in batches for memory efficiency.
    # Auto-compute batch size to keep per-row Hinvs under ~2 GB.
    if parallel <= 0:
        max_bytes = 2 * 1024**3  # 2 GB
        per_row_bytes = cols * cols * 4  # float32
        parallel = max(1, int(max_bytes / per_row_bytes))
    batch_size = min(parallel, rows)

    for i1 in range(0, rows, batch_size):
        i2 = min(i1 + batch_size, rows)
        count = i2 - i1

        # Per-row H⁻¹ copies for this batch
        Hinvs = Hinv.unsqueeze(0).expand(count, -1, -1).clone()
        w = W[i1:i2].clone()
        pruned = existing_zeros[i1:i2].clone()  # pre-mark existing zeros
        # Pre-fill group capacity with existing zeros
        group_zeros = pruned.reshape(count, n_groups, m).sum(dim=2).long()

        for step in range(target_zeros_per_row):
            diags = torch.diagonal(Hinvs, dim1=1, dim2=2)
            sal = w ** 2 / diags.clamp(min=1e-10)
            sal[pruned] = float('inf')

            # N:M capacity: mask full groups
            group_full = group_zeros >= n_prune
            col_groups = col_to_group.unsqueeze(0).expand(count, -1)
            sal[group_full.gather(1, col_groups)] = float('inf')

            # Early termination: if all groups full for all rows, stop
            if group_full.all():
                break

            # Each row picks cheapest
            j = sal.argmin(dim=1)  # (count,)
            j_idx = j.unsqueeze(1)
            range_count = torch.arange(count, device=W.device)

            # Gather per-row values
            w_j = w[range_count, j]
            d = diags[range_count, j]

            # OBS compensation: w -= (w_j / d) * Hinv[:, j]
            j_exp = j.unsqueeze(1).unsqueeze(2).expand(-1, cols, 1)
            row = Hinvs.gather(2, j_exp).squeeze(2)  # (count, cols)
            w -= row * (w_j / d.clamp(min=1e-10)).unsqueeze(1)

            # Accumulate loss: L = w_j² / (2 * d)
            total_loss += (w_j ** 2 / d.clamp(min=1e-10)).sum().item() / 2

            # Zero pruned weight and mark
            pruned.scatter_(1, j_idx, True)
            w.scatter_(1, j_idx, torch.zeros(count, 1, device=W.device))

            # Update group counter
            group_zeros.scatter_add_(
                1, col_to_group[j].unsqueeze(1),
                torch.ones(count, 1, dtype=torch.long, device=W.device))

            # Rank-1 H⁻¹ update: Hinv -= outer(row, row) / d
            row_normed = row / d.sqrt().clamp(min=1e-10).unsqueeze(1)
            Hinvs -= torch.bmm(row_normed.unsqueeze(2), row_normed.unsqueeze(1))

        # Final cleanup: zero all pruned positions (compensation leaks)
        w[pruned] = 0.0
        W[i1:i2] = w

    return W, total_loss
