"""SparseGPT-style Optimal Brain Surgeon for N:M structured pruning of SNNs.

Strictly follows the SparseGPT algorithm (Frantar & Alistarh, ICML 2023):
  1. Collect per-layer Hessians H = X·Xᵀ from calibration forward passes
  2. Compute upper Cholesky of H⁻¹ for efficient row-wise access
  3. Block-wise column processing with intra-block + cross-block compensation
  4. N:M mask selection via OBS saliency: w²/[H⁻¹]²_{ii}

Key differences from vanilla SparseGPT:
  - Conv2d support via im2col Hessian (weight reshaped to 2D)
  - Dead column handling for SNN binary activations (many zero-firing neurons)
  - Pluggable scorer interface for future SNN-specific metrics (firing rates, etc.)
  - CUTLASS dimension validation (rows%32==0, cols%64==0 for 2:4 Sparse TC)

Reference:
    Frantar & Alistarh, "SparseGPT: Massive Language Models Can Be Accurately
    Pruned in One-Shot", ICML 2023.
"""

import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional, Callable

from sparse.pruning import verify_n_m


# ---------------------------------------------------------------------------
# Layer eligibility (separate from permutation eligibility)
# ---------------------------------------------------------------------------

# CUTLASS 2:4 requirements for SparseSemiStructuredTensor (fp16)
_CUTLASS_ROW_ALIGN = 32
_CUTLASS_COL_ALIGN = 64


def _is_eligible_for_obs(module: nn.Module) -> bool:
    """Check if a module is eligible for OBS pruning.

    Eligible: nn.Linear or nn.Conv2d (groups=1) with enough input features
    for at least one N:M group (>= 4 for 2:4).
    """
    if isinstance(module, nn.Linear):
        return module.in_features >= 4
    if isinstance(module, nn.Conv2d):
        if module.groups != 1:
            return False
        return module.in_channels >= 1  # im2col may expand to >= 4
    return False


def _get_weight_2d(module: nn.Module) -> tuple[torch.Tensor, tuple]:
    """Get the 2D weight view for OBS.

    Linear: (out_features, in_features) — already 2D.
    Conv2d: (C_out, C_in*Kh*Kw) — im2col layout.

    Returns:
        (W_2d, orig_shape) where orig_shape is the Conv2d weight shape
        (None for Linear).
    """
    W = module.weight.data
    if isinstance(module, nn.Conv2d):
        orig_shape = W.shape
        return W.reshape(W.shape[0], -1).contiguous(), orig_shape
    return W, None


def _check_cutlass_dims(w_2d_shape: tuple[int, int]) -> tuple[bool, str]:
    """Check if a 2D weight shape satisfies CUTLASS 2:4 Sparse TC requirements.

    Returns:
        (ok, reason) where ok=True if dimensions are aligned.
    """
    rows, cols = w_2d_shape
    ok = (rows % _CUTLASS_ROW_ALIGN == 0) and (cols % _CUTLASS_COL_ALIGN == 0)
    if ok:
        return True, ''
    return False, (f'({rows},{cols}): need rows%{_CUTLASS_ROW_ALIGN}==0, '
                   f'cols%{_CUTLASS_COL_ALIGN}==0')


def _detect_neuron_fed_layers(
    model: nn.Module,
    eligible: OrderedDict,
    neuron_types: tuple,
) -> dict[str, bool]:
    """Detect which eligible layers receive input from a spiking neuron.

    Uses forward hooks to trace actual execution order. A Linear/Conv2d is
    "neuron-fed" if ANY spiking neuron has fired before it in the forward
    pass. Only the very first Linear/Conv2d layers (before any neuron has
    executed) are considered "dense-input" — these receive raw images.

    This correctly handles cross-container boundaries (e.g., LIF in
    patch_embed feeds q_linear in block.0.attn).
    """
    import torch

    # Track execution order via hooks
    exec_order = []  # list of (name, is_neuron, is_eligible)
    hooks = []
    modules_dict = dict(model.named_modules())

    for name, mod in model.named_modules():
        is_neuron = isinstance(mod, neuron_types)
        is_elig = name in eligible

        if is_neuron or is_elig:
            def make_hook(n, is_n, is_e):
                def hook(module, inp, out):
                    exec_order.append((n, is_n, is_e))
                return hook
            hooks.append(mod.register_forward_hook(make_hook(name, is_neuron, is_elig)))

    # Single dummy forward pass to trace execution order
    device = next(model.parameters()).device
    # Infer input shape from first Conv2d or model config
    first_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            first_conv = m
            break
    in_ch = first_conv.in_channels if first_conv else 3
    # Try common image sizes
    from models.neurons import reset_net
    with torch.no_grad():
        try:
            model(torch.randn(1, in_ch, 32, 32, device=device))
        except Exception:
            try:
                model(torch.randn(1, in_ch, 224, 224, device=device))
            except Exception:
                pass
        reset_net(model)

    for h in hooks:
        h.remove()

    # Walk execution order: layers before any neuron fires are "dense-input"
    result = {}
    neuron_has_fired = False
    for name, is_neuron, is_elig in exec_order:
        if is_neuron:
            neuron_has_fired = True
        if is_elig:
            result[name] = neuron_has_fired

    # Any eligible layers not seen in exec_order → assume dense
    for name in eligible:
        if name not in result:
            result[name] = False

    return result


# ---------------------------------------------------------------------------
# Hessian collection — following SparseGPT add_batch
# ---------------------------------------------------------------------------

def collect_hessians(
    model: nn.Module,
    dataloader,
    device: torch.device,
    max_batches: int = 128,
    exclude_names: Optional[list] = None,
) -> dict[str, torch.Tensor]:
    """Collect per-layer Hessians H = 2/n * Σ(X·Xᵀ) from calibration data.

    Uses running-average accumulation matching SparseGPT's add_batch:
    H is rescaled at each batch so the result is a stable average.

    For Conv2d layers, input is unfolded via im2col to match the 2D weight
    layout (C_out, C_in*Kh*Kw).

    Args:
        model:         SNN model (eval mode, on device).
        dataloader:    Calibration data (no labels needed).
        device:        Compute device.
        max_batches:   Number of calibration batches (default 128).
        exclude_names: Layer name prefixes to skip (default ['head']).

    Returns:
        {layer_name: H} where H is (d_in, d_in) on *device*.
    """
    from models.neurons import reset_net

    if exclude_names is None:
        exclude_names = ['head']

    eligible = OrderedDict()
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        if not _is_eligible_for_obs(module):
            continue
        if any(name == e or name.startswith(e + '.') for e in exclude_names):
            continue
        eligible[name] = module

    # Structural check: which layers receive input from a spiking neuron?
    # Walk model modules in forward order. A Linear/Conv2d is "neuron-fed"
    # if the most recent preceding module (in the sequential child list of
    # its parent) is or contains a spiking neuron. This correctly handles
    # the first layer (receives dense images, no upstream neuron).
    from models.neurons import (
        MultiStepLIFNeuron, MultiStepIFNeuron, LIFNeuron, IFNeuron,
    )
    _NEURON_TYPES = (MultiStepLIFNeuron, MultiStepIFNeuron, LIFNeuron, IFNeuron)

    input_from_neuron = _detect_neuron_fed_layers(model, eligible, _NEURON_TYPES)
    # Reset neuron states after the detection forward pass
    reset_net(model)

    # Per-layer accumulators
    hessians: dict[str, torch.Tensor] = {}
    n_samples: dict[str, int] = {}
    hooks = []

    def _make_hook(name, mod):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()

            if isinstance(mod, nn.Conv2d):
                x_unf = F.unfold(
                    x, mod.kernel_size,
                    dilation=mod.dilation, padding=mod.padding,
                    stride=mod.stride,
                )  # (B, C_in*Kh*Kw, L)
                x = x_unf.permute(1, 0, 2).reshape(x_unf.shape[1], -1)  # (K, B*L)
            elif isinstance(mod, nn.Linear):
                x = x.reshape(-1, x.shape[-1]).T  # (d_in, n)
            else:
                return

            n_new = x.shape[1]
            d = x.shape[0]

            if name not in hessians:
                hessians[name] = torch.zeros(d, d, device=x.device, dtype=torch.float32)
                n_samples[name] = 0

            # Running-average Hessian (SparseGPT add_batch style)
            n_old = n_samples[name]
            n_total = n_old + n_new
            hessians[name] *= n_old / n_total
            x_scaled = math.sqrt(2.0 / n_total) * x
            hessians[name].addmm_(x_scaled, x_scaled.T)
            n_samples[name] = n_total

        return hook_fn

    for name, mod in eligible.items():
        hooks.append(mod.register_forward_hook(_make_hook(name, mod)))

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            images = batch[0].to(device)
            model(images)
            reset_net(model)
            if (batch_idx + 1) % 20 == 0:
                print(f"  Profiled {batch_idx + 1}/{max_batches} batches")

    for h in hooks:
        h.remove()

    # Report neuron-fed vs dense-input detection
    n_neuron = sum(1 for v in input_from_neuron.values() if v)
    n_dense = sum(1 for v in input_from_neuron.values() if not v)
    dense_layers = [n for n, v in input_from_neuron.items() if not v]
    print(f"Collected Hessians for {len(hessians)} layers "
          f"({min(batch_idx + 1, max_batches)} batches)")
    print(f"  Neuron-fed: {n_neuron}, Dense-input: {n_dense}")
    if dense_layers:
        print(f"  Dense-input layers: {', '.join(dense_layers)}")

    return hessians, input_from_neuron


# ---------------------------------------------------------------------------
# Scorer interface — pluggable mask selection for future SNN metrics
# ---------------------------------------------------------------------------

def default_obs_scorer(
    W_block: torch.Tensor,
    Hinv_diag: torch.Tensor,
    col_offset: int,
    m: int,
    **kwargs,
) -> torch.Tensor:
    """Default SparseGPT saliency scorer: w² / [H⁻¹]²_{ii}.

    Args:
        W_block:     (rows, m) weight values for the current M-group.
        Hinv_diag:   (m,) diagonal of Hinv for these m columns.
        col_offset:  Global column index of the first column in this group.
        m:           Group size.
        **kwargs:    Reserved for future SNN-specific scorers.

    Returns:
        (rows, m) saliency scores — lower = safer to prune.
    """
    return W_block ** 2 / (Hinv_diag.unsqueeze(0) ** 2)


def wanda_scorer(
    W_block: torch.Tensor,
    Hinv_diag: torch.Tensor,
    col_offset: int,
    m: int,
    H_diag: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Wanda saliency scorer: |w| · √(H_{jj}).

    Derived from OBS via diagonal approximation of H⁻¹:
        S_q = w² / [H⁻¹]_{qq}  ≈  w² · H_{qq}  =  (|w| · √H_{qq})²

    For binary SNN spikes: H_{jj} = firing_rate_j, so this becomes:
        S_q = |w| · √(firing_rate_j)

    This is the Wanda metric (Sun et al., 2023), which is a principled
    first-order approximation to the full OBS objective. It requires no
    matrix inversion and is numerically stable for any layer size.

    Args:
        W_block:     (rows, m) weight values.
        Hinv_diag:   (m,) diagonal of Hinv (NOT used by this scorer).
        col_offset:  Global column index of the first column.
        m:           Group size.
        H_diag:      (cols,) diagonal of the original H (= firing rates for SNNs).
                     Must be provided via scorer_kwargs={'H_diag': H.diag()}.

    Returns:
        (rows, m) saliency scores — lower = safer to prune.
    """
    if H_diag is None:
        # Fallback to OBS scorer if H_diag not provided
        return W_block ** 2 / (Hinv_diag.unsqueeze(0) ** 2)
    h_diag_block = H_diag[col_offset:col_offset + m].clamp(min=1e-10)
    return W_block.abs() * h_diag_block.sqrt().unsqueeze(0)


def hybrid_scorer(
    W_block: torch.Tensor,
    Hinv_diag: torch.Tensor,
    col_offset: int,
    m: int,
    H_diag: Optional[torch.Tensor] = None,
    lam: float = 0.5,
    **kwargs,
) -> torch.Tensor:
    """Hybrid scorer: OBS saliency + Wanda firing-rate term.

    Combines the full second-order OBS score with the diagonal (firing rate)
    term from Wanda. The motivation: OBS captures weight-Hessian interactions
    but can be noisy for ill-conditioned layers. The Wanda term adds a stable
    activation-magnitude signal that is especially informative for SNNs.

        score = (1-λ) · w²/d² + λ · |w|·√(H_{jj})

    Both terms are normalized to [0,1] per group before combining, so λ
    directly controls the interpolation.

    For binary SNNs: H_{jj} = firing_rate_j, making the Wanda term equivalent
    to weighting by √(firing_rate). Channels with higher firing rates are
    more important and less likely to be pruned.

    Args:
        lam: Interpolation weight for Wanda term. 0 = pure OBS, 1 = pure Wanda.
             Default 0.5 (equal mix).
    """
    # OBS term: w² / d²
    obs_scores = W_block ** 2 / (Hinv_diag.unsqueeze(0) ** 2)

    if H_diag is None or lam == 0.0:
        return obs_scores

    # Wanda term: |w| · √(H_{jj})
    h_diag_block = H_diag[col_offset:col_offset + m].clamp(min=1e-10)
    wanda_scores = W_block.abs() * h_diag_block.sqrt().unsqueeze(0)

    # Normalize each to [0,1] per group for fair combination
    obs_max = obs_scores.max(dim=1, keepdim=True).values.clamp(min=1e-10)
    wanda_max = wanda_scores.max(dim=1, keepdim=True).values.clamp(min=1e-10)

    return (1 - lam) * (obs_scores / obs_max) + lam * (wanda_scores / wanda_max)


# Type alias for scorer functions
Scorer = Callable[..., torch.Tensor]


# ---------------------------------------------------------------------------
# Per-layer OBS solver — strictly follows SparseGPT fasterprune
# ---------------------------------------------------------------------------

def obs_prune_layer(
    W_2d: torch.Tensor,
    H: torch.Tensor,
    n: int = 2,
    m: int = 4,
    blocksize: int = 128,
    percdamp: float = 0.01,
    scorer: Optional[Scorer] = None,
    scorer_kwargs: Optional[dict] = None,
) -> tuple[torch.Tensor, float]:
    """SparseGPT OBS N:M pruning for a single layer.

    Algorithm (matching SparseGPT fasterprune):
      1. Dampen H, compute upper Cholesky of H⁻¹
      2. Process columns in blocks of `blocksize`:
         a. Within block, process column-by-column:
            - At each M-group boundary, select N:M mask via scorer
            - Zero pruned weights, apply intra-block OBS compensation
         b. After block, apply cross-block compensation to all remaining cols
      3. Return pruned+compensated weights and total OBS loss

    Args:
        W_2d:       (rows, cols) weight matrix.
        H:          (cols, cols) Hessian (output of collect_hessians).
        n, m:       N:M sparsity (default 2:4).
        blocksize:  Column block size for batched updates (default 128).
        percdamp:   Dampening as fraction of mean(diag(H)).
        scorer:     Custom saliency scorer (default: w²/d² SparseGPT).
        scorer_kwargs: Extra kwargs passed to scorer (e.g. firing rates).

    Returns:
        (W_pruned, loss) — pruned weight tensor and scalar OBS loss.
    """
    if scorer is None:
        scorer = default_obs_scorer
    if scorer_kwargs is None:
        scorer_kwargs = {}

    rows, cols = W_2d.shape
    W = W_2d.clone().float()
    Hs = H.clone().float()

    # --- Dead columns (never-activated inputs in SNN) ---
    dead = Hs.diag() == 0
    dead_ratio = dead.float().mean().item()

    # Fall back to magnitude pruning if activations are too sparse for OBS.
    # Criteria: too many dead columns, or Hessian diagonal too small overall
    # (near-zero firing rates make the inverse numerically unstable).
    diag_mean = Hs.diag().mean().item()
    if dead_ratio > 0.5 or diag_mean < 1e-4:
        from sparse.pruning import prune_n_m
        return prune_n_m(W_2d, n=n, m=m), 0.0

    Hs[dead, dead] = 1
    W[:, dead] = 0

    # --- Dampening ---
    damp = percdamp * Hs.diag().mean()
    diag_idx = torch.arange(cols, device=Hs.device)
    Hs[diag_idx, diag_idx] += damp

    # --- Upper Cholesky of H⁻¹ (SparseGPT key step) ---
    # H = L L^T  →  H⁻¹ = cholesky_inverse(L)  →  Hinv = cholesky(H⁻¹, upper=True)
    # Hinv is upper triangular: Hinv[i,j]=0 for j<i
    # This ensures compensation from column i only affects columns j>i.
    try:
        L = torch.linalg.cholesky(Hs)
        H_inv_full = torch.cholesky_inverse(L)
        Hinv = torch.linalg.cholesky(H_inv_full, upper=True)
    except RuntimeError:
        # Fallback: more aggressive dampening
        Hs[diag_idx, diag_idx] += 10 * damp
        L = torch.linalg.cholesky(Hs)
        H_inv_full = torch.cholesky_inverse(L)
        Hinv = torch.linalg.cholesky(H_inv_full, upper=True)

    Losses = torch.zeros(rows, device=W.device)

    # --- Block-wise column processing ---
    num_prune = m - n

    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()                    # (rows, count)
        Q1 = torch.zeros_like(W1)                    # pruned output
        Err1 = torch.zeros_like(W1)                  # scaled errors for cross-block
        Losses1 = torch.zeros_like(W1)               # per-element loss
        Hinv1 = Hinv[i1:i2, i1:i2]                   # block diagonal of upper Cholesky

        # N:M mask for this block — initially all False (nothing pruned)
        mask1 = torch.zeros(rows, count, dtype=torch.bool, device=W.device)

        for i in range(count):
            w = W1[:, i]                              # (rows,)
            d = Hinv1[i, i]                           # scalar diagonal

            # At each M-group boundary, select which columns to prune
            global_col = i1 + i
            if global_col % m == 0:
                group_end = min(i + m, count)
                group_len = group_end - i
                if group_len == m:
                    # Full M-group: use scorer to rank columns
                    group_diag = torch.diag(Hinv1)[i:group_end]
                    group_w = W1[:, i:group_end]
                    scores = scorer(
                        group_w, group_diag, col_offset=global_col, m=m,
                        **scorer_kwargs,
                    )
                    # Prune the num_prune lowest-scoring per row
                    _, prune_idx = scores.topk(num_prune, dim=1, largest=False)
                    mask1[:, i:group_end].scatter_(1, prune_idx, True)

            q = w.clone()
            q[mask1[:, i]] = 0.0                      # zero pruned weights

            Q1[:, i] = q
            Losses1[:, i] = (w - q) ** 2 / d ** 2

            # Intra-block OBS compensation
            err1 = (w - q) / d                         # (rows,)
            W1[:, i:] -= err1.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err1

        # Write pruned block back
        W[:, i1:i2] = Q1
        Losses += torch.sum(Losses1, dim=1) / 2

        # Cross-block compensation: propagate errors to all remaining columns
        if i2 < cols:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    loss = Losses.sum().item()
    return W.reshape_as(W_2d), loss


# ---------------------------------------------------------------------------
# Model-level OBS pruning
# ---------------------------------------------------------------------------

def apply_obs_pruning(
    model: nn.Module,
    hessians: dict[str, torch.Tensor],
    n: int = 2,
    m: int = 4,
    blocksize: int = 128,
    percdamp: float = 0.01,
    exclude_names: Optional[list] = None,
    scorer: Optional[Scorer] = None,
    scorer_kwargs: Optional[dict] = None,
    warn_cutlass: bool = True,
    input_from_neuron: Optional[dict[str, bool]] = None,
    skip_dense_input: bool = True,
) -> tuple[nn.Module, dict]:
    """Apply OBS N:M pruning to all eligible layers in-place.

    Args:
        model:           SNN model (dense weights). Modified in-place.
        hessians:        {layer_name: H} from collect_hessians.
        n, m:            N:M parameters (default 2:4).
        blocksize:       Column block size for SparseGPT (default 128).
        percdamp:        Hessian dampening fraction.
        exclude_names:   Layer name prefixes to skip (default ['head']).
        scorer:          Custom saliency scorer (default: SparseGPT w²/d²).
        scorer_kwargs:   Extra kwargs passed to scorer per layer.
        warn_cutlass:    Warn about layers ineligible for CUTLASS 2:4 Sparse TC.
        input_from_neuron: {layer_name: bool} from collect_hessians. True if
            the layer's input comes directly from a spiking neuron. If None,
            all layers are assumed to be neuron-fed.
        skip_dense_input: Skip layers whose inputs do NOT come from a
            spiking neuron (e.g., the first Conv2d receiving dense RGB
            images). These layers have no activation sparsity to exploit,
            and pruning them wastes accuracy. Default True.

    Returns:
        (model, stats) — the modified model and per-layer statistics dict.
    """
    if exclude_names is None:
        exclude_names = ['head']
    if input_from_neuron is None:
        input_from_neuron = {}

    print(f"\n=== OBS {n}:{m} pruning (blocksize={blocksize}, percdamp={percdamp}) ===\n")
    print(f"{'Layer':<42} {'Type':>6} {'shape':>16} {'Input':>6} {'loss':>10} {'rel_err':>8}  {'2:4':>3}  CUTLASS")
    print("-" * 110)

    stats = OrderedDict()
    skipped = []

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        if not _is_eligible_for_obs(module):
            continue
        if any(name == e or name.startswith(e + '.') for e in exclude_names):
            skipped.append((name, 'excluded'))
            continue
        if name not in hessians:
            skipped.append((name, 'no Hessian'))
            continue

        # Skip layers whose input does NOT come from a spiking neuron
        is_neuron_fed = input_from_neuron.get(name, True)  # default True if unknown
        if skip_dense_input and not is_neuron_fed:
            skipped.append((name, 'dense input'))
            continue

        H = hessians[name]
        is_conv = isinstance(module, nn.Conv2d)
        W_2d, orig_shape = _get_weight_2d(module)
        W_2d = W_2d.cpu().float()
        W_orig = W_2d.clone()

        r, c = W_2d.shape

        # Build scorer kwargs — auto-inject H_diag for Wanda/hybrid scorers
        layer_scorer_kwargs = dict(scorer_kwargs or {})
        if scorer in (wanda_scorer, hybrid_scorer) and 'H_diag' not in layer_scorer_kwargs:
            layer_scorer_kwargs['H_diag'] = H.cpu().diag()

        # OBS prune
        W_pruned, loss = obs_prune_layer(
            W_2d, H.cpu(), n=n, m=m, blocksize=blocksize,
            percdamp=percdamp, scorer=scorer, scorer_kwargs=layer_scorer_kwargs,
        )

        # Reconstruction error metric
        r_proxy = H.cpu().diag().sqrt().clamp(min=1e-8)[:c]
        err = ((W_pruned - W_orig) @ r_proxy).norm().item()
        ref = (W_orig @ r_proxy).norm().item()
        rel_err = err / (ref + 1e-8)

        pattern_ok = verify_n_m(W_pruned, n=n, m=m)
        cutlass_ok, cutlass_reason = _check_cutlass_dims((r, c))

        ltype = 'Conv2d' if is_conv else 'Linear'
        ok_str = 'ok' if pattern_ok else 'FAIL'
        cut_str = 'ok' if cutlass_ok else cutlass_reason
        inp_str = 'neuron' if is_neuron_fed else 'dense'

        print(f"  {name:<40} {ltype:>6} {f'({r},{c})':>16} {inp_str:>6} "
              f"{loss:>10.2f} {rel_err:>8.4f}  {ok_str:>4}  {cut_str}")

        # Write back
        if is_conv:
            W_back = W_pruned.reshape(orig_shape).contiguous()
        else:
            W_back = W_pruned
        module.weight.data.copy_(W_back)

        stats[name] = {
            'type': ltype,
            'shape_2d': (r, c),
            'input_from_neuron': is_neuron_fed,
            'loss': loss,
            'rel_err': rel_err,
            'pattern_ok': pattern_ok,
            'cutlass_ok': cutlass_ok,
            'cutlass_reason': cutlass_reason,
        }

    # Summary
    if stats:
        mean_err = sum(s['rel_err'] for s in stats.values()) / len(stats)
        total_loss = sum(s['loss'] for s in stats.values())
        bad_nm = [n_ for n_, s in stats.items() if not s['pattern_ok']]
        bad_cut = [n_ for n_, s in stats.items() if not s['cutlass_ok']]

        print(f"\n  Pruned {len(stats)} layers  |  total_loss: {total_loss:.2f}  |  "
              f"mean rel_err: {mean_err:.4f}")
        if bad_nm:
            print(f"  WARNING: N:M pattern broken in {len(bad_nm)} layer(s): {bad_nm}")
        else:
            print(f"  All {len(stats)} layers satisfy {n}:{m} pattern")
        if bad_cut and warn_cutlass:
            print(f"  NOTE: {len(bad_cut)} layer(s) ineligible for CUTLASS 2:4 Sparse TC:")
            for n_ in bad_cut:
                print(f"    {n_}: {stats[n_]['cutlass_reason']}")
    if skipped:
        skip_summary = {}
        for n_, reason in skipped:
            skip_summary.setdefault(reason, []).append(n_)
        for reason, names in skip_summary.items():
            print(f"  Skipped {len(names)} layer(s) ({reason}): {', '.join(names)}")

    return model, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='OBS-based N:M structured pruning for SNNs (training-free)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Dense model checkpoint (.pth)')
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config for transformer models')
    parser.add_argument('--model', type=str, default=None,
                        help='ResNet model name (e.g. ms_resnet34)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (cifar10, cifar100, imagenet, ...)')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to dataset root directory')
    parser.add_argument('--T', type=int, default=4,
                        help='Number of timesteps (default 4)')
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--m', type=int, default=4)
    parser.add_argument('--blocksize', type=int, default=128,
                        help='Column block size for SparseGPT (default 128)')
    parser.add_argument('--percdamp', type=float, default=0.01,
                        help='Hessian dampening fraction (default 0.01)')
    parser.add_argument('--calib-batches', type=int, default=128,
                        help='Number of calibration batches (default 128)')
    parser.add_argument('--exclude', type=str, nargs='*', default=['head'],
                        help='Layer name prefixes to skip (default: head)')
    parser.add_argument('--scorer', type=str, default='obs',
                        choices=['obs', 'wanda', 'hybrid'],
                        help='Saliency scorer: obs (default SparseGPT w²/d²), '
                             'wanda (|w|·√firing_rate), hybrid (interpolation)')
    parser.add_argument('--lam', type=float, default=0.5,
                        help='Hybrid scorer interpolation: 0=pure OBS, 1=pure Wanda')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation after pruning')
    parser.add_argument('--output', type=str, default=None,
                        help='Save pruned model to this path')
    args = parser.parse_args()

    from tengine.utils import (
        load_model_config, build_model_from_config, build_model,
        get_dataset_config, build_dataloaders,
    )
    from models.neurons import reset_net

    gpu_id = int(args.gpu_ids.split(',')[0])
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    ds_cfg = get_dataset_config(args.dataset)

    if args.config:
        config = load_model_config(args.config)
        config.update(ds_cfg)
        config['T'] = args.T
        model = build_model_from_config(config)
    elif args.model:
        model = build_model(args.model, num_classes=ds_cfg['num_classes'],
                            in_channels=ds_cfg['in_channels'], T=args.T)
    else:
        parser.error('Must specify --config or --model')

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    dense_state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(dense_state)
    model = model.to(device)
    print(f"Loaded dense checkpoint from {args.checkpoint}")

    train_loader, val_loader = build_dataloaders(
        args.dataset, args.data_root, args.batch_size,
        img_size=ds_cfg['img_size'], num_workers=4,
    )

    # Resolve scorer
    scorer_map = {'obs': default_obs_scorer, 'wanda': wanda_scorer, 'hybrid': hybrid_scorer}
    scorer_fn = scorer_map[args.scorer]
    scorer_kwargs = {}
    if args.scorer == 'hybrid':
        scorer_kwargs['lam'] = args.lam
    print(f"Scorer: {args.scorer}" + (f" (lam={args.lam})" if args.scorer == 'hybrid' else ''))

    original_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\n--- Collecting Hessians ({args.calib_batches} batches) ---")
    hessians, input_from_neuron = collect_hessians(
        model, train_loader, device,
        max_batches=args.calib_batches, exclude_names=args.exclude,
    )
    model, stats = apply_obs_pruning(
        model, hessians,
        n=args.n, m=args.m, blocksize=args.blocksize,
        percdamp=args.percdamp, exclude_names=args.exclude,
        input_from_neuron=input_from_neuron,
        scorer=scorer_fn, scorer_kwargs=scorer_kwargs,
    )

    # Save
    if args.output:
        torch.save({
            'model': model.state_dict(),
            'method': 'obs',
            'original_state': original_state,
            'params': {
                'n': args.n, 'm': args.m,
                'blocksize': args.blocksize,
                'percdamp': args.percdamp,
                'calib_batches': args.calib_batches,
            },
            'stats': dict(stats),
        }, args.output)
        print(f"\nSaved OBS-pruned model to {args.output}")

    # Evaluate
    if args.evaluate:
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                reset_net(model)
                correct += outputs.argmax(1).eq(targets).sum().item()
                total += targets.size(0)
        acc = 100.0 * correct / total
        print(f"\nEvaluation accuracy: {acc:.2f}%  ({correct}/{total})")
