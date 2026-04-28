"""SNN-specific SBC pipeline: module-wise compression with SMP Hessian.

Pipeline:
  1. Identify modules: each Linear/Conv(+BN) → LIF is one module
  2. Collect SMP Hessian per module: H = 2·(MX)^T·(MX)
  3. ExactOBS-style N:M pruning per module (global greedy + im2col)
  4. BN recalibration

Usage:
    python -m sparse.snn_sbc \
        --model sew_resnet_cifar56 \
        --dense-checkpoint output/.../best.pth \
        --dataset cifar100 --data-root /data/twt/datasets \
        --T 4 --nm 2 4 --evaluate
"""

import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional

from models.neurons import reset_net
from sparse.sbc import build_vrd_matrix, sbc_prune_layer_nm_global
from sparse.utils import (
    _is_eligible, _get_weight_2d, _write_weight_back,
    _detect_neuron_fed_layers,
    _get_neuron_types, _get_inner_neuron, _get_tau,
)

_NEURON_TYPES = _get_neuron_types()


# ---------------------------------------------------------------------------
# SMP Hessian collection (module-wise)
# ---------------------------------------------------------------------------

def collect_smp_hessians(
    model: nn.Module,
    dataloader,
    device: torch.device,
    T: int = 4,
    max_batches: int = 0,
    exclude_names: Optional[list] = None,
    im2col: bool = False,
    pattern_weight: bool = False,
    pw_n: int = 2,
    pw_m: int = 4,
) -> tuple[dict, dict]:
    """Collect SMP Hessians: H = 2·(MX)^T·(MX) per eligible layer.

    For each layer, builds the VRD matrix M from the downstream neuron's tau,
    applies M to the temporal input, and accumulates H.

    Args:
        model:         SNN model (eval mode).
        dataloader:    Calibration data.
        device:        Compute device.
        T:             Number of timesteps.
        max_batches:   Batches to use (0 = all).
        exclude_names: Layer prefixes to skip.
        im2col:        If True, use nn.Unfold for Conv2d to build
                       (C*R*S, C*R*S) Hessian instead of (C, C).
                       Captures cross-spatial correlations for better
                       pruning quality (ExactOBS-style).
        pattern_weight: If True, weight each sample's Hessian contribution
                       by its spike pattern alignment with the N:M constraint.
                       Samples with hardware-friendly patterns (popcount ≤ n
                       per M-group) contribute more; expensive patterns
                       (popcount > n) contribute less. Requires im2col=True
                       for Conv2d.
        pw_n:          N in N:M for pattern weighting (non-zeros to keep).
        pw_m:          M in N:M for pattern weighting (group size).

    Returns:
        (hessians, input_from_neuron)
    """
    if exclude_names is None:
        exclude_names = ['head', 'fc', 'classifier']

    eligible = OrderedDict()
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        if not _is_eligible(module):
            continue
        if any(name == e or name.startswith(e + '.') for e in exclude_names):
            continue
        eligible[name] = module

    input_from_neuron = _detect_neuron_fed_layers(model, eligible, _NEURON_TYPES)
    reset_net(model)

    # Detect downstream neuron for each layer
    exec_order = []
    hooks_trace = []
    for name, mod in model.named_modules():
        is_elig = name in eligible
        is_neuron = isinstance(mod, _NEURON_TYPES) and not name.endswith('.neuron')
        if is_elig or is_neuron:
            def make_h(n, ie, isn):
                def h(m, i, o):
                    exec_order.append((n, ie, isn))
                return h
            hooks_trace.append(mod.register_forward_hook(make_h(name, is_elig, is_neuron)))

    model.eval()
    with torch.no_grad():
        first_conv = next((m for m in model.modules() if isinstance(m, nn.Conv2d)), None)
        in_ch = first_conv.in_channels if first_conv else 3
        try:
            model(torch.randn(1, in_ch, 32, 32, device=device))
        except Exception:
            model(torch.randn(1, in_ch, 224, 224, device=device))
        reset_net(model)
    for h in hooks_trace:
        h.remove()

    layer_to_neuron = {}
    modules_dict = dict(model.named_modules())
    for i, (name, is_elig, is_neuron) in enumerate(exec_order):
        if is_elig:
            for j in range(i + 1, len(exec_order)):
                if exec_order[j][2]:
                    layer_to_neuron[name] = exec_order[j][0]
                    break

    # Build per-layer VRD matrix
    layer_M = {}
    for layer_name, neuron_name in layer_to_neuron.items():
        neuron_mod = modules_dict[neuron_name]
        inner = _get_inner_neuron(neuron_mod)
        tau = _get_tau(inner)
        if tau is None:
            tau = float('inf')
        layer_M[layer_name] = build_vrd_matrix(T, tau, device)

    # Accumulate Hessians
    hessians: dict[str, torch.Tensor] = {}
    n_samples: dict[str, int] = {}
    hooks = []

    def _make_hook(name, mod):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()

            if isinstance(mod, nn.Conv2d) and im2col:
                # im2col mode: use nn.Unfold to get (C*R*S)-dim input vectors
                unfold = nn.Unfold(
                    mod.kernel_size, dilation=mod.dilation,
                    padding=mod.padding, stride=mod.stride,
                )
                if x.ndim == 5:
                    T_dim, B_dim = x.shape[0], x.shape[1]
                    patches = []
                    for t in range(T_dim):
                        patches.append(unfold(x[t]))  # (B, C*R*S, L)
                    patches = torch.stack(patches, 0)  # (T, B, C*R*S, L)
                else:
                    TB = x.shape[0]
                    B_dim = TB // T
                    T_dim = T
                    x_5d = x.reshape(T, B_dim, *x.shape[1:])
                    patches = []
                    for t in range(T):
                        patches.append(unfold(x_5d[t]))
                    patches = torch.stack(patches, 0)  # (T, B, C*R*S, L)

                D = patches.shape[2]  # C*R*S
                # (T, B, C*R*S, L) → (T, B*L, C*R*S)
                x_flat = patches.permute(0, 1, 3, 2).reshape(T_dim, -1, D)

            elif isinstance(mod, nn.Conv2d):
                # Channel-only mode (original)
                if x.ndim == 5:
                    T_dim, B_dim = x.shape[0], x.shape[1]
                    D = x.shape[2]
                    x_flat = x.permute(0, 1, 3, 4, 2).reshape(T_dim, -1, D)
                else:
                    TB = x.shape[0]
                    D = x.shape[1]
                    B_dim = TB // T
                    T_dim = T
                    x_flat = x.reshape(T, B_dim, D, x.shape[2], x.shape[3])
                    x_flat = x_flat.permute(0, 1, 3, 4, 2).reshape(T, -1, D)

            elif isinstance(mod, nn.Linear):
                D = x.shape[-1]
                T_dim = T
                if x.ndim == 3 and x.shape[0] == T:
                    x_flat = x.reshape(T, -1, D)
                else:
                    total = x.shape[0]
                    B_dim = total // T
                    x_flat = x.reshape(T, B_dim, -1, D).reshape(T, -1, D)
            else:
                return

            D = x_flat.shape[2]

            # Apply VRD matrix M across time dimension
            M_vrd = layer_M.get(name, None)
            if M_vrd is not None:
                # x_flat: (T, N, D) → reshape to (T, N*D) for matmul
                Mx = M_vrd @ x_flat.reshape(T_dim, -1)
                Mx = Mx.reshape(T_dim, -1, D)
            else:
                Mx = x_flat

            # Compute per-sample pattern weights if enabled
            if pattern_weight and D % pw_m == 0:
                # x_flat contains original (pre-VRD) activations
                # Binarize and compute per-M-group popcount
                x_binary = (x_flat > 0.5).float()  # (T, N, D)
                n_groups = D // pw_m
                # Reshape to (T*N, n_groups, m) and count spikes per group
                x_groups = x_binary.reshape(-1, n_groups, pw_m)  # (T*N, n_groups, m)
                popcount = x_groups.sum(dim=2)  # (T*N, n_groups)

                # Weight: aligned (popcount ≤ n) → high weight,
                #         misaligned (popcount > n) → low weight.
                # Per-group alignment: 1 if popcount ≤ n, 0 otherwise
                aligned = (popcount <= pw_n).float()  # (T*N, n_groups)
                # Per-sample weight = fraction of aligned groups
                # Range: [0, 1], with 1 = perfectly aligned
                align_frac = aligned.mean(dim=1)  # (T*N,)

                # Soft weighting: w = align_frac^alpha to control contrast
                # alpha=1: linear, alpha=2: quadratic (sharper)
                sample_weights = align_frac  # (T*N,)
                # Normalize so weights sum to n_samples (preserves scale)
                w_sum = sample_weights.sum().clamp(min=1e-6)
                sample_weights = sample_weights * (sample_weights.numel() / w_sum)
                # Apply: scale each sample's Mx by sqrt(weight)
                sqrt_w = sample_weights.sqrt().unsqueeze(1)  # (T*N, 1)
                Mx_2d = Mx.reshape(-1, D) * sqrt_w  # (T*N, D)
                Mx_2d = Mx_2d.T  # (D, T*N)
            else:
                Mx_2d = Mx.reshape(-1, D).T  # (D, T*N)

            # H += (MX)^T (MX) via running average
            n_new = Mx_2d.shape[1]

            if name not in hessians:
                hessians[name] = torch.zeros(D, D, device=device, dtype=torch.float32)
                n_samples[name] = 0

            n_old = n_samples[name]
            n_total = n_old + n_new
            hessians[name] *= n_old / n_total
            x_scaled = math.sqrt(2.0 / n_total) * Mx_2d
            hessians[name].addmm_(x_scaled, x_scaled.T)
            n_samples[name] = n_total

        return hook_fn

    for name, mod in eligible.items():
        hooks.append(mod.register_forward_hook(_make_hook(name, mod)))

    model.eval()
    batch_count = 0
    total_batches = len(dataloader) if max_batches <= 0 else max_batches
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            model(batch[0].to(device))
            reset_net(model)
            batch_count = batch_idx + 1
            if batch_count % 20 == 0:
                print(f"  Profiled {batch_count}/{total_batches} batches")

    for h in hooks:
        h.remove()

    n_vrd = sum(1 for n in hessians if n in layer_M)
    print(f"Collected SMP Hessians for {len(hessians)} layers "
          f"({batch_count} batches, T={T})")
    print(f"  VRD-weighted: {n_vrd}")

    return hessians, input_from_neuron


# ---------------------------------------------------------------------------
# N:M structured SBC pipeline with global greedy ordering
# ---------------------------------------------------------------------------

def _build_columnslast_perm(C: int, R: int, S: int, device) -> torch.Tensor:
    """Build permutation: (C, R, S) → (R, S, C) flattened indices.

    After this permutation, consecutive columns correspond to consecutive
    input channels at the same spatial position. This makes M-groups of
    4 consecutive columns = 4 consecutive C channels, which is exactly
    the TensorRT 2:4 layout requirement.
    """
    return torch.arange(C * R * S, device=device).reshape(C, R, S).permute(1, 2, 0).flatten()


def _compute_channel_permutation(
    W: torch.Tensor,
    n: int = 2,
    m: int = 4,
) -> torch.Tensor:
    """Compute channel permutation that balances importance across M-groups.

    Sorts input channels by aggregate importance, then distributes them
    so each M-group gets n "likely-keep" (high importance) and (m-n)
    "likely-prune" (low importance) channels.

    When H (im2col Hessian in columnslast layout) is provided, importance
    is Hessian-weighted:
        importance_c = ||W[:, c]||₂² × Σ_{r,s} H_{(r,s,c),(r,s,c)}
    This accounts for both weight magnitude AND input activity/curvature.
    Channels with high weight energy AND high Hessian diagonal (active,
    hard-to-compensate) are most important to separate.

    Args:
        W: (K, C, R, S) Conv2d or (K, C) Linear weight tensor.
        n: Non-zeros to keep per group.
        m: Group size.
        H: (D, D) Hessian in im2col columnslast layout (optional).

    Returns:
        (C,) permutation mapping new channel index → old channel index.
    """
    if W.ndim == 4:
        importance = W.float().flatten(2).pow(2).sum(dim=(0, 2))  # (C,)
    else:
        importance = W.float().pow(2).sum(0)  # (C,)

    C = importance.shape[0]
    n_groups = C // m
    n_prune = m - n

    # Sort channels by importance (descending)
    _, ranked = importance.sort(descending=True)
    top_half = ranked[:n_groups * n]      # n*G channels: "keep candidates"
    bot_half = ranked[n_groups * n:]      # (m-n)*G channels: "prune candidates"
    bot_half = bot_half.flip(0)           # reverse so least important pairs with most important group

    # Build balanced groups: group g gets top[g*n:(g+1)*n] + bot[g*n_prune:(g+1)*n_prune]
    perm = torch.empty(n_groups * m, dtype=torch.long, device=W.device)
    for g in range(n_groups):
        perm[g * m : g * m + n] = top_half[g * n : (g + 1) * n]
        perm[g * m + n : (g + 1) * m] = bot_half[g * n_prune : (g + 1) * n_prune]

    # Handle remainder channels
    if C > n_groups * m:
        remainder = ranked[n_groups * m:]
        perm = torch.cat([perm, remainder])

    return perm


def _build_columnslast_perm_with_channel_perm(
    C: int, R: int, S: int,
    chan_perm: torch.Tensor,
    device,
) -> torch.Tensor:
    """Compose channel permutation with columnslast.

    Standard columnslast maps (C, R, S) → (R, S, C) with C innermost.
    The channel permutation π reorders the C dimension first, then
    columnslast is applied. The result: M-groups of 4 consecutive
    columns contain the channels selected by π, not the natural order.
    """
    idx = torch.arange(C * R * S, device=device).reshape(C, R, S)
    idx_permuted = idx[chan_perm]               # permute C dimension
    return idx_permuted.permute(1, 2, 0).flatten()  # then columnslast


def sbc_nm_global_pipeline(
    model: nn.Module,
    dataloader,
    device: torch.device,
    T: int = 4,
    n: int = 2,
    m: int = 4,
    rel_damp: float = 0.01,
    parallel: int = 0,
    calib_batches: int = 128,
    bn_batches: int = 64,
    exclude_names: Optional[list] = None,
    skip_dense_input: bool = True,
    pattern_weight: bool = False,
    permute_channels: bool = False,
) -> tuple[nn.Module, dict]:
    """ExactOBS-style SBC N:M pruning with im2col Hessian.

    Uses im2col (C*R*S, C*R*S) SMP Hessian for Conv2d layers with
    columnslast permutation for TRT-compatible 2:4 along input channels.
    Element-wise greedy OBS with per-row H⁻¹ and N:M capacity constraint.

    For Conv2d: W reshaped to (K, C*R*S) → columnslast → (K, R*S*C).
    M-groups of m consecutive columns = m consecutive C channels at
    the same kernel spatial position. Inverse-permuted after pruning.

    For Linear: standard (out, in) layout, unchanged.

    Args:
        model:          Dense SNN model (modified in-place).
        dataloader:     Calibration data.
        device:         Compute device.
        T:              Number of timesteps.
        n:              Non-zeros to keep per M-group.
        m:              Group size.
        rel_damp:       Hessian damping.
        parallel:       Row batch size for ExactOBS (0 = all rows).
        calib_batches:  Batches for Hessian collection (0 = all).
        bn_batches:     BN recalibration batches (0 = disable).
        exclude_names:  Layer prefixes to skip.
        skip_dense_input: Skip dense-input layers.
        pattern_weight: Weight Hessian by spike pattern alignment with N:M.

    Returns: (model, info)
    """
    if exclude_names is None:
        exclude_names = ['head', 'fc', 'classifier']
    info = {'n': n, 'm': m, 'sparsity': 1.0 - n / m, 'global_greedy': True,
            'im2col': True, 'pattern_weight': pattern_weight}

    # Step 1: Collect SMP Hessians (im2col for Conv2d)
    pw_tag = " + pattern-weighted" if pattern_weight else ""
    print(f"\n--- Step 1: Collecting im2col SMP Hessians{pw_tag} ({calib_batches} batches, T={T}) ---")
    hessians, input_from_neuron = collect_smp_hessians(
        model, dataloader, device, T=T,
        max_batches=calib_batches, exclude_names=exclude_names,
        im2col=True, pattern_weight=pattern_weight, pw_n=n, pw_m=m,
    )

    # Step 2: ExactOBS-style N:M pruning per layer
    print(f"\n--- Step 2: SBC {n}:{m} pruning (ExactOBS + im2col SMP Hessian{pw_tag}) ---")
    print(f"{'Layer':<42} {'Type':>6} {'shape':>16} "
          f"{'loss':>10} {'rel_err':>8} {'valid':>6}")
    print("-" * 100)

    from sparse.pruning import verify_n_m
    stats = OrderedDict()
    skipped = []

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        if not _is_eligible(module):
            continue
        if any(name == e or name.startswith(e + '.') for e in exclude_names):
            skipped.append((name, 'excluded'))
            continue
        if name not in hessians:
            skipped.append((name, 'no Hessian'))
            continue

        is_neuron_fed = input_from_neuron.get(name, True)
        if skip_dense_input and not is_neuron_fed:
            skipped.append((name, 'dense input'))
            continue

        H = hessians[name]
        is_conv = isinstance(module, nn.Conv2d)

        if is_conv:
            # ExactOBS layout: (K, C*R*S) → columnslast → (K, R*S*C)
            W_raw = module.weight.data.clone().float()
            K, C_in, kR, kS = W_raw.shape
            W_2d = W_raw.flatten(1)  # (K, C*R*S)

            if permute_channels and C_in >= m:
                chan_perm = _compute_channel_permutation(W_raw, n, m)
                perm = _build_columnslast_perm_with_channel_perm(
                    C_in, kR, kS, chan_perm, device=W_2d.device)
            else:
                perm = _build_columnslast_perm(C_in, kR, kS, device=W_2d.device)
            W_2d = W_2d[:, perm]  # (K, R*S*C) — C is innermost (permuted)
            H = H[perm][:, perm]   # permute Hessian accordingly
        else:
            # Linear: (out_features, in_features)
            W_2d = module.weight.data.clone().float()
            if permute_channels and W_2d.shape[1] >= m:
                chan_perm = _compute_channel_permutation(module.weight.data, n, m)
                W_2d = W_2d[:, chan_perm]
                H = H[chan_perm][:, chan_perm]
                perm = chan_perm  # for inverse permute at write-back
            else:
                perm = None

        W_2d = W_2d.to(H.device)
        W_orig = W_2d.clone()
        r, c = W_2d.shape

        if c % m != 0:
            skipped.append((name, f'cols={c} not /{m}'))
            continue

        W_pruned, loss = sbc_prune_layer_nm_global(
            W_2d, H, n=n, m=m,
            rel_damp=rel_damp, parallel=parallel,
        )

        # Verify N:M validity (in permuted order — this IS the deployment order)
        valid = verify_n_m(W_pruned, n, m)
        valid_str = 'ok' if valid else 'FAIL'

        # Reconstruction error
        h_diag_sqrt = H.float().diag().sqrt().clamp(min=1e-8)[:c]
        err = ((W_pruned - W_orig) @ h_diag_sqrt).norm().item()
        ref = (W_orig @ h_diag_sqrt).norm().item()
        rel_err = err / (ref + 1e-8)

        ltype = 'Conv2d' if is_conv else 'Linear'
        print(f"  {name:<40} {ltype:>6} {f'({r},{c})':>16} "
              f"{loss:>10.2f} {rel_err:>8.4f} {valid_str:>6}")

        # Write back: inverse-permute and reshape to original
        if is_conv:
            inv_perm = torch.argsort(perm)
            W_back = W_pruned[:, inv_perm]  # (K, C*R*S) original col order
            module.weight.data.copy_(W_back.reshape(K, C_in, kR, kS))
        elif perm is not None:
            inv_perm = torch.argsort(perm)
            module.weight.data.copy_(W_pruned[:, inv_perm])
        else:
            module.weight.data.copy_(W_pruned)

        stats[name] = {
            'type': ltype, 'shape_2d': (r, c),
            'loss': loss, 'rel_err': rel_err,
            'valid': valid,
        }

    if stats:
        mean_err = sum(s['rel_err'] for s in stats.values()) / len(stats)
        total_loss = sum(s['loss'] for s in stats.values())
        n_valid = sum(1 for s in stats.values() if s['valid'])
        print(f"\n  Pruned {len(stats)} layers  |  total_loss: {total_loss:.2f}  |  "
              f"mean rel_err: {mean_err:.4f}  |  {n}:{m} valid: {n_valid}/{len(stats)}")
    if skipped:
        skip_summary = {}
        for n_, reason in skipped:
            skip_summary.setdefault(reason, []).append(n_)
        for reason, names in skip_summary.items():
            print(f"  Skipped {len(names)} layer(s) ({reason}): {', '.join(names)}")

    info['prune_stats'] = stats

    # Step 3: BN recalibration
    if bn_batches > 0:
        print(f"\n--- Step 3: BN recalibration ({bn_batches} batches) ---")
        from utils.fuse import recalibrate_bn
        recalibrate_bn(model, dataloader, device, max_batches=bn_batches)
        print("  Done")

    return model, info


# ---------------------------------------------------------------------------
# Sparse fine-tuning (mask-frozen, all weights trainable)
# ---------------------------------------------------------------------------

def finetune_sparse(
    model_sparse: nn.Module,
    model_dense: nn.Module,
    dataloader,
    device: torch.device,
    T: int = 4,
    epochs: int = 1,
    lr: float = 1e-4,
    temperature: float = 4.0,
    alpha_kd: float = 0.9,
) -> None:
    """Fine-tune all weights with fixed sparse mask + KD from dense teacher.

    After each optimizer step, re-applies the original zero mask so the
    sparse pattern is preserved. BN running stats are updated normally.

    Args:
        model_sparse: Pruned model (modified in-place).
        model_dense:  Dense teacher (frozen).
        dataloader:   Training data.
        device:       Compute device.
        T:            SNN timesteps.
        epochs:       Fine-tuning epochs.
        lr:           Learning rate.
        temperature:  KD temperature.
        alpha_kd:     KD vs CE weight.
    """
    model_dense.eval()

    # Capture sparse masks (True = zero/pruned, False = kept)
    masks = {}
    for name, mod in model_sparse.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)) and hasattr(mod, 'weight'):
            masks[name] = (mod.weight.data == 0)

    n_total = sum(p.numel() for p in model_sparse.parameters())
    n_masked = sum(m.sum().item() for m in masks.values())
    print(f"  Total params: {n_total}  |  Masked zeros: {int(n_masked)}  |  "
          f"Trainable (non-zero): {n_total - int(n_masked)}")

    model_sparse.train()
    optimizer = torch.optim.SGD(model_sparse.parameters(), lr=lr, momentum=0.9)
    ce_fn = nn.CrossEntropyLoss()
    kl_fn = nn.KLDivLoss(reduction='batchmean')

    for epoch in range(epochs):
        total_loss = 0.0
        correct = total = 0

        for batch_idx, (images, targets) in enumerate(dataloader):
            images, targets = images.to(device), targets.to(device)

            with torch.no_grad():
                logits_d = model_dense(images)
                reset_net(model_dense)

            logits_s = model_sparse(images)
            reset_net(model_sparse)

            soft_d = F.softmax(logits_d / temperature, dim=1)
            log_soft_s = F.log_softmax(logits_s / temperature, dim=1)
            loss = (alpha_kd * kl_fn(log_soft_s, soft_d) * temperature ** 2
                    + (1 - alpha_kd) * ce_fn(logits_s, targets))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Re-apply sparse mask: zero out pruned positions
            with torch.no_grad():
                for name, mod in model_sparse.named_modules():
                    if name in masks:
                        mod.weight.data[masks[name]] = 0.0

            total_loss += loss.item()
            correct += logits_s.argmax(1).eq(targets).sum().item()
            total += targets.size(0)

            if (batch_idx + 1) % 100 == 0:
                print(f"    [{epoch+1}/{epochs}] batch {batch_idx+1}  "
                      f"loss={total_loss/(batch_idx+1):.4f}  "
                      f"acc={100.*correct/total:.2f}%")

        print(f"  Epoch {epoch+1}/{epochs}: "
              f"loss={total_loss/(batch_idx+1):.4f}  "
              f"train_acc={100.*correct/total:.2f}%")

    model_sparse.eval()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SBC: Spiking Brain Compression — N:M structured pruning')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--dense-checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--nm', type=int, nargs=2, default=[2, 4], metavar=('N', 'M'),
                        help='N:M structured sparsity (e.g. --nm 2 4)')
    parser.add_argument('--rel-damp', type=float, default=0.01)
    parser.add_argument('--pattern-weight', action='store_true',
                        help='Weight Hessian by spike pattern alignment with N:M')
    parser.add_argument('--permute-channels', action='store_true',
                        help='Reorder input channels to balance importance across '
                             'M-groups before 2:4 pruning')
    parser.add_argument('--finetune', type=int, default=0, metavar='EPOCHS',
                        help='Sparse fine-tune all weights (mask frozen) with KD for N epochs')
    parser.add_argument('--calib-batches', type=int, default=128)
    parser.add_argument('--bn-batches', type=int, default=64)
    parser.add_argument('--exclude', type=str, nargs='*',
                        default=['head', 'fc', 'classifier'])
    parser.add_argument('--frames-number',type=int,default=None)
    parser.add_argument('--img-size', type=int, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()

    from tengine.utils import (
        load_model_config, build_model_from_config, build_model,
        get_dataset_config, build_dataloaders, set_seed,
    )

    set_seed(42)
    gpu_id = int(args.gpu_ids.split(',')[0])
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    ds_cfg = get_dataset_config(args.dataset)
    img_size = args.img_size or ds_cfg['img_size']

    if args.config:
        config = load_model_config(args.config)
        config.update(ds_cfg)
        config['T'] = args.T
        config['img_size'] = img_size
        model = build_model_from_config(config)
    elif args.model:
        model = build_model(args.model, num_classes=ds_cfg['num_classes'],
                            in_channels=ds_cfg['in_channels'], T=args.T)
    else:
        raise ValueError("Must provide --config or --model")

    model = model.to(device).eval()
    ckpt = torch.load(args.dense_checkpoint, map_location='cpu', weights_only=False)
    # model.load_state_dict(ckpt.get('model', ckpt) if 'model' in ckpt else ckpt.get('state_dict',ckpt))
    for key in ('model', 'state_dict', 'net', 'model_state_dict'):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            state_dict = ckpt[key]
            break
    model.load_state_dict(state_dict, strict=False)
    dl_kwargs = dict(img_size=img_size, num_workers=4)
    if args.frames_number is not None:
        dl_kwargs['frames_number'] = args.frames_number
    train_loader, val_loader = build_dataloaders(
        args.dataset, args.data_root, args.batch_size, **dl_kwargs)

    n_keep, m_group = args.nm
    pw_tag = " + pattern-weighted" if args.pattern_weight else ""
    print(f"Dense checkpoint: {args.dense_checkpoint}")
    print(f"Mode: N:M structured ({n_keep}:{m_group}) + global greedy{pw_tag}")
    print(f"T={args.T}")

    model, info = sbc_nm_global_pipeline(
        model, train_loader, device,
        T=args.T, n=n_keep, m=m_group,
        rel_damp=args.rel_damp,
        calib_batches=args.calib_batches,
        bn_batches=args.bn_batches,
        exclude_names=args.exclude,
        pattern_weight=args.pattern_weight,
        permute_channels=args.permute_channels,
    )
    method_tag = f'sbc_{n_keep}_{m_group}_global'
    if args.pattern_weight:
        method_tag += '_pw'
    if args.permute_channels:
        method_tag += '_perm'

    # Sparse fine-tuning (if requested)
    if args.finetune > 0:
        import copy
        print(f"\n--- Sparse fine-tuning ({args.finetune} epoch(s), KD, mask frozen) ---")
        dense_teacher = copy.deepcopy(model)
        dense_teacher.load_state_dict(state_dict, strict=False)
        dense_teacher.to(device).eval()
        finetune_sparse(
            model, dense_teacher, train_loader, device,
            T=args.T, epochs=args.finetune,
        )
        del dense_teacher
        method_tag += f'_ft{args.finetune}'

    # Save
    import os
    model_tag = args.model or os.path.splitext(os.path.basename(args.config))[0]
    os.makedirs('obc_pt', exist_ok=True)
    save_path = os.path.join('obc_pt', f'{model_tag}_{args.dataset}_{method_tag}.pth')
    save_payload = {
        'model': model.state_dict(),
        'method': method_tag,
        'info': info,
        'params': {
            'nm': args.nm,
            'sparsity': 1.0 - n_keep / m_group,
            'rel_damp': args.rel_damp,
            'T': args.T,
        },
    }
    torch.save(save_payload, save_path)
    print(f"\nSaved to {save_path}")

    if args.output:
        torch.save(save_payload, args.output)

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
        print(f"\nEvaluation accuracy: {acc:.2f}% ({correct}/{total})")
        save_payload['accuracy'] = acc
        torch.save(save_payload, save_path)
