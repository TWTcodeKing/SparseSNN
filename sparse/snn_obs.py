"""SNN-aware OBS: spike-boundary-weighted Hessian + threshold calibration.

SNN-specific adaptation of SparseGPT for 2:4 structured weight conversion:
  1. Spike-aware Hessian: H_spike = X · diag(D²) · Xᵀ, weights samples near
     the spike boundary (D[t] = exp(-|v_t - v_th|²/2σ²))
  2. Hybrid scorer: (1-λ)·w²/d² + λ·|w|·√(firing_rate) — OBS + Wanda
  3. Skip dense-input layers (first Conv receiving RGB images)
  4. Threshold calibration: adjust v_th to restore dense firing rates
  5. BN recalibration: update BatchNorm stats after pruning

Usage:
    python -m sparse.snn_obs \
        --config configs/spikformer/spikformer_cifar.yaml \
        --dense-checkpoint output/.../best.pth \
        --dataset cifar100 --data-root /home/twt/datasets \
        --scorer hybrid --lam 0.3 --alpha 0.15 --sigma 0.2 \
        --output spikformer_snn_obs.pth --evaluate
"""

import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional

from models.neurons import reset_net
from sparse.pruning import verify_n_m
from sparse.OBS import (
    obs_prune_layer, _is_eligible_for_obs, _get_weight_2d,
    _check_cutlass_dims, _detect_neuron_fed_layers,
    default_obs_scorer, wanda_scorer, hybrid_scorer,
    Scorer,
)
from sparse.utils import (
    _get_neuron_types, _get_inner_neuron, _get_vth, _set_vth,
    collect_firing_rates, collect_membrane_potentials,
)

_NEURON_TYPES = _get_neuron_types()


# ---------------------------------------------------------------------------
# Spike-aware Hessian collection (incremental, memory-efficient)
# ---------------------------------------------------------------------------

def collect_spike_aware_hessians(
    model: nn.Module,
    dataloader,
    device: torch.device,
    max_batches: int = 128,
    exclude_names: Optional[list] = None,
    sigma: float = 0.5,
) -> tuple[dict, dict, dict]:
    """Collect spike-boundary-weighted Hessians incrementally.

    Uses two-phase hooks:
      - Layer hook: stores input X temporarily
      - Downstream neuron hook: reads membrane potential V, computes
        boundary weight D, accumulates H_spike += (X·D)(X·D)ᵀ, discards X

    Both H_spike and H_standard are accumulated in O(d²) memory per layer.

    Returns:
        (hessians_spike, hessians_standard, input_from_neuron)
    """
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

    input_from_neuron = _detect_neuron_fed_layers(model, eligible, _NEURON_TYPES)
    reset_net(model)

    # Find downstream neuron for each layer via execution order trace
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
        model(torch.randn(1, in_ch, 32, 32, device=device))
        reset_net(model)
    for h in hooks_trace:
        h.remove()

    layer_to_neuron = {}
    for i, (name, is_elig, is_neuron) in enumerate(exec_order):
        if is_elig:
            for j in range(i + 1, len(exec_order)):
                if exec_order[j][2]:
                    layer_to_neuron[name] = exec_order[j][0]
                    break

    modules_dict = dict(model.named_modules())

    # Accumulators (incremental, O(d²) per layer)
    H_spike = {}      # spike-weighted Hessian
    H_std = {}         # standard Hessian
    n_samples = {}     # sample count per layer
    pending_x = {}     # temporary X storage (cleared after neuron hook)

    hooks = []

    # Phase 1 hooks: layer hooks store X temporarily
    for layer_name, mod in eligible.items():
        def make_layer_hook(ln, m):
            def hook(module, inp, out):
                x = inp[0].detach().float()
                if isinstance(m, nn.Conv2d):
                    # Handle 5D (T,B,C,H,W) from MultiStep wrappers
                    if x.ndim == 5:
                        x = x.flatten(0, 1)  # (T*B, C, H, W)
                    x_unf = F.unfold(x, m.kernel_size, dilation=m.dilation,
                                     padding=m.padding, stride=m.stride)
                    x = x_unf.permute(1, 0, 2).reshape(x_unf.shape[1], -1)
                elif isinstance(m, nn.Linear):
                    x = x.reshape(-1, x.shape[-1]).T
                else:
                    return
                pending_x[ln] = x  # store temporarily
            return hook
        hooks.append(mod.register_forward_hook(make_layer_hook(layer_name, mod)))

    # Phase 2 hooks: neuron hooks compute D and accumulate H
    neuron_to_layers = {}
    for ln, nn_ in layer_to_neuron.items():
        neuron_to_layers.setdefault(nn_, []).append(ln)

    for neuron_name, layer_names in neuron_to_layers.items():
        if neuron_name not in modules_dict:
            continue
        neuron_mod = modules_dict[neuron_name]
        inner = _get_inner_neuron(neuron_mod)

        def make_neuron_hook(l_names, inner_n, sig):
            def hook(module, inp, out):
                v = getattr(inner_n, 'v', None)
                if v is None:
                    # MSNeuron: stateless, use input as proxy
                    x = inp[0]
                    v = x[-1] if (x.ndim >= 3 and x.shape[0] <= 16) else x
                vth = _get_vth(inner_n)

                # Compute mean boundary weight for this batch
                if isinstance(v, torch.Tensor):
                    v_flat = v.detach().float().reshape(-1)
                    boundary_dist = (v_flat - vth).abs()
                    w_batch = torch.exp(-boundary_dist ** 2 / (2 * sig ** 2)).mean().item()
                else:
                    w_batch = 1.0

                for ln in l_names:
                    if ln not in pending_x:
                        continue
                    x = pending_x[ln]  # (d_in, n)
                    d_in = x.shape[0]
                    n_new = x.shape[1]

                    if ln not in H_spike:
                        H_spike[ln] = torch.zeros(d_in, d_in, device=x.device, dtype=torch.float32)
                        H_std[ln] = torch.zeros(d_in, d_in, device=x.device, dtype=torch.float32)
                        n_samples[ln] = 0

                    # Incremental running average (SparseGPT style)
                    n_old = n_samples[ln]
                    n_total = n_old + n_new

                    # Standard Hessian
                    H_std[ln] *= n_old / n_total
                    x_scaled = math.sqrt(2.0 / n_total) * x
                    H_std[ln].addmm_(x_scaled, x_scaled.T)

                    # Spike-weighted Hessian
                    H_spike[ln] *= n_old / n_total
                    x_weighted = math.sqrt(2.0 / n_total) * w_batch * x
                    H_spike[ln].addmm_(x_weighted, x_weighted.T)

                    n_samples[ln] = n_total
                    del pending_x[ln]  # free memory

            return hook
        hooks.append(neuron_mod.register_forward_hook(
            make_neuron_hook(layer_names, inner, sigma)))

    # Also handle layers without downstream neurons (accumulate standard only)
    layers_without_neuron = set(eligible.keys()) - set(layer_to_neuron.keys())

    # Collect data
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            model(batch[0].to(device))
            reset_net(model)

            # Handle layers without downstream neuron: accumulate from pending_x
            for ln in list(pending_x.keys()):
                if ln in layers_without_neuron:
                    x = pending_x.pop(ln)
                    d_in = x.shape[0]
                    n_new = x.shape[1]
                    if ln not in H_std:
                        H_std[ln] = torch.zeros(d_in, d_in, device=x.device, dtype=torch.float32)
                        H_spike[ln] = torch.zeros(d_in, d_in, device=x.device, dtype=torch.float32)
                        n_samples[ln] = 0
                    n_old = n_samples[ln]
                    n_total = n_old + n_new
                    H_std[ln] *= n_old / n_total
                    x_scaled = math.sqrt(2.0 / n_total) * x
                    H_std[ln].addmm_(x_scaled, x_scaled.T)
                    H_spike[ln] = H_std[ln].clone()  # no weighting
                    n_samples[ln] = n_total

            # Clear any remaining pending (shouldn't happen, but safety)
            pending_x.clear()

            if (batch_idx + 1) % 20 == 0:
                print(f"  Profiled {batch_idx + 1}/{max_batches} batches")

    for h in hooks:
        h.remove()

    n_spike = sum(1 for v in input_from_neuron.values() if v)
    n_dense = sum(1 for v in input_from_neuron.values() if not v)
    dense_layers = [n for n, v in input_from_neuron.items() if not v]
    print(f"Collected spike-aware Hessians for {len(H_spike)} layers "
          f"({min(batch_idx + 1, max_batches)} batches, σ={sigma})")
    print(f"  Neuron-fed: {n_spike}, Dense-input: {n_dense}")
    if dense_layers:
        print(f"  Dense-input layers: {', '.join(dense_layers)}")

    return H_spike, H_std, input_from_neuron


# ---------------------------------------------------------------------------
# Spike-aware OBS + threshold calibration pipeline
# ---------------------------------------------------------------------------

def spike_aware_obs_pruning(
    model: nn.Module,
    hessians_spike: dict,
    hessians_standard: dict,
    n: int = 2,
    m: int = 4,
    blocksize: int = 128,
    percdamp: float = 0.01,
    exclude_names: Optional[list] = None,
    scorer: Optional[Scorer] = None,
    scorer_kwargs: Optional[dict] = None,
    input_from_neuron: Optional[dict] = None,
    skip_dense_input: bool = True,
) -> tuple[nn.Module, dict]:
    """Apply OBS pruning using spike-aware Hessians."""
    if exclude_names is None:
        exclude_names = ['head']
    if scorer is None:
        scorer = default_obs_scorer
    if scorer_kwargs is None:
        scorer_kwargs = {}
    if input_from_neuron is None:
        input_from_neuron = {}

    print(f"\n=== Spike-Aware OBS {n}:{m} (blocksize={blocksize}, percdamp={percdamp}) ===\n")
    print(f"{'Layer':<42} {'Type':>6} {'shape':>16} {'Input':>6} {'loss':>10} {'rel_err':>8}  {'2:4':>3}")
    print("-" * 100)

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
        if name not in hessians_spike:
            skipped.append((name, 'no Hessian'))
            continue

        is_neuron_fed = input_from_neuron.get(name, True)
        if skip_dense_input and not is_neuron_fed:
            skipped.append((name, 'dense input'))
            continue

        Hs = hessians_spike[name]
        H_std = hessians_standard.get(name, Hs)
        is_conv = isinstance(module, nn.Conv2d)
        W_2d, orig_shape = _get_weight_2d(module)
        W_2d = W_2d.cpu().float()
        W_orig = W_2d.clone()
        r, c = W_2d.shape

        layer_scorer_kwargs = dict(scorer_kwargs)
        if scorer in (wanda_scorer, hybrid_scorer) and 'H_diag' not in layer_scorer_kwargs:
            layer_scorer_kwargs['H_diag'] = H_std.cpu().diag()

        W_pruned, loss = obs_prune_layer(
            W_2d, Hs.cpu(), n=n, m=m, blocksize=blocksize,
            percdamp=percdamp, scorer=scorer, scorer_kwargs=layer_scorer_kwargs,
        )

        r_proxy = H_std.cpu().diag().sqrt().clamp(min=1e-8)[:c]
        err = ((W_pruned - W_orig) @ r_proxy).norm().item()
        ref = (W_orig @ r_proxy).norm().item()
        rel_err = err / (ref + 1e-8)
        pattern_ok = verify_n_m(W_pruned, n=n, m=m)

        ltype = 'Conv2d' if is_conv else 'Linear'
        ok_str = 'ok' if pattern_ok else 'FAIL'
        inp_str = 'neuron' if is_neuron_fed else 'dense'
        print(f"  {name:<40} {ltype:>6} {f'({r},{c})':>16} {inp_str:>6} "
              f"{loss:>10.2f} {rel_err:>8.4f}  {ok_str:>4}")

        if is_conv:
            W_back = W_pruned.reshape(orig_shape).contiguous()
        else:
            W_back = W_pruned
        module.weight.data.copy_(W_back)

        stats[name] = {
            'type': ltype, 'shape_2d': (r, c),
            'loss': loss, 'rel_err': rel_err, 'pattern_ok': pattern_ok,
        }

    if stats:
        mean_err = sum(s['rel_err'] for s in stats.values()) / len(stats)
        total_loss = sum(s['loss'] for s in stats.values())
        print(f"\n  Pruned {len(stats)} layers  |  total_loss: {total_loss:.2f}  |  "
              f"mean rel_err: {mean_err:.4f}")
    if skipped:
        skip_summary = {}
        for n_, reason in skipped:
            skip_summary.setdefault(reason, []).append(n_)
        for reason, names in skip_summary.items():
            print(f"  Skipped {len(names)} ({reason}): {', '.join(names)}")

    return model, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Spike-Aware OBS: boundary-weighted Hessian + threshold calibration')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--dense-checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--m', type=int, default=4)
    parser.add_argument('--blocksize', type=int, default=128)
    parser.add_argument('--percdamp', type=float, default=0.01)
    parser.add_argument('--calib-batches', type=int, default=128)
    parser.add_argument('--scorer', type=str, default='hybrid',
                        choices=['obs', 'wanda', 'hybrid'])
    parser.add_argument('--lam', type=float, default=0.3)
    parser.add_argument('--sigma', type=float, default=0.5,
                        help='Gaussian width for boundary weighting')
    parser.add_argument('--alpha', type=float, default=0.15,
                        help='Threshold calibration strength')
    parser.add_argument('--bn-batches', type=int, default=64,
                        help='BN recalibration batches (0 to disable)')
    parser.add_argument('--img-size', type=int, default=None,
                        help='Override image size (e.g. 128 for transfer models)')
    parser.add_argument('--exclude', type=str, nargs='*', default=['head'])
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
        build_fn = lambda: build_model_from_config(config)
    elif args.model:
        build_fn = lambda: build_model(args.model, num_classes=ds_cfg['num_classes'],
                                        in_channels=ds_cfg['in_channels'], T=args.T)
    else:
        raise ValueError("Must provide --config or --model")

    model = build_fn().to(device).eval()
    ckpt = torch.load(args.dense_checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('model', ckpt))

    train_loader, val_loader = build_dataloaders(
        args.dataset, args.data_root, args.batch_size,
        img_size=img_size, num_workers=4)

    print(f"Dense checkpoint: {args.dense_checkpoint}")
    print(f"Scorer: {args.scorer}, σ={args.sigma}, α={args.alpha}")

    # Step 1: Collect spike-aware Hessians (incremental)
    print(f"\n--- Collecting spike-aware Hessians ({args.calib_batches} batches) ---")
    H_spike, H_std, input_from_neuron = collect_spike_aware_hessians(
        model, train_loader, device,
        max_batches=args.calib_batches, exclude_names=args.exclude,
        sigma=args.sigma,
    )

    # Step 2: OBS prune with spike-aware Hessians
    scorer_map = {'obs': default_obs_scorer, 'wanda': wanda_scorer, 'hybrid': hybrid_scorer}
    scorer_fn = scorer_map[args.scorer]
    scorer_kwargs = {'lam': args.lam} if args.scorer == 'hybrid' else {}

    model, stats = spike_aware_obs_pruning(
        model, H_spike, H_std,
        n=args.n, m=args.m, blocksize=args.blocksize,
        percdamp=args.percdamp, exclude_names=args.exclude,
        scorer=scorer_fn, scorer_kwargs=scorer_kwargs,
        input_from_neuron=input_from_neuron,
    )

    # Step 3: Co-calibration (BN → Threshold → BN)
    from utils.fuse import recalibrate_bn

    # Phase 1: BN recalibration (correct distribution shift from pruning)
    if args.bn_batches > 0:
        print(f"\n--- BN recalibration pass 1 ({args.bn_batches} batches) ---")
        recalibrate_bn(model, train_loader, device, max_batches=args.bn_batches)
        print("  Done")

    # Phase 2: Threshold calibration (restore dense firing rates)
    if args.alpha > 0:
        print(f"\n--- Threshold calibration (α={args.alpha}) ---")
        dense_model = build_fn().to(device).eval()
        dense_model.load_state_dict(ckpt.get('model', ckpt))
        dense_rates = collect_firing_rates(dense_model, train_loader, device, max_batches=64)
        pruned_potentials = collect_membrane_potentials(model, train_loader, device, max_batches=32)
        del dense_model; torch.cuda.empty_cache()

        n_calibrated = 0
        for name, module in model.named_modules():
            if not isinstance(module, _NEURON_TYPES) or name.endswith('.neuron'):
                continue
            if name not in dense_rates or name not in pruned_potentials:
                continue
            target_rate = dense_rates[name]
            if target_rate <= 0 or target_rate >= 1:
                continue
            v_dist = pruned_potentials[name].float()
            if v_dist.numel() > 1_000_000:
                v_dist = v_dist[torch.randperm(v_dist.numel())[:1_000_000]]
            target_vth = torch.quantile(v_dist, 1.0 - target_rate).item()
            inner = _get_inner_neuron(module)
            old_vth = _get_vth(inner)
            _set_vth(inner, old_vth + args.alpha * (target_vth - old_vth))
            n_calibrated += 1
        print(f"  Calibrated {n_calibrated} neurons")

    # Phase 3: BN recalibration again (correct for threshold changes)
    if args.bn_batches > 0 and args.alpha > 0:
        print(f"\n--- BN recalibration pass 2 ({args.bn_batches} batches) ---")
        recalibrate_bn(model, train_loader, device, max_batches=args.bn_batches)
        print("  Done")

    # Save
    threshold_map = {}
    for name, module in model.named_modules():
        if isinstance(module, _NEURON_TYPES) and not name.endswith('.neuron'):
            threshold_map[name] = _get_vth(_get_inner_neuron(module))

    if args.output:
        torch.save({
            'model': model.state_dict(),
            'threshold_map': threshold_map,
            'method': 'spike_aware_obs',
            'params': {
                'scorer': args.scorer, 'lam': args.lam,
                'sigma': args.sigma, 'alpha': args.alpha,
            },
            'stats': dict(stats),
        }, args.output)
        print(f"\nSaved to {args.output}")

    if args.evaluate:
        correct = total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                reset_net(model)
                correct += outputs.argmax(1).eq(targets).sum().item()
                total += targets.size(0)
        print(f"\nEvaluation accuracy: {100.0 * correct / total:.2f}% ({correct}/{total})")
