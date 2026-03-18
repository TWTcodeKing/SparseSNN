"""Hyperparameter sweep for neuron-level firing-rate-aware N:M pruning.

Profiles neuron firing rates once, then sweeps over (lam, alpha, scoring)
combinations, evaluating each on the validation set. Results are saved to
a CSV and the best model checkpoint is kept.

Usage:
    python -m sparse.sweep_fr_prune \
        --checkpoint output/spikformer_cifar_cifar100_bs128_lr0.0001_normal/best.pth \
        --config configs/spikformer/spikformer_cifar.yaml \
        --dataset cifar100 --data-root /home/twt/datasets \
        --profile-batches 60 --max-spatial 16 \
        --output-dir sweep_results
"""

import argparse
import copy
import csv
import os
import time
from itertools import product

import torch
import torch.nn as nn

from sparse.fr_prune import (
    profile_neuron_firing_rates,
    apply_neuron_aware_pruning,
)


def evaluate(model, val_loader, device):
    """Run evaluation, return top-1 accuracy percentage."""
    from models.neurons import reset_net

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            reset_net(model)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def load_model(args, ds_cfg, device):
    """Build model and load checkpoint weights. Returns model on device."""
    from tengine.utils import (
        load_model_config, build_model_from_config, build_model,
    )

    if args.config:
        config = load_model_config(args.config)
        config.update(ds_cfg)
        config['T'] = args.T
        model = build_model_from_config(config)
    else:
        model = build_model(args.model, num_classes=ds_cfg['num_classes'], T=args.T)

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)

    return model.to(device)


def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter sweep for neuron-aware N:M pruning')

    # Model / data
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)

    # Profiling
    parser.add_argument('--profile-batches', type=int, default=60)
    parser.add_argument('--max-spatial', type=int, default=16)

    # Sparsity
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--m', type=int, default=4)
    parser.add_argument('--exclude', type=str, nargs='*', default=['head'])

    # Sweep grid (comma-separated values)
    parser.add_argument('--lams', type=str, default='0.0,0.3,0.5,0.7,1.0,1.5,2.0',
                        help='Comma-separated lam values to sweep')
    parser.add_argument('--alphas', type=str, default='0.0,0.3,0.5,0.7,1.0',
                        help='Comma-separated alpha values to sweep')
    parser.add_argument('--scorings', type=str, default='multiplicative,additive',
                        help='Comma-separated scoring methods to sweep')

    # Output
    parser.add_argument('--output-dir', type=str, default='sweep_results',
                        help='Directory to save results CSV and best model')
    parser.add_argument('--save-all', action='store_true',
                        help='Save checkpoint for every combination (not just best)')

    args = parser.parse_args()

    # Parse sweep grid
    lams = [float(x) for x in args.lams.split(',')]
    alphas = [float(x) for x in args.alphas.split(',')]
    scorings = [s.strip() for s in args.scorings.split(',')]

    total_combos = len(lams) * len(alphas) * len(scorings)
    print(f"Sweep grid: {len(lams)} lams x {len(alphas)} alphas x "
          f"{len(scorings)} scorings = {total_combos} combinations")
    print(f"  lams:     {lams}")
    print(f"  alphas:   {alphas}")
    print(f"  scorings: {scorings}")

    # Setup
    from tengine.utils import get_dataset_config, build_dataloaders

    gpu_id = int(args.gpu_ids.split(',')[0])
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    ds_cfg = get_dataset_config(args.dataset)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'sweep_results.csv')

    # Build dataloader
    _, val_loader = build_dataloaders(
        args.dataset, args.data_root, args.batch_size,
        img_size=ds_cfg['img_size'], num_workers=4,
    )

    # Load model once to get base weights and profile
    print("\n=== Loading model and profiling neuron firing rates ===")
    model = load_model(args, ds_cfg, device)

    # Evaluate unpruned baseline
    print("\n=== Evaluating unpruned baseline ===")
    baseline_acc = evaluate(model, val_loader, device)
    print(f"Baseline accuracy: {baseline_acc:.2f}%")

    # Profile neuron rates (done once, reused for all combos)
    neuron_rates = profile_neuron_firing_rates(
        model, val_loader, device,
        max_batches=args.profile_batches,
        max_spatial=args.max_spatial,
    )

    # Save original state_dict for reloading between trials
    original_state = copy.deepcopy(model.state_dict())

    # Run sweep
    results = []
    best_acc = -1.0
    best_combo = None

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['lam', 'alpha', 'scoring', 'accuracy', 'acc_drop', 'time_s'])

        for i, (lam, alpha, scoring) in enumerate(product(lams, alphas, scorings)):
            combo_str = f"lam={lam}, alpha={alpha}, scoring={scoring}"
            print(f"\n{'='*60}")
            print(f"[{i+1}/{total_combos}] {combo_str}")
            print(f"{'='*60}")

            # Restore original weights
            model.load_state_dict(copy.deepcopy(original_state))

            # Prune
            t0 = time.time()
            apply_neuron_aware_pruning(
                model, neuron_rates,
                lam=lam, alpha=alpha, scoring=scoring,
                n=args.n, m=args.m, exclude_names=args.exclude,
            )

            # Evaluate
            acc = evaluate(model, val_loader, device)
            elapsed = time.time() - t0
            drop = baseline_acc - acc

            print(f"  -> Accuracy: {acc:.2f}%  (drop: {drop:+.2f}%)  [{elapsed:.1f}s]")

            row = [lam, alpha, scoring, acc, drop, round(elapsed, 1)]
            results.append(row)
            writer.writerow(row)
            f.flush()

            # Track best
            if acc > best_acc:
                best_acc = acc
                best_combo = (lam, alpha, scoring)
                # Save best model
                torch.save({
                    'model': model.state_dict(),
                    'method': 'neuron_aware',
                    'params': {'lam': lam, 'alpha': alpha, 'scoring': scoring,
                               'n': args.n, 'm': args.m},
                    'accuracy': acc,
                    'baseline_accuracy': baseline_acc,
                }, os.path.join(args.output_dir, 'best_model.pth'))

            # Optionally save every checkpoint
            if args.save_all:
                name = f"model_lam{lam}_alpha{alpha}_{scoring}.pth"
                torch.save({'model': model.state_dict(), 'accuracy': acc},
                           os.path.join(args.output_dir, name))

    # Print final summary sorted by accuracy
    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE — {total_combos} combinations")
    print(f"{'='*60}")
    print(f"Baseline accuracy: {baseline_acc:.2f}%\n")

    results.sort(key=lambda r: r[3], reverse=True)
    print(f"{'Rank':<5} {'lam':<6} {'alpha':<7} {'scoring':<16} {'Acc%':<8} {'Drop':<8}")
    print('-' * 55)
    for rank, row in enumerate(results, 1):
        lam, alpha, scoring, acc, drop, _ = row
        marker = ' <-- BEST' if (lam, alpha, scoring) == best_combo else ''
        print(f"{rank:<5} {lam:<6} {alpha:<7} {scoring:<16} {acc:<8.2f} {drop:<+8.2f}{marker}")

    print(f"\nBest: lam={best_combo[0]}, alpha={best_combo[1]}, "
          f"scoring={best_combo[2]} -> {best_acc:.2f}%")
    print(f"Results CSV: {csv_path}")
    print(f"Best model:  {os.path.join(args.output_dir, 'best_model.pth')}")


if __name__ == '__main__':
    main()
