"""
Per-channel firing rate profiling for SNN spiking neuron layers.

Registers forward hooks on LIF/IF neuron modules to record per-channel
spike firing rates. These rates are used to compute optimal channel
permutations that align low-firing channels with 2:4 pruning positions.
"""

import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Optional
from models.neurons import (
    LIFNeuron, IFNeuron,
    MultiStepLIFNeuron, MultiStepIFNeuron,
)


_NEURON_TYPES = (LIFNeuron, IFNeuron, MultiStepLIFNeuron, MultiStepIFNeuron)


def build_stratified_subset_loader(
    dataset,
    fraction: float = 0.1,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
) -> torch.utils.data.DataLoader:
    """Build a dataloader from a stratified subset of the dataset.

    Samples `fraction` of the data while preserving the per-class distribution.
    This gives representative firing rate estimates without running on the
    entire validation set.

    Args:
        dataset: A torch Dataset with .targets or similar label attribute.
        fraction: Fraction of data to sample (0, 1]. Default 0.1 (10%).
        batch_size: Batch size for the returned loader.
        num_workers: Number of dataloader workers.
        seed: Random seed for reproducibility.

    Returns:
        DataLoader over the stratified subset.
    """
    # Extract labels
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
    elif hasattr(dataset, 'labels'):
        labels = dataset.labels
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
    else:
        # Fallback: scan dataset (slow but works for any dataset)
        labels = []
        for i in range(len(dataset)):
            _, target = dataset[i]
            if isinstance(target, torch.Tensor):
                target = target.item()
            labels.append(target)

    # Group indices by class
    from collections import defaultdict
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    # Sample fraction per class (at least 1 per class)
    generator = torch.Generator().manual_seed(seed)
    selected_indices = []
    for cls, indices in sorted(class_indices.items()):
        k = max(1, int(len(indices) * fraction))
        perm = torch.randperm(len(indices), generator=generator)[:k]
        selected_indices.extend([indices[i] for i in perm.tolist()])

    subset = torch.utils.data.Subset(dataset, selected_indices)
    return torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


class ChannelFiringRateProfiler:
    """Register hooks on spiking neuron layers to record per-channel firing rates.

    For each LIF/IF neuron output (binary spike tensor), compute:
      rate[c] = mean(spike[:, :, c, ...]) across batch, timesteps, spatial dims

    Accumulate across multiple batches for stable statistics.

    Usage:
        profiler = ChannelFiringRateProfiler(model)
        profiler.register_hooks()
        for images, _ in dataloader:
            model(images)
            reset_net(model)
        rates = profiler.get_firing_rates()
        profiler.remove_hooks()
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        # {layer_name: list of (C,) tensors, one per batch}
        self._channel_accum: dict[str, list[torch.Tensor]] = OrderedDict()
        self._batch_counts: dict[str, int] = OrderedDict()

    def register_hooks(self) -> int:
        """Register forward hooks on all spiking neuron layers.

        Returns:
            Number of hooks registered.
        """
        count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, _NEURON_TYPES):
                hook = module.register_forward_hook(
                    self._make_channel_hook(name)
                )
                self.hooks.append(hook)
                self._channel_accum[name] = []
                self._batch_counts[name] = 0
                count += 1
        return count

    def _make_channel_hook(self, name: str):
        """Create a forward hook that records per-channel firing rates."""

        def hook_fn(module, inp, out):
            spike = out.detach()

            # Detect tensor layout and compute per-channel mean firing rate.
            # MultiStep neurons output (T, B, ...) tensors.
            # Possible shapes:
            #   (T, B, C, H, W) - convolutional layers (SPS, ResNet)
            #   (T, B, N, C)    - transformer token layers (SSA, MLP)
            #   (B, C, H, W)    - single-step conv
            #   (B, N, C)       - single-step token
            #   (B, C)          - single-step FC

            ndim = spike.ndim

            if ndim == 5:
                # (T, B, C, H, W) - average over T, B, H, W; keep C
                rates = spike.mean(dim=(0, 1, 3, 4))  # (C,)
            elif ndim == 4:
                # Could be (T, B, N, C) or (B, C, H, W)
                # Heuristic: if last dim is small relative to dim 2, likely
                # (T, B, N, C) where C < N. But more reliably: check if
                # this is a MultiStep neuron (output has T leading dim).
                if isinstance(module, (MultiStepLIFNeuron, MultiStepIFNeuron)):
                    # (T, B, N, C) - transformer layout
                    rates = spike.mean(dim=(0, 1, 2))  # (C,)
                else:
                    # (B, C, H, W) - conv layout
                    rates = spike.mean(dim=(0, 2, 3))  # (C,)
            elif ndim == 3:
                # (B, N, C) or (T, B, C)
                if isinstance(module, (MultiStepLIFNeuron, MultiStepIFNeuron)):
                    # (T, B, C)
                    rates = spike.mean(dim=(0, 1))  # (C,)
                else:
                    # (B, N, C) - treat last dim as channel
                    rates = spike.mean(dim=(0, 1))  # (C,)
            elif ndim == 2:
                # (B, C)
                rates = spike.mean(dim=0)  # (C,)
            else:
                # Unexpected shape, skip
                return

            self._channel_accum[name].append(rates.cpu())
            self._batch_counts[name] += 1

        return hook_fn

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_firing_rates(self) -> dict[str, torch.Tensor]:
        """Return mean per-channel firing rates for each neuron layer.

        Returns:
            {layer_name: (C,) tensor of firing rates}
        """
        rates = OrderedDict()
        for name, accum in self._channel_accum.items():
            if not accum:
                continue
            # Stack all batch results and average
            stacked = torch.stack(accum, dim=0)  # (num_batches, C)
            rates[name] = stacked.mean(dim=0)
        return rates

    def get_sorted_channels(self) -> dict[str, torch.Tensor]:
        """Return channel indices sorted by firing rate (ascending) per layer.

        Returns:
            {layer_name: (C,) LongTensor of indices, lowest-firing first}
        """
        rates = self.get_firing_rates()
        sorted_channels = OrderedDict()
        for name, rate in rates.items():
            sorted_channels[name] = rate.argsort()
        return sorted_channels

    def clear(self):
        """Clear all accumulated data."""
        for name in self._channel_accum:
            self._channel_accum[name].clear()
            self._batch_counts[name] = 0


def profile_model_firing_rates(
    model: nn.Module,
    dataloader,
    device: torch.device,
    max_batches: int = 50,
    dataset: Optional[torch.utils.data.Dataset] = None,
    sample_fraction: Optional[float] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Convenience function: run profiling and return per-layer channel rates.

    Supports two modes:
    1. Pass a `dataloader` directly (original behavior, uses max_batches).
    2. Pass `dataset` + `sample_fraction` to build a stratified subset loader
       that preserves class distribution while using only a fraction of data.

    Args:
        model: The SNN model (already loaded with weights, in eval mode).
        dataloader: Validation/test dataloader. Ignored if dataset +
            sample_fraction are provided.
        device: Device to run inference on.
        max_batches: Maximum number of batches to profile (for speed).
            Only used when dataloader is provided without sample_fraction.
        dataset: The full validation dataset. When provided with
            sample_fraction, a stratified subset loader is built from it.
        sample_fraction: Fraction of dataset to sample (0, 1].
            E.g. 0.1 uses 10% of data, stratified by class.
        batch_size: Batch size for the stratified subset loader.
        num_workers: Number of dataloader workers for stratified loader.
        seed: Random seed for stratified sampling reproducibility.

    Returns:
        {layer_name: (C,) tensor of per-channel firing rates}
    """
    from models.neurons import reset_net

    # Build stratified subset loader if requested
    if dataset is not None and sample_fraction is not None:
        dataloader = build_stratified_subset_loader(
            dataset, fraction=sample_fraction, batch_size=batch_size,
            num_workers=num_workers, seed=seed,
        )
        total_samples = len(dataloader.dataset)
        print(f"Using stratified subset: {total_samples} samples "
              f"({sample_fraction * 100:.1f}% of dataset)")
        max_batches = len(dataloader)  # use all batches of the subset

    model.eval()
    profiler = ChannelFiringRateProfiler(model)
    num_hooks = profiler.register_hooks()
    print(f"Registered {num_hooks} neuron hooks for firing rate profiling")

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= max_batches:
                break
            images = images.to(device)
            model(images)
            reset_net(model)
            if (i + 1) % 10 == 0:
                print(f"  Profiled {i + 1}/{max_batches} batches")

    rates = profiler.get_firing_rates()
    profiler.remove_hooks()

    print(f"\nFiring rate summary ({len(rates)} layers):")
    for name, rate in rates.items():
        print(f"  {name}: C={rate.shape[0]}, "
              f"mean={rate.mean():.4f}, min={rate.min():.4f}, max={rate.max():.4f}")

    return rates


# Re-export from canonical location for backward compatibility
from sparse.permutation import compute_permutation_for_n_m, compute_permutation_for_2_4


def _find_upstream_linear(model: nn.Module, neuron_name: str) -> tuple:
    """Find the Linear layer that feeds into a given neuron layer.

    Looks for patterns like 'block.0.attn.q_lif' -> 'block.0.attn.q_linear'
    or 'block.0.mlp.fc1_lif' -> 'block.0.mlp.fc1_linear'.

    Returns:
        (linear_name, linear_module) or (None, None) if not found.
    """
    # Common naming patterns: replace '_lif' with '_linear', 'lif' with 'linear'
    candidates = []
    if neuron_name.endswith('_lif'):
        candidates.append(neuron_name.rsplit('_lif', 1)[0] + '_linear')
    elif neuron_name.endswith('.lif'):
        candidates.append(neuron_name.rsplit('.lif', 1)[0] + '.linear')

    # Also try replacing 'neuron' -> 'linear'
    if 'neuron' in neuron_name:
        candidates.append(neuron_name.replace('neuron', 'linear'))

    modules_dict = dict(model.named_modules())
    for cand in candidates:
        if cand in modules_dict and isinstance(modules_dict[cand], nn.Linear):
            return cand, modules_dict[cand]

    return None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Profile per-channel firing rates of spiking neuron layers')
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config file for transformer models')
    parser.add_argument('--model', type=str, default=None,
                        help='ResNet model name (e.g. sew_resnet18)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name: cifar10, cifar100, imagenet, cifar10dvs')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to dataset root directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='GPU IDs (e.g. 0)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for profiling')
    parser.add_argument('--max-batches', type=int, default=50,
                        help='Maximum number of batches to profile (ignored if --sample-fraction is set)')
    parser.add_argument('--sample-fraction', type=float, default=None,
                        help='Fraction of validation set to sample (e.g. 0.1 for 10%%). '
                             'Uses stratified sampling to preserve class distribution. '
                             'When set, --max-batches is ignored.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for stratified sampling')
    parser.add_argument('--T', type=int, default=4,
                        help='Number of timesteps')
    parser.add_argument('--output', type=str, default='firing_rates.pt',
                        help='Output file path for firing rates (.pt)')
    args = parser.parse_args()

    import yaml
    from tengine.utils import (
        load_model_config, build_model_from_config, build_model,
        get_dataset_config, build_dataloaders, load_checkpoint,
    )
    from models.neurons import reset_net

    # Device
    gpu_id = int(args.gpu_ids.split(',')[0])
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    # Dataset config
    ds_cfg = get_dataset_config(args.dataset)

    # Build model
    if args.config:
        config = load_model_config(args.config)
        config.update(ds_cfg)
        config['T'] = args.T
        model = build_model_from_config(config)
    elif args.model:
        model = build_model(args.model, num_classes=ds_cfg['num_classes'], T=args.T)
    else:
        parser.error('Must specify either --config or --model')

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    # Build dataloader (validation split)
    _, val_loader = build_dataloaders(
        args.dataset, args.data_root, args.batch_size,
        img_size=ds_cfg['img_size'], num_workers=4,
    )

    # Profile — use stratified subset if --sample-fraction is set
    if args.sample_fraction is not None:
        rates = profile_model_firing_rates(
            model, val_loader, device,
            dataset=val_loader.dataset,
            sample_fraction=args.sample_fraction,
            batch_size=args.batch_size,
            num_workers=4,
            seed=args.seed,
        )
    else:
        rates = profile_model_firing_rates(model, val_loader, device, args.max_batches)

    # Compute permutations
    permutations = {}
    for name, rate in rates.items():
        perm = compute_permutation_for_2_4(rate)
        permutations[name] = perm

    # Save
    output = {
        'firing_rates': {k: v for k, v in rates.items()},
        'permutations': permutations,
        'config': {
            'dataset': args.dataset,
            'checkpoint': args.checkpoint,
            'max_batches': args.max_batches,
            'T': args.T,
        },
    }
    torch.save(output, args.output)
    print(f"\nSaved firing rates and permutations to {args.output}")
