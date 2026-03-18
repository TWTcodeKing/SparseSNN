"""
Firing rate profiling utilities for SNN models.

Two profiler granularities:
  ChannelFiringRateProfiler  — per-channel (C,) rates; used for channel permutation
  NeuronFiringRateProfiler   — spatial/token-level (C,H,W) or (N,C) rates; used for
                               neuron-aware N:M pruning scoring

Convenience wrappers:
  profile_model_firing_rates   — run ChannelFiringRateProfiler over a dataloader
  profile_neuron_firing_rates  — run NeuronFiringRateProfiler over a dataloader

Rate computation helpers (for Conv2d and Linear scoring):
  compute_effective_rates_conv   — per-kernel-position rates via F.unfold
  compute_enhanced_rates_linear  — variance-boosted channel rates for transformers

Dataloader helper:
  build_stratified_subset_loader — stratified fraction of a dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional

from models.neurons import (
    LIFNeuron, IFNeuron,
    MultiStepLIFNeuron, MultiStepIFNeuron,
)


_NEURON_TYPES = (LIFNeuron, IFNeuron, MultiStepLIFNeuron, MultiStepIFNeuron)


# ---------------------------------------------------------------------------
# Dataloader helper
# ---------------------------------------------------------------------------

def build_stratified_subset_loader(
    dataset,
    fraction: float = 0.1,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
) -> torch.utils.data.DataLoader:
    """Build a dataloader from a stratified subset of the dataset.

    Samples `fraction` of the data while preserving the per-class distribution.

    Args:
        dataset: A torch Dataset with .targets or similar label attribute.
        fraction: Fraction of data to sample (0, 1]. Default 0.1 (10%).
        batch_size: Batch size for the returned loader.
        num_workers: Number of dataloader workers.
        seed: Random seed for reproducibility.

    Returns:
        DataLoader over the stratified subset.
    """
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
    elif hasattr(dataset, 'labels'):
        labels = dataset.labels
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
    else:
        labels = []
        for i in range(len(dataset)):
            _, target = dataset[i]
            if isinstance(target, torch.Tensor):
                target = target.item()
            labels.append(target)

    from collections import defaultdict
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

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


# ---------------------------------------------------------------------------
# Channel-level profiler (used for permutation)
# ---------------------------------------------------------------------------

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
        self._channel_accum: dict[str, list[torch.Tensor]] = OrderedDict()
        self._batch_counts: dict[str, int] = OrderedDict()

    def register_hooks(self) -> int:
        count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, _NEURON_TYPES):
                hook = module.register_forward_hook(self._make_channel_hook(name))
                self.hooks.append(hook)
                self._channel_accum[name] = []
                self._batch_counts[name] = 0
                count += 1
        return count

    def _make_channel_hook(self, name: str):
        def hook_fn(module, inp, out):
            spike = out.detach()
            ndim = spike.ndim
            if ndim == 5:
                rates = spike.mean(dim=(0, 1, 3, 4))   # (C,)
            elif ndim == 4:
                if isinstance(module, (MultiStepLIFNeuron, MultiStepIFNeuron)):
                    rates = spike.mean(dim=(0, 1, 2))   # (C,) from (T,B,N,C)
                else:
                    rates = spike.mean(dim=(0, 2, 3))   # (C,) from (B,C,H,W)
            elif ndim == 3:
                rates = spike.mean(dim=(0, 1))          # (C,)
            elif ndim == 2:
                rates = spike.mean(dim=0)               # (C,)
            else:
                return
            self._channel_accum[name].append(rates.cpu())
            self._batch_counts[name] += 1
        return hook_fn

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_firing_rates(self) -> dict[str, torch.Tensor]:
        """Return mean per-channel firing rates. Returns {layer_name: (C,) tensor}."""
        rates = OrderedDict()
        for name, accum in self._channel_accum.items():
            if not accum:
                continue
            stacked = torch.stack(accum, dim=0)   # (num_batches, C)
            rates[name] = stacked.mean(dim=0)
        return rates

    def get_sorted_channels(self) -> dict[str, torch.Tensor]:
        """Return channel indices sorted by firing rate ascending per layer."""
        rates = self.get_firing_rates()
        return OrderedDict((name, rate.argsort()) for name, rate in rates.items())

    def clear(self):
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
    """Run ChannelFiringRateProfiler and return per-layer channel rates.

    Supports two modes:
    1. Pass a `dataloader` directly (uses max_batches).
    2. Pass `dataset` + `sample_fraction` to build a stratified subset loader.

    Returns:
        {layer_name: (C,) tensor of per-channel firing rates}
    """
    from models.neurons import reset_net

    if dataset is not None and sample_fraction is not None:
        dataloader = build_stratified_subset_loader(
            dataset, fraction=sample_fraction, batch_size=batch_size,
            num_workers=num_workers, seed=seed,
        )
        total_samples = len(dataloader.dataset)
        print(f"Using stratified subset: {total_samples} samples "
              f"({sample_fraction * 100:.1f}% of dataset)")
        max_batches = len(dataloader)

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


# ---------------------------------------------------------------------------
# Spatial/token-level profiler (used for neuron-aware pruning scoring)
# ---------------------------------------------------------------------------

class NeuronFiringRateProfiler:
    """Profile spatial/token-level firing rates from spiking neuron layers.

    Unlike ChannelFiringRateProfiler which reduces to (C,), this profiler
    preserves spatial dimensions:
      - Conv (T,B,C,H,W) -> spatial_rates (C,H,W)
      - Token (T,B,N,C)  -> spatial_rates (N,C)

    Also computes per-channel statistics: channel_mean (C,), channel_std (C,).

    Usage:
        profiler = NeuronFiringRateProfiler(model)
        profiler.register_hooks()
        for images, _ in dataloader:
            model(images)
            reset_net(model)
        rates = profiler.get_neuron_rates()
        profiler.remove_hooks()
    """

    def __init__(self, model: nn.Module, max_spatial: int = 16):
        self.model = model
        self.max_spatial = max_spatial
        self.hooks = []
        self._spatial_accum: dict[str, list[torch.Tensor]] = OrderedDict()
        self._layouts: dict[str, str] = OrderedDict()

    def register_hooks(self) -> int:
        count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, _NEURON_TYPES):
                if name.endswith('.neuron'):
                    parent_name = name.rsplit('.neuron', 1)[0]
                    parent = dict(self.model.named_modules()).get(parent_name)
                    if isinstance(parent, (MultiStepLIFNeuron, MultiStepIFNeuron)):
                        continue
                hook = module.register_forward_hook(self._make_hook(name, module))
                self.hooks.append(hook)
                self._spatial_accum[name] = []
                count += 1
        return count

    def _make_hook(self, name: str, module: nn.Module):
        def hook_fn(mod, inp, out):
            spike = out.detach()
            ndim = spike.ndim

            if ndim == 5:
                rates = spike.mean(dim=(0, 1))   # (C, H, W)
                _, H, W = rates.shape
                if H > self.max_spatial or W > self.max_spatial:
                    rates = F.adaptive_avg_pool2d(
                        rates.unsqueeze(0),
                        (min(H, self.max_spatial), min(W, self.max_spatial))
                    ).squeeze(0)
                self._layouts[name] = 'conv'

            elif ndim == 4:
                if isinstance(mod, (MultiStepLIFNeuron, MultiStepIFNeuron)):
                    rates = spike.mean(dim=(0, 1))   # (N, C)
                    self._layouts[name] = 'token'
                else:
                    rates = spike.mean(dim=0)        # (C, H, W)
                    _, H, W = rates.shape
                    if H > self.max_spatial or W > self.max_spatial:
                        rates = F.adaptive_avg_pool2d(
                            rates.unsqueeze(0),
                            (min(H, self.max_spatial), min(W, self.max_spatial))
                        ).squeeze(0)
                    self._layouts[name] = 'conv'

            elif ndim == 3:
                if isinstance(mod, (MultiStepLIFNeuron, MultiStepIFNeuron)):
                    rates = spike.mean(dim=(0, 1))   # (C,)
                    self._layouts[name] = 'flat'
                else:
                    rates = spike.mean(dim=0)        # (N, C)
                    self._layouts[name] = 'token'

            elif ndim == 2:
                rates = spike.mean(dim=0)            # (C,)
                self._layouts[name] = 'flat'
            else:
                return

            self._spatial_accum[name].append(rates.cpu())
        return hook_fn

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_neuron_rates(self) -> dict[str, dict]:
        """Return per-layer neuron firing rate info.

        Returns:
            {layer_name: {
                'spatial_rates': Tensor,   # (C,H,W), (N,C), or (C,)
                'channel_mean': Tensor,    # (C,)
                'channel_std': Tensor,     # (C,)
                'layout': 'conv'|'token'|'flat'
            }}
        """
        result = OrderedDict()
        for name, accum in self._spatial_accum.items():
            if not accum:
                continue
            stacked = torch.stack(accum, dim=0)   # (num_batches, ...)
            spatial_rates = stacked.mean(dim=0)
            layout = self._layouts.get(name, 'flat')

            if layout == 'conv':
                channel_mean = spatial_rates.mean(dim=(1, 2))   # (C,)
                channel_std = spatial_rates.std(dim=(1, 2))     # (C,)
            elif layout == 'token':
                channel_mean = spatial_rates.mean(dim=0)        # (C,)
                channel_std = spatial_rates.std(dim=0)          # (C,)
            else:
                channel_mean = spatial_rates
                channel_std = torch.zeros_like(spatial_rates)

            result[name] = {
                'spatial_rates': spatial_rates,
                'channel_mean': channel_mean,
                'channel_std': channel_std,
                'layout': layout,
            }
        return result

    def clear(self):
        for name in self._spatial_accum:
            self._spatial_accum[name].clear()
        self._layouts.clear()


def profile_neuron_firing_rates(
    model: nn.Module,
    dataloader,
    device: torch.device,
    max_batches: int = 50,
    max_spatial: int = 16,
) -> dict[str, dict]:
    """Run NeuronFiringRateProfiler and return spatial/token-level rates.

    Args:
        model: SNN model (set to eval mode).
        dataloader: Validation dataloader.
        device: Device for inference.
        max_batches: Number of batches to profile.
        max_spatial: Maximum spatial dimension to retain (larger dims downsampled).

    Returns:
        {layer_name: {spatial_rates, channel_mean, channel_std, layout}}
    """
    from models.neurons import reset_net

    model.eval()
    profiler = NeuronFiringRateProfiler(model, max_spatial=max_spatial)
    num_hooks = profiler.register_hooks()
    print(f"Registered {num_hooks} neuron hooks for spatial firing rate profiling")

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= max_batches:
                break
            images = images.to(device)
            model(images)
            reset_net(model)
            if (i + 1) % 10 == 0:
                print(f"  Profiled {i + 1}/{max_batches} batches")

    rates = profiler.get_neuron_rates()
    profiler.remove_hooks()

    print(f"\nNeuron firing rate summary ({len(rates)} layers):")
    for name, info in rates.items():
        sr = info['spatial_rates']
        cm = info['channel_mean']
        print(f"  {name}: layout={info['layout']}, spatial_shape={tuple(sr.shape)}, "
              f"C={cm.shape[0]}, mean={cm.mean():.4f}, std_mean={info['channel_std'].mean():.4f}")

    return rates


# ---------------------------------------------------------------------------
# Rate computation helpers for pruning scoring
# ---------------------------------------------------------------------------

def compute_effective_rates_conv(
    spatial_rates: torch.Tensor,
    kernel_size: tuple,
    stride: tuple = (1, 1),
    padding: tuple = (0, 0),
    dilation: tuple = (1, 1),
) -> torch.Tensor:
    """Compute per-kernel-position effective firing rates for Conv2d.

    Uses F.unfold to extract patches at each kernel position, giving the
    average firing rate seen by each (channel, kh, kw) pair across all
    valid spatial locations.

    Args:
        spatial_rates: (C_in, H, W) spatial firing rate map.
        kernel_size: (Kh, Kw) of the Conv2d.
        stride: Conv2d stride.
        padding: Conv2d padding.
        dilation: Conv2d dilation.

    Returns:
        (C_in, Kh, Kw) effective firing rate per kernel position.
    """
    C_in, H, W = spatial_rates.shape
    Kh, Kw = kernel_size

    unfolded = F.unfold(
        spatial_rates.unsqueeze(0),
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )   # (1, C_in*Kh*Kw, L)

    unfolded = unfolded.squeeze(0)                    # (C_in*Kh*Kw, L)
    unfolded = unfolded.reshape(C_in, Kh * Kw, -1)   # (C_in, Kh*Kw, L)
    effective = unfolded.mean(dim=2)                  # (C_in, Kh*Kw)
    return effective.reshape(C_in, Kh, Kw)


def compute_enhanced_rates_linear(
    token_rates: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Compute variance-boosted channel rates for transformer Linear layers.

    Channels with high across-token variance carry positional information
    and should be less likely to be pruned.

    Args:
        token_rates: (N, C) per-token firing rates.
        alpha: Weight for variance boost.

    Returns:
        (C,) enhanced firing rates: mean + alpha * std.
    """
    r_mean = token_rates.mean(dim=0)   # (C,)
    r_std = token_rates.std(dim=0)     # (C,)
    return r_mean + alpha * r_std
