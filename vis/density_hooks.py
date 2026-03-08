"""
Computation density analysis for SNN models via forward hooks.

Tracks:
1. Conv/Linear layer density: fraction of effective (non-zero input) multiply operations.
   Since weights are dense, density ≈ input spike firing rate.
2. Attention density (Spikformer SSA): fraction of multiply operations where BOTH
   operands are non-zero in K^T@V (binary x binary) and Q@(K^T@V) (binary x real).
3. Spatial density maps: per-spatial-position computation density, up-sampled back
   to input image resolution for GradCAM-style overlay.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


class LayerDensityRecord:
    """Stores density information for a single layer."""

    def __init__(self, name, layer_type):
        self.name = name
        self.layer_type = layer_type  # 'conv', 'linear', 'attn_kv', 'attn_qr'
        # Per-sample metrics (accumulated over forward passes)
        self.densities = []          # scalar density per sample
        self.spatial_maps = []       # (H, W) spatial density maps (for conv layers)
        self.total_ops = []          # total multiply ops
        self.effective_ops = []      # non-zero multiply ops

    def add(self, density, spatial_map=None, total=0, effective=0):
        self.densities.append(density)
        if spatial_map is not None:
            self.spatial_maps.append(spatial_map)
        self.total_ops.append(total)
        self.effective_ops.append(effective)

    @property
    def mean_density(self):
        return np.mean(self.densities) if self.densities else 0.0

    @property
    def mean_spatial_map(self):
        if not self.spatial_maps:
            return None
        return np.mean(self.spatial_maps, axis=0)


class DensityTracker:
    """Registers forward hooks to track computation density across layers.

    Supports:
    - Spikformer (SSA attention + SPS patch embedding + MLP blocks)
    - SEW-ResNet (conv blocks with IF neurons)
    - MS-ResNet (conv blocks with LIF neurons)

    Usage:
        tracker = DensityTracker(model, img_size=32)
        tracker.register_hooks()
        with torch.no_grad():
            output = model(image)
            reset_net(model)
        tracker.finalize_sample()
        records = tracker.records
        tracker.remove_hooks()
    """

    def __init__(self, model, img_size=32):
        self.model = model
        self.img_size = img_size
        self.records = OrderedDict()
        self.hooks = []
        self._attn_captures = {}  # temporary storage for SSA hooks

    def register_hooks(self):
        """Register forward hooks on all Conv, Linear, and SSA modules."""
        self._register_conv_linear_hooks(self.model, prefix='')
        self._register_ssa_hooks(self.model, prefix='')

    def _register_conv_linear_hooks(self, module, prefix):
        """Hook Conv2d, Conv1d, Linear layers to track input spike density."""
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, (nn.Conv2d, nn.Conv1d)):
                rec = LayerDensityRecord(full_name, 'conv')
                self.records[full_name] = rec
                hook = child.register_forward_hook(
                    self._make_conv_hook(full_name, child))
                self.hooks.append(hook)
            elif isinstance(child, nn.Linear):
                rec = LayerDensityRecord(full_name, 'linear')
                self.records[full_name] = rec
                hook = child.register_forward_hook(
                    self._make_linear_hook(full_name, child))
                self.hooks.append(hook)
            else:
                self._register_conv_linear_hooks(child, full_name)

    def _register_ssa_hooks(self, module, prefix):
        """Hook SSA (Spiking Self-Attention) modules for attention density."""
        from models.spikformer import SSA
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, SSA):
                rec_kv = LayerDensityRecord(f"{full_name}/KtV", 'attn_kv')
                rec_qr = LayerDensityRecord(f"{full_name}/Q@KtV", 'attn_qr')
                self.records[f"{full_name}/KtV"] = rec_kv
                self.records[f"{full_name}/Q@KtV"] = rec_qr
                hook = child.register_forward_hook(
                    self._make_ssa_hook(full_name, child))
                self.hooks.append(hook)
                # Also hook Q/K/V LIF neurons to capture binary spike outputs
                for qkv_name in ['q_lif', 'k_lif', 'v_lif']:
                    lif = getattr(child, qkv_name)
                    cap_key = f"{full_name}/{qkv_name}"
                    hook = lif.register_forward_hook(
                        self._make_capture_hook(cap_key))
                    self.hooks.append(hook)
            else:
                self._register_ssa_hooks(child, full_name)

    def _make_conv_hook(self, name, layer):
        """Create a hook for Conv layers that computes input spike density."""
        def hook_fn(module, inp, out):
            x = inp[0].detach()  # (T*B, C, H, W) or (T*B, C, N)
            weight = module.weight.detach()

            # Input firing rate (fraction of non-zero elements)
            input_nonzero = (x != 0).float()
            density = input_nonzero.mean().item()

            # Weight density (fraction of non-zero weights)
            weight_nonzero_rate = (weight != 0).float().mean().item()
            # True computation density = input_density * weight_density
            # (but weights are almost always dense, so this ≈ input_density)
            true_density = density * weight_nonzero_rate

            # Spatial density map (averaged over batch and channels)
            spatial_map = None
            if x.dim() == 4:  # Conv2d: (TB, C, H, W)
                spatial_density = input_nonzero.mean(dim=1)  # (TB, H, W)
                spatial_map = spatial_density.mean(dim=0).cpu().numpy()  # (H, W)

            # Compute total and effective ops
            if isinstance(module, nn.Conv2d):
                kH, kW = module.kernel_size
                C_in = module.in_channels
                H_out, W_out = out.shape[2], out.shape[3]
                C_out = module.out_channels
                total = C_out * H_out * W_out * C_in * kH * kW * x.shape[0]
            elif isinstance(module, nn.Conv1d):
                kW = module.kernel_size[0]
                C_in = module.in_channels
                N_out = out.shape[2]
                C_out = module.out_channels
                total = C_out * N_out * C_in * kW * x.shape[0]
            else:
                total = x.numel() * weight.shape[0]
            effective = int(total * true_density)

            self.records[name].add(true_density, spatial_map, total, effective)

        return hook_fn

    def _make_linear_hook(self, name, layer):
        """Create a hook for Linear layers."""
        def hook_fn(module, inp, out):
            x = inp[0].detach()
            weight = module.weight.detach()
            input_nonzero = (x != 0).float()
            density = input_nonzero.mean().item()
            weight_density = (weight != 0).float().mean().item()
            true_density = density * weight_density
            total = x.shape[0] * weight.shape[0] * weight.shape[1]
            effective = int(total * true_density)
            self.records[name].add(true_density, None, total, effective)

        return hook_fn

    def _make_capture_hook(self, key):
        """Capture LIF neuron output for later SSA analysis."""
        def hook_fn(module, inp, out):
            self._attn_captures[key] = out.detach()
        return hook_fn

    def _make_ssa_hook(self, name, ssa_module):
        """Hook SSA module to compute attention density from captured Q, K, V."""
        def hook_fn(module, inp, out):
            q_key = f"{name}/q_lif"
            k_key = f"{name}/k_lif"
            v_key = f"{name}/v_lif"

            if q_key not in self._attn_captures:
                return

            # Q, K, V are spike outputs from LIF: shape (T, B, C, N) for 1d
            q_spike = self._attn_captures[q_key]  # (T, B, C, N)
            k_spike = self._attn_captures[k_key]
            v_spike = self._attn_captures[v_key]

            T, B, C, N = q_spike.shape
            num_heads = ssa_module.num_heads
            d_head = C // num_heads

            # Reshape to multi-head format: (T, B, H, N, d_head)
            q = q_spike.transpose(-1, -2).reshape(T, B, N, num_heads, d_head)\
                .permute(0, 1, 3, 2, 4)
            k = k_spike.transpose(-1, -2).reshape(T, B, N, num_heads, d_head)\
                .permute(0, 1, 3, 2, 4)
            v = v_spike.transpose(-1, -2).reshape(T, B, N, num_heads, d_head)\
                .permute(0, 1, 3, 2, 4)

            # --- K^T @ V density ---
            # K: (T,B,H,N,d), V: (T,B,H,N,d)
            # K^T: (T,B,H,d,N), K^T@V: (T,B,H,d,d)
            # For each output[i,j] = sum_n K[n,i]*V[n,j]
            # Effective: K[n,i]!=0 AND V[n,j]!=0
            k_binary = (k != 0).float()
            v_binary = (v != 0).float()
            # k_binary.T @ v_binary gives count of jointly non-zero per (i,j)
            kv_joint = k_binary.transpose(-2, -1) @ v_binary  # (T,B,H,d,d)
            kv_total = N  # total ops per output element
            kv_density = (kv_joint / kv_total).mean().item()

            # Spatial density for K^T@V: per-token (spatial position) contribution
            # For each token n, its contribution to density is proportional to
            # how often it fires in both K and V
            k_fire = k_binary.mean(dim=-1)  # (T,B,H,N) - firing rate per token
            v_fire = v_binary.mean(dim=-1)
            # Joint firing per token: geometric mean of k and v firing rates
            token_density_kv = (k_fire * v_fire).mean(dim=(0, 1, 2))  # (N,)
            H_spatial = int(np.sqrt(N))
            W_spatial = N // H_spatial if H_spatial > 0 else 1
            if H_spatial * W_spatial == N:
                spatial_kv = token_density_kv.cpu().numpy().reshape(H_spatial, W_spatial)
            else:
                spatial_kv = None

            self.records[f"{name}/KtV"].add(kv_density, spatial_kv,
                                            int(kv_total * T * B * num_heads * d_head * d_head),
                                            int(kv_joint.sum().item()))

            # --- Q @ (K^T @ V) density ---
            # Q: binary (T,B,H,N,d), Result: real (T,B,H,d,d)
            # Output[n,j] = sum_i Q[n,i] * Result[i,j]
            # Effective: Q[n,i]!=0 AND Result[i,j]!=0
            kv_result = k.transpose(-2, -1) @ v  # actual K^T@V (real-valued)
            q_binary = (q != 0).float()
            result_binary = (kv_result != 0).float()
            # For each (n, j): count over i where both non-zero
            qr_joint = q_binary @ result_binary  # (T,B,H,N,d)
            qr_total = d_head  # ops per output element
            qr_density = (qr_joint / qr_total).mean().item()

            # Spatial density for Q@result: per-token density
            # Token n's density is how active Q is at that position
            q_fire = q_binary.mean(dim=-1)  # (T,B,H,N)
            token_density_q = q_fire.mean(dim=(0, 1, 2))  # (N,)
            if H_spatial * W_spatial == N:
                spatial_q = token_density_q.cpu().numpy().reshape(H_spatial, W_spatial)
            else:
                spatial_q = None

            self.records[f"{name}/Q@KtV"].add(qr_density, spatial_q,
                                              int(qr_total * T * B * num_heads * N * d_head),
                                              int(qr_joint.sum().item()))

            # Cleanup captures
            for k_ in [q_key, k_key, v_key]:
                self._attn_captures.pop(k_, None)

        return hook_fn

    def finalize_sample(self):
        """Called after processing each sample/batch. No-op for now (records accumulate)."""
        pass

    def clear(self):
        """Clear all accumulated records."""
        for rec in self.records.values():
            rec.densities.clear()
            rec.spatial_maps.clear()
            rec.total_ops.clear()
            rec.effective_ops.clear()
        self._attn_captures.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_summary(self):
        """Return a summary dict: {layer_name: {type, mean_density, total_ops, effective_ops}}."""
        summary = OrderedDict()
        for name, rec in self.records.items():
            if not rec.densities:
                continue
            summary[name] = {
                'type': rec.layer_type,
                'mean_density': rec.mean_density,
                'total_ops': int(np.sum(rec.total_ops)),
                'effective_ops': int(np.sum(rec.effective_ops)),
                'num_samples': len(rec.densities),
            }
        return summary

    def get_spatial_maps(self, target_size=None):
        """Return {layer_name: upsampled_spatial_density_map}.

        If target_size is given, all maps are bilinearly interpolated to that size.
        """
        if target_size is None:
            target_size = (self.img_size, self.img_size)

        maps = OrderedDict()
        for name, rec in self.records.items():
            smap = rec.mean_spatial_map
            if smap is None:
                continue
            # Upsample to target size
            t = torch.from_numpy(smap).float().unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=target_size, mode='bilinear', align_corners=False)
            maps[name] = t.squeeze().numpy()
        return maps


def compute_overall_spatial_density(tracker, img_size):
    """Compute a single aggregate spatial density map across all layers.

    Weights each layer's spatial map by its total operation count, so layers
    with more computation contribute proportionally more.
    """
    maps = tracker.get_spatial_maps(target_size=(img_size, img_size))
    if not maps:
        return None

    weighted_sum = np.zeros((img_size, img_size), dtype=np.float64)
    total_weight = 0.0
    for name, smap in maps.items():
        rec = tracker.records[name]
        w = np.sum(rec.total_ops) if rec.total_ops else 1.0
        weighted_sum += smap * w
        total_weight += w

    if total_weight > 0:
        weighted_sum /= total_weight
    return weighted_sum
