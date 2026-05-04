"""TDL-3: Temporal Attention Decomposition — DSSA 4D rewrite.

Provides DSSA4D, a drop-in replacement for SpikingResformer's DSSA module
that operates entirely in 4D (T*B, C, H, W). Eliminates all 5D tensor
operations, enabling hardware inference compilers to propagate NHWC format
through the attention block without layout reformats.

The key insight: DSSA's attention matmul (Q^T @ K, V @ attn) operates
independently per frame — no cross-timestep interaction. The only temporal
ops (LIF neurons, firing rate EMA) are handled by TDL-2 fused kernels
or are frozen constants at inference time.

Platform-agnostic — pure PyTorch, no inference engine dependencies.
"""

import torch
import torch.nn as nn


class DSSA4D(nn.Module):
    """4D-native Deformable Spike-driven Self-Attention.

    Accepts (T*B, C, H, W) input with T as a construction parameter.
    All internal operations stay in 4D — no 5D tensors at any point.

    At inference time, this is numerically identical to the original DSSA
    because all operations are per-frame (temporally decomposable).
    """

    def __init__(self, dim, num_heads, lenth, T,
                 activation_in=None, activation_attn=None, activation_out=None,
                 W=None, norm=None, Wproj=None, norm_proj=None,
                 scale1=None, scale2=None,
                 attn_lif_tau=2.0, attn_lif_v_threshold=1.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.lenth = lenth
        self.T = T
        self._attn_lif_tau = float(attn_lif_tau)
        self._attn_lif_v_thresh = float(attn_lif_v_threshold)

        # Neurons (will be patched by TDL-2 with fused ops)
        self.activation_in = activation_in
        self.activation_attn = activation_attn
        self.activation_out = activation_out

        # Stateless layers — plain nn.Conv2d / nn.BatchNorm2d (not wrapped)
        self.W = W
        self.norm = norm
        self.Wproj = Wproj
        self.norm_proj = norm_proj

        # Precomputed scale tensors from frozen firing rate buffers
        # shape: (1, num_heads, 1, 1) for broadcasting with 4D attention
        if scale1 is not None:
            self.register_buffer('scale1', scale1)
        if scale2 is not None:
            self.register_buffer('scale2', scale2)

    @classmethod
    def from_dssa(cls, dssa, T):
        """Convert a 5D DSSA module to 4D DSSA4D, sharing weights.

        Args:
            dssa: Original DSSA module (models.spikingresformer.DSSA)
            T: Number of timesteps
        """
        dim = dssa.dim
        num_heads = dssa.num_heads
        lenth = dssa.lenth

        # Extract plain Conv2d from _MultiStepConv2d wrappers
        # _MultiStepConv2d inherits from nn.Conv2d — it IS a Conv2d
        # We just need to call nn.Conv2d.forward on it (skip the 5D reshape)
        W_conv = dssa.W       # _MultiStepConv2d (is nn.Conv2d)
        Wproj_conv = dssa.Wproj  # Conv1x1 (is nn.Conv2d)

        # Extract plain BN from BN wrapper (BN has .bn attribute)
        norm_bn = dssa.norm.bn       # nn.BatchNorm2d
        norm_proj_bn = dssa.norm_proj.bn  # nn.BatchNorm2d

        # Precompute scale tensors from frozen firing rate buffers
        # Original: (1, 1, num_heads, 1, 1) → 4D: (1, num_heads, 1, 1)
        fr_x = dssa.firing_rate_x.squeeze(0).squeeze(0)  # → (num_heads, 1, 1)
        scale1 = 1.0 / torch.sqrt(fr_x.clamp(min=1e-8) * (dim // num_heads))
        scale1 = scale1.unsqueeze(0)  # (1, num_heads, 1, 1)

        fr_attn = dssa.firing_rate_attn.squeeze(0).squeeze(0)
        scale2 = 1.0 / torch.sqrt(fr_attn.clamp(min=1e-8) * lenth)
        scale2 = scale2.unsqueeze(0)  # (1, num_heads, 1, 1)

        # Extract activation_attn LIF params
        attn_lif_tau = 2.0
        attn_lif_v_thresh = 1.0
        if hasattr(dssa.activation_attn, 'neuron'):
            n = dssa.activation_attn.neuron
            tau = n.tau
            attn_lif_tau = float(tau.item() if isinstance(tau, torch.Tensor) else tau)
            attn_lif_v_thresh = float(n.v_threshold.item()
                                      if isinstance(n.v_threshold, torch.Tensor)
                                      else n.v_threshold)

        return cls(
            dim=dim, num_heads=num_heads, lenth=lenth, T=T,
            activation_in=dssa.activation_in,
            activation_attn=dssa.activation_attn,
            activation_out=dssa.activation_out,
            W=W_conv, norm=norm_bn, Wproj=Wproj_conv, norm_proj=norm_proj_bn,
            scale1=scale1, scale2=scale2,
            attn_lif_tau=attn_lif_tau, attn_lif_v_threshold=attn_lif_v_thresh,
        )

    def forward(self, x):
        """
        Args:
            x: (T*B, C, H, W) input tensor

        Returns:
            (T*B, C, H, W) output tensor
        """
        C = self.dim
        H_in = int(x.shape[2])
        W_in = int(x.shape[3])

        x_feat = x.clone()

        # LIF neuron — operates on 4D via TDL-2
        x = self.activation_in(x)

        # Conv + BN — produces (TB, 2C, h', w') with K and V concatenated
        y = nn.Conv2d.forward(self.W, x)
        y = self.norm(y)

        # Fused attention core: split K/V → K^T@Q*scale1 → LIF → V@attn*scale2 → reshape
        from sengine.tdl.attention_ops import FusedDSSAAttnOp
        out = FusedDSSAAttnOp.apply(
            y, x, self.scale1, self.scale2,
            self.T, self.num_heads, self.head_dim, H_in, W_in,
            self._attn_lif_tau, self._attn_lif_v_thresh)

        # LIF neuron + projection + residual
        out = self.activation_out(out)
        out = nn.Conv2d.forward(self.Wproj, out)
        out = self.norm_proj(out)
        out = out + x_feat

        return out
