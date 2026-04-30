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
                 scale1=None, scale2=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.lenth = lenth
        self.T = T

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

        return cls(
            dim=dim, num_heads=num_heads, lenth=lenth, T=T,
            activation_in=dssa.activation_in,
            activation_attn=dssa.activation_attn,
            activation_out=dssa.activation_out,
            W=W_conv, norm=norm_bn, Wproj=Wproj_conv, norm_proj=norm_proj_bn,
            scale1=scale1, scale2=scale2,
        )

    def forward(self, x):
        """
        Args:
            x: (T*B, C, H, W) input tensor

        Returns:
            (T*B, C, H, W) output tensor
        """
        C = self.dim
        head_dim = self.head_dim

        x_feat = x.clone()

        # LIF neuron — operates on 4D via TDL-2
        x = self.activation_in(x)

        # Conv + BN — plain 4D ops
        y = nn.Conv2d.forward(self.W, x)
        y = self.norm(y)

        # Head split: (TB, 2C, h', w') → (TB, heads, 2*head_dim, spatial)
        # Use explicit shape[0] for batch dim so TRT can track it
        # through Slice/Concat (native ONNX neurons) + reshapes.
        TB = y.shape[0]
        h_out, w_out = y.shape[2], y.shape[3]
        spatial = h_out * w_out
        y = y.view(TB, self.num_heads, 2 * head_dim, spatial)
        y1 = y[:, :, :head_dim, :]      # keys
        y2 = y[:, :, head_dim:, :]       # values

        # Query: (TB, C, H, W) → (TB, heads, head_dim, spatial_q)
        TB_x = x.shape[0]
        spatial_q = x.shape[2] * x.shape[3]
        xq = x.view(TB_x, self.num_heads, head_dim, spatial_q)

        # Attention — per-frame matmul
        attn = torch.matmul(y1.transpose(-2, -1), xq)
        attn = attn * self.scale1

        # LIF neuron on attention scores
        attn = self.activation_attn(attn)

        # Output: (TB, heads, head_dim, spatial) @ (TB, heads, spatial, spatial)
        out = torch.matmul(y2, attn)
        out = out * self.scale2

        # Reshape back to 4D (TB, C, H, W) for Conv projection.
        # Use contiguous reshape so ONNX tracer sees a clean 4D tensor
        # (not a view alias of the multi-head layout).
        H_in = x_feat.shape[2]
        W_in = x_feat.shape[3]
        out = out.reshape(out.shape[0], C, H_in, W_in).contiguous()

        # LIF neuron + projection + residual
        out = self.activation_out(out)
        out = nn.Conv2d.forward(self.Wproj, out)
        out = self.norm_proj(out)
        out = out + x_feat

        return out
