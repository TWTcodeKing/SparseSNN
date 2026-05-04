"""TDL-3: Temporal Attention Decomposition — SSA / QKA 4D rewrites.

Provides 4D-native replacements for SpikFormer and MaxFormer attention modules:

  SpikformerSSA4D  — replaces spikformer.SSA  (Linear QKV, standard matmul order)
  SpikformerMLP4D  — replaces spikformer.MLP  (Linear + BN1d + LIF)
  MaxFormerSSA4D   — replaces maxformer.SSA   (Conv1d QKV, reversed matmul order)
  TokenQKA4D       — replaces maxformer.Token_QK_Attention (scalar Q, no V)

All modules accept (T*B, ...) input with T as a construction parameter.
All 5D temporal reshapes are eliminated — the only temporal state lives
inside fused neuron kernels (patched by TDL-2).

Platform-agnostic — pure PyTorch, no inference engine dependencies.
"""

import torch
import torch.nn as nn


# ── Conv1d↔Conv2d / BN1d↔BN2d conversion helpers ───────────────────────

def _conv1d_to_conv2d(conv1d):
    """Convert nn.Conv1d(k=1) to nn.Conv2d(k=1×1), sharing weights."""
    conv2d = nn.Conv2d(
        conv1d.in_channels, conv1d.out_channels, kernel_size=1,
        stride=1, padding=0, bias=conv1d.bias is not None,
        groups=conv1d.groups,
    )
    conv2d.weight = nn.Parameter(conv1d.weight.unsqueeze(-1))
    if conv1d.bias is not None:
        conv2d.bias = conv1d.bias
    return conv2d


def _bn1d_to_bn2d(bn1d):
    """Convert nn.BatchNorm1d to nn.BatchNorm2d, sharing parameters."""
    bn2d = nn.BatchNorm2d(
        bn1d.num_features, eps=bn1d.eps, momentum=bn1d.momentum,
        affine=bn1d.affine, track_running_stats=bn1d.track_running_stats,
    )
    if bn1d.affine:
        bn2d.weight = bn1d.weight
        bn2d.bias = bn1d.bias
    if bn1d.track_running_stats:
        bn2d.running_mean = bn1d.running_mean
        bn2d.running_var = bn1d.running_var
        bn2d.num_batches_tracked = bn1d.num_batches_tracked
    return bn2d


# ── SpikFormer ──────────────────────────────────────────────────────────


class SpikformerSSA4D(nn.Module):
    """4D-native Spiking Self-Attention for SpikFormer.

    Original SSA operates on (T, B, N, C) with 5D reshapes for neurons.
    This version operates on (T*B, N, C) — neurons are TDL-2 fused (4D).

    Attention pattern: Q/K/V projections → FusedSpikformerAttention → proj
    The attention core (reshape→matmul→scale→matmul→merge→lif) is emitted
    as a single ONNX custom op to avoid layout tracking issues.
    """

    def __init__(self, dim, num_heads, scale, T,
                 q_linear=None, q_bn=None, q_lif=None,
                 k_linear=None, k_bn=None, k_lif=None,
                 v_linear=None, v_bn=None, v_lif=None,
                 attn_lif=None,
                 proj_linear=None, proj_bn=None, proj_lif=None,
                 attn_lif_tau=2.0, attn_lif_v_threshold=1.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = scale
        self.T = T

        self.q_linear = q_linear
        self.q_bn = q_bn
        self.q_lif = q_lif
        self.k_linear = k_linear
        self.k_bn = k_bn
        self.k_lif = k_lif
        self.v_linear = v_linear
        self.v_bn = v_bn
        self.v_lif = v_lif
        self.attn_lif = attn_lif
        self.proj_linear = proj_linear
        self.proj_bn = proj_bn
        self.proj_lif = proj_lif
        # Store LIF params for the fused attention custom op
        self._attn_lif_tau = float(attn_lif_tau)
        self._attn_lif_v_thresh = float(attn_lif_v_threshold)

    @classmethod
    def from_ssa(cls, ssa, T):
        """Convert a 5D spikformer.SSA to 4D, sharing weights."""
        # Extract attn_lif params
        attn_lif_tau = 2.0
        attn_lif_v_thresh = 1.0
        if hasattr(ssa.attn_lif, 'neuron'):
            n = ssa.attn_lif.neuron
            tau = n.tau
            attn_lif_tau = float(tau.item() if isinstance(tau, torch.Tensor) else tau)
            attn_lif_v_thresh = float(n.v_threshold.item()
                                      if isinstance(n.v_threshold, torch.Tensor)
                                      else n.v_threshold)
        return cls(
            dim=ssa.dim, num_heads=ssa.num_heads, scale=ssa.scale, T=T,
            q_linear=ssa.q_linear, q_bn=ssa.q_bn, q_lif=ssa.q_lif,
            k_linear=ssa.k_linear, k_bn=ssa.k_bn, k_lif=ssa.k_lif,
            v_linear=ssa.v_linear, v_bn=ssa.v_bn, v_lif=ssa.v_lif,
            attn_lif=ssa.attn_lif,
            proj_linear=ssa.proj_linear, proj_bn=ssa.proj_bn,
            proj_lif=ssa.proj_lif,
            attn_lif_tau=attn_lif_tau,
            attn_lif_v_threshold=attn_lif_v_thresh,
        )

    def forward(self, x):
        """
        Args:
            x: (T*B, N, C) token sequence

        Returns:
            (T*B, N, C) output
        """
        TB = x.shape[0]
        N = x.shape[1]
        C = self.dim
        head_dim = self.head_dim

        # Q projection: Linear → BN1d → LIF
        q = self.q_linear(x)
        q = self.q_bn(q.transpose(-1, -2)).transpose(-1, -2)
        q = self.q_lif(q)                              # (TB, N, C)

        # K projection
        k = self.k_linear(x)
        k = self.k_bn(k.transpose(-1, -2)).transpose(-1, -2)
        k = self.k_lif(k)

        # V projection
        v = self.v_linear(x)
        v = self.v_bn(v.transpose(-1, -2)).transpose(-1, -2)
        v = self.v_lif(v)

        # Fused attention core: reshape→Q@K^T*scale→@V→merge→attn_lif
        # Emits a single FusedSpikformerAttention ONNX custom op.
        from sengine.tdl.attention_ops import FusedSpikformerAttnOp
        x = FusedSpikformerAttnOp.apply(
            q, k, v, self.T, self.num_heads, self.head_dim, self.scale,
            self._attn_lif_tau, self._attn_lif_v_thresh)

        # Output projection: Linear → BN1d → proj_lif
        x = self.proj_linear(x)
        x = self.proj_bn(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.proj_lif(x)
        return x


class SpikformerMLP4D(nn.Module):
    """4D-native MLP for SpikFormer.

    Original MLP operates on (T, B, N, C) with 5D reshapes for neurons.
    This version operates on (T*B, N, C).
    """

    def __init__(self, c_hidden, c_output, T,
                 fc1_linear=None, fc1_bn=None, fc1_lif=None,
                 fc2_linear=None, fc2_bn=None, fc2_lif=None):
        super().__init__()
        self.c_hidden = c_hidden
        self.c_output = c_output
        self.T = T
        self.fc1_linear = fc1_linear
        self.fc1_bn = fc1_bn
        self.fc1_lif = fc1_lif
        self.fc2_linear = fc2_linear
        self.fc2_bn = fc2_bn
        self.fc2_lif = fc2_lif

    @classmethod
    def from_mlp(cls, mlp, T):
        """Convert a 5D spikformer.MLP to 4D, sharing weights."""
        return cls(
            c_hidden=mlp.c_hidden, c_output=mlp.c_output, T=T,
            fc1_linear=mlp.fc1_linear, fc1_bn=mlp.fc1_bn, fc1_lif=mlp.fc1_lif,
            fc2_linear=mlp.fc2_linear, fc2_bn=mlp.fc2_bn, fc2_lif=mlp.fc2_lif,
        )

    def forward(self, x):
        """
        Args:
            x: (T*B, N, C)
        Returns:
            (T*B, N, C)
        """
        x = self.fc1_linear(x)
        x = self.fc1_bn(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.fc1_lif(x)

        x = self.fc2_linear(x)
        x = self.fc2_bn(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.fc2_lif(x)
        return x


# ── MaxFormer ───────────────────────────────────────────────────────────


class MaxFormerSSA4D(nn.Module):
    """4D-native Spiking Self-Attention for MaxFormer.

    Original SSA operates on (T, B, C, H, W) with 5D reshapes for neurons.
    This version operates on (T*B, C, H, W).

    Attention pattern (reversed / linear attention):
        Q/K/V projections → FusedMaxformerAttention → proj + residual
    The attention core is emitted as a single ONNX custom op.
    """

    def __init__(self, dim, num_heads, scale, T,
                 x_lif=None,
                 q_conv=None, q_bn=None, q_lif=None,
                 k_conv=None, k_bn=None, k_lif=None,
                 v_conv=None, v_bn=None, v_lif=None,
                 attn_lif=None,
                 proj_conv=None, proj_bn=None,
                 attn_lif_tau=2.0, attn_lif_v_threshold=1.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = scale
        self.T = T

        self.x_lif = x_lif
        self.q_conv = q_conv
        self.q_bn = q_bn
        self.q_lif = q_lif
        self.k_conv = k_conv
        self.k_bn = k_bn
        self.k_lif = k_lif
        self.v_conv = v_conv
        self.v_bn = v_bn
        self.v_lif = v_lif
        self.attn_lif = attn_lif
        self.proj_conv = proj_conv
        self.proj_bn = proj_bn
        self._attn_lif_tau = float(attn_lif_tau)
        self._attn_lif_v_thresh = float(attn_lif_v_threshold)

    @classmethod
    def from_ssa(cls, ssa, T):
        """Convert a 5D maxformer.SSA to 4D, sharing weights.

        Conv1d weights are reshaped to Conv2d 1×1, BN1d converted to BN2d,
        so the entire module operates on 4D (TB, C, H, W) without any
        3D reshaping that would confuse ONNX shape inference.
        """
        attn_lif_tau = 2.0
        attn_lif_v_thresh = 1.0
        if hasattr(ssa.attn_lif, 'neuron'):
            n = ssa.attn_lif.neuron
            tau = n.tau
            attn_lif_tau = float(tau.item() if isinstance(tau, torch.Tensor) else tau)
            attn_lif_v_thresh = float(n.v_threshold.item()
                                      if isinstance(n.v_threshold, torch.Tensor)
                                      else n.v_threshold)
        return cls(
            dim=ssa.dim, num_heads=ssa.num_heads, scale=ssa.scale, T=T,
            x_lif=ssa.x_lif,
            q_conv=_conv1d_to_conv2d(ssa.q_conv),
            q_bn=_bn1d_to_bn2d(ssa.q_bn), q_lif=ssa.q_lif,
            k_conv=_conv1d_to_conv2d(ssa.k_conv),
            k_bn=_bn1d_to_bn2d(ssa.k_bn), k_lif=ssa.k_lif,
            v_conv=_conv1d_to_conv2d(ssa.v_conv),
            v_bn=_bn1d_to_bn2d(ssa.v_bn), v_lif=ssa.v_lif,
            attn_lif=ssa.attn_lif,
            proj_conv=_conv1d_to_conv2d(ssa.proj_conv),
            proj_bn=_bn1d_to_bn2d(ssa.proj_bn),
            attn_lif_tau=attn_lif_tau,
            attn_lif_v_threshold=attn_lif_v_thresh,
        )

    def forward(self, x):
        """
        Args:
            x: (T*B, C, H, W)
        Returns:
            (T*B, C, H, W) with residual already added
        """
        H = int(x.shape[2])
        W = int(x.shape[3])
        identity = x

        x = self.x_lif(x)                                # (TB, C, H, W)

        # Q/K/V projections: Conv2d(k=1×1) → BN2d → LIF — all 4D
        q = self.q_lif(self.q_bn(self.q_conv(x)))         # (TB, C, H, W)
        k = self.k_lif(self.k_bn(self.k_conv(x)))
        v = self.v_lif(self.v_bn(self.v_conv(x)))

        # Fused attention core: reshape→K^T@V→Q@result*scale→merge→attn_lif
        from sengine.tdl.attention_ops import FusedMaxformerAttnOp
        x = FusedMaxformerAttnOp.apply(
            q, k, v, self.T, self.num_heads, self.head_dim, self.scale,
            H, W, self._attn_lif_tau, self._attn_lif_v_thresh)

        # Output projection + residual
        x = self.proj_bn(self.proj_conv(x))               # (TB, C, H, W)
        return x + identity


class TokenQKA4D(nn.Module):
    """4D-native Token Q-K Attention for MS_QKFormer.

    Original operates on (T, B, C, H, W) with 5D reshapes.
    This version operates on (T*B, C, H, W).

    Attention pattern: sum(Q, head_dim) → attn_lif → Q * K → proj
    """

    def __init__(self, dim, num_heads, T,
                 proj_lif=None,
                 q_conv=None, q_bn=None, q_lif=None,
                 k_conv=None, k_bn=None, k_lif=None,
                 attn_lif=None,
                 proj_conv=None, proj_bn=None,
                 attn_lif_tau=2.0, attn_lif_v_threshold=1.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.T = T

        self.proj_lif = proj_lif
        self.q_conv = q_conv
        self.q_bn = q_bn
        self.q_lif = q_lif
        self.k_conv = k_conv
        self.k_bn = k_bn
        self.k_lif = k_lif
        self.attn_lif = attn_lif
        self.proj_conv = proj_conv
        self.proj_bn = proj_bn
        self._attn_lif_tau = float(attn_lif_tau)
        self._attn_lif_v_thresh = float(attn_lif_v_threshold)

    @classmethod
    def from_qka(cls, qka, T):
        """Convert a 5D maxformer.Token_QK_Attention to 4D, sharing weights."""
        attn_lif_tau = 2.0
        attn_lif_v_thresh = 1.0
        if hasattr(qka.attn_lif, 'neuron'):
            n = qka.attn_lif.neuron
            tau = n.tau
            attn_lif_tau = float(tau.item() if isinstance(tau, torch.Tensor) else tau)
            attn_lif_v_thresh = float(n.v_threshold.item()
                                      if isinstance(n.v_threshold, torch.Tensor)
                                      else n.v_threshold)
        return cls(
            dim=qka.dim, num_heads=qka.num_heads, T=T,
            proj_lif=qka.proj_lif,
            q_conv=_conv1d_to_conv2d(qka.q_conv),
            q_bn=_bn1d_to_bn2d(qka.q_bn), q_lif=qka.q_lif,
            k_conv=_conv1d_to_conv2d(qka.k_conv),
            k_bn=_bn1d_to_bn2d(qka.k_bn), k_lif=qka.k_lif,
            attn_lif=qka.attn_lif,
            proj_conv=_conv1d_to_conv2d(qka.proj_conv),
            proj_bn=_bn1d_to_bn2d(qka.proj_bn),
            attn_lif_tau=attn_lif_tau,
            attn_lif_v_threshold=attn_lif_v_thresh,
        )

    def forward(self, x):
        """
        Args:
            x: (T*B, C, H, W)
        Returns:
            (T*B, C, H, W) with residual already added
        """
        H = int(x.shape[2])
        W = int(x.shape[3])
        identity = x

        x = self.proj_lif(x)                              # (TB, C, H, W)

        # Q/K: Conv2d(k=1×1) → BN2d → LIF — all 4D
        q = self.q_lif(self.q_bn(self.q_conv(x)))          # (TB, C, H, W)
        k = self.k_lif(self.k_bn(self.k_conv(x)))

        # Fused scalar attention: sum(Q,head_dim) → LIF → mul(attn, K) → merge
        from sengine.tdl.attention_ops import FusedTokenQKAttnOp
        x = FusedTokenQKAttnOp.apply(
            q, k, self.T, self.num_heads, self.head_dim,
            H, W, self._attn_lif_tau, self._attn_lif_v_thresh)

        # proj → residual
        x = self.proj_bn(self.proj_conv(x))
        return x + identity
