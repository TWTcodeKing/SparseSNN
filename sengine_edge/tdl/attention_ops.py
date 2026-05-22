"""Fused attention ONNX custom ops for SNN transformers.

Three fused ops that replace 10-15 tiny ONNX nodes (Reshape, Transpose,
Slice, MatMul, Scale) in the attention core with a single custom op node.
This eliminates all layout tracking issues in the sengine runtime.

Pattern: [Conv+BN+LIF projections] → FusedAttention → [Conv+BN projection]

Each op encapsulates: reshape to heads → matmul → scale → LIF → matmul
→ merge heads, producing a single ONNX node that the sengine parser
maps to OpType.FusedAttention.

Follows the same torch.autograd.Function pattern as neuron_ops.py
(FusedLIFPluginOp, FusedIFPluginOp).
"""

import torch


# ---------------------------------------------------------------------------
# LIF helper (shared by all attention cores)
# ---------------------------------------------------------------------------

def _lif_step(v, x_t, tau, v_threshold):
    """Single-step LIF neuron: returns (spike, new_v)."""
    recip = 1.0 / tau
    h = (1.0 - recip) * v + recip * x_t
    spike = (h >= v_threshold).to(x_t.dtype)
    v_new = (1.0 - spike) * h  # hard reset to 0
    return spike, v_new


def _lif_forward_seq(x_seq, T, tau, v_threshold):
    """Multi-step LIF: x_seq (T*B, ...) → spikes (T*B, ...)."""
    B = x_seq.shape[0] // T
    spatial_shape = (B,) + tuple(x_seq.shape[1:])
    v = torch.zeros(spatial_shape, device=x_seq.device, dtype=x_seq.dtype)
    spikes = []
    for t in range(T):
        x_t = x_seq[t * B:(t + 1) * B]
        spike, v = _lif_step(v, x_t, tau, v_threshold)
        spikes.append(spike)
    return torch.cat(spikes, dim=0)


# ---------------------------------------------------------------------------
# SpikFormer: standard Q@K^T attention
# ---------------------------------------------------------------------------

def _spikformer_attn_forward(ctx, q, k, v, T, num_heads, head_dim, scale,
                              attn_lif_tau, attn_lif_v_thresh):
    """SpikFormer attention core.

    Args:
        q, k, v: (TB, N, C) after Linear+BN+LIF projections
        T: timesteps
        num_heads, head_dim: head geometry (C = num_heads * head_dim)
        scale: attention scale factor (scalar)
        attn_lif_tau, attn_lif_v_thresh: LIF neuron params

    Returns:
        (TB, N, C) attention output after attn_lif
    """
    TB = q.shape[0]
    N = q.shape[1]
    C = num_heads * head_dim

    # Reshape to multi-head
    q = q.view(TB, N, num_heads, head_dim).permute(0, 2, 1, 3)
    k = k.view(TB, N, num_heads, head_dim).permute(0, 2, 1, 3)
    v = v.view(TB, N, num_heads, head_dim).permute(0, 2, 1, 3)

    # Attention
    attn = (q @ k.transpose(-2, -1)) * scale   # (TB, heads, N, N)
    x = attn @ v                                 # (TB, heads, N, head_dim)

    # Merge heads
    x = x.transpose(1, 2).reshape(TB, N, C)

    # attn_lif
    x = _lif_forward_seq(x, T, attn_lif_tau, attn_lif_v_thresh)
    return x


class FusedSpikformerAttnOp(torch.autograd.Function):
    """SpikFormer attention core → single ONNX custom op."""
    forward = staticmethod(_spikformer_attn_forward)

    @staticmethod
    def symbolic(g, q, k, v, T, num_heads, head_dim, scale,
                 attn_lif_tau, attn_lif_v_thresh):
        return g.op(
            "FusedSpikformerAttention", q, k, v,
            T_i=T, num_heads_i=num_heads, head_dim_i=head_dim,
            scale_f=scale,
            attn_lif_tau_f=attn_lif_tau,
            attn_lif_v_threshold_f=attn_lif_v_thresh,
        )


# ---------------------------------------------------------------------------
# MaxFormer: linear K^T@V attention (reversed order)
# ---------------------------------------------------------------------------

def _maxformer_attn_forward(ctx, q, k, v, T, num_heads, head_dim, scale,
                             H, W, attn_lif_tau, attn_lif_v_thresh):
    """MaxFormer linear attention core.

    Args:
        q, k, v: (TB, C, H, W) after Conv2d+BN+LIF projections (NCHW)
        T: timesteps
        num_heads, head_dim: head geometry
        scale: attention scale factor
        H, W: spatial dims for reshape back
        attn_lif_tau, attn_lif_v_thresh: LIF neuron params

    Returns:
        (TB, C, H, W) attention output after attn_lif (NCHW)
    """
    TB = q.shape[0]
    C = num_heads * head_dim
    N = H * W

    # Reshape to multi-head: (TB, C, H, W) → (TB, heads, N, head_dim)
    q = q.view(TB, num_heads, head_dim, N).transpose(-2, -1)
    k = k.view(TB, num_heads, head_dim, N).transpose(-2, -1)
    v = v.view(TB, num_heads, head_dim, N).transpose(-2, -1)

    # Linear attention: K^T @ V → Q @ result * scale
    kv = k.transpose(-2, -1) @ v           # (TB, heads, head_dim, head_dim)
    x = (q @ kv) * scale                   # (TB, heads, N, head_dim)

    # Merge heads → 4D
    x = x.transpose(-2, -1).reshape(TB, C, H, W)

    # attn_lif
    x = _lif_forward_seq(x, T, attn_lif_tau, attn_lif_v_thresh)
    return x


class FusedMaxformerAttnOp(torch.autograd.Function):
    """MaxFormer linear attention core → single ONNX custom op."""
    forward = staticmethod(_maxformer_attn_forward)

    @staticmethod
    def symbolic(g, q, k, v, T, num_heads, head_dim, scale,
                 H, W, attn_lif_tau, attn_lif_v_thresh):
        return g.op(
            "FusedMaxformerAttention", q, k, v,
            T_i=T, num_heads_i=num_heads, head_dim_i=head_dim,
            scale_f=scale, H_i=H, W_i=W,
            attn_lif_tau_f=attn_lif_tau,
            attn_lif_v_threshold_f=attn_lif_v_thresh,
        )


# ---------------------------------------------------------------------------
# SpikingResFormer DSSA: deformable split-KV attention
# ---------------------------------------------------------------------------

def _dssa_attn_forward(ctx, y_kv, x_query, scale1, scale2,
                        T, num_heads, head_dim, H_in, W_in,
                        attn_lif_tau, attn_lif_v_thresh):
    """DSSA deformable attention core.

    Args:
        y_kv: (TB, 2C, h', w') from W conv+BN (contains K and V concatenated)
        x_query: (TB, C, H, W) from activation_in LIF (query source)
        scale1: (1, heads, 1, 1) scale for K^T @ Q
        scale2: (1, heads, 1, 1) scale for V @ attn
        T: timesteps
        num_heads, head_dim: head geometry (C = num_heads * head_dim)
        H_in, W_in: input spatial dims (for reshape back)
        attn_lif_tau, attn_lif_v_thresh: LIF neuron params

    Returns:
        (TB, C, H_in, W_in) attention output (NCHW, before activation_out)
    """
    C = num_heads * head_dim
    h_out = y_kv.shape[2]
    w_out = y_kv.shape[3]
    spatial = h_out * w_out
    spatial_q = H_in * W_in

    # Head split for K/V
    y = y_kv.view(-1, num_heads, 2 * head_dim, spatial)
    hd = head_dim
    y1 = y[:, :, :hd, :]           # keys:   (TB, heads, head_dim, spatial)
    y2 = y[:, :, hd:2 * hd, :]     # values: (TB, heads, head_dim, spatial)

    # Query from input
    xq = x_query.view(-1, num_heads, head_dim, spatial_q)

    # Attention
    attn = torch.matmul(y1.transpose(-2, -1), xq)  # (TB, heads, spatial, spatial_q)
    attn = attn * scale1

    # attn_lif
    attn = _lif_forward_seq(attn, T, attn_lif_tau, attn_lif_v_thresh)

    # Output
    out = torch.matmul(y2, attn)    # (TB, heads, head_dim, spatial_q)
    out = out * scale2

    # Reshape back to 4D
    out = out.reshape(-1, C, H_in, W_in).contiguous()
    return out


# ---------------------------------------------------------------------------
# MS_QKFormer: Token Q-K scalar attention (sum + multiply, no MatMul)
# ---------------------------------------------------------------------------

def _token_qk_attn_forward(ctx, q, k, T, num_heads, head_dim,
                             H, W, attn_lif_tau, attn_lif_v_thresh):
    """Token Q-K scalar attention core.

    Args:
        q, k: (TB, C, H, W) after Conv2d+BN+LIF projections (NCHW)
        T: timesteps
        num_heads, head_dim: head geometry
        H, W: spatial dims for reshape back
        attn_lif_tau, attn_lif_v_thresh: LIF neuron params

    Returns:
        (TB, C, H, W) attention output (NCHW)
    """
    TB = q.shape[0]
    C = num_heads * head_dim
    N = H * W

    # Reshape to multi-head
    q = q.view(TB, num_heads, head_dim, N)
    k = k.view(TB, num_heads, head_dim, N)

    # Scalar attention: sum Q over head_dim → LIF → multiply with K
    attn = q.sum(dim=2, keepdim=True)                    # (TB, heads, 1, N)
    attn = _lif_forward_seq(attn, T, attn_lif_tau, attn_lif_v_thresh)
    x = torch.mul(attn, k)                               # (TB, heads, head_dim, N)

    # Merge heads
    x = x.reshape(TB, C, H, W)
    return x


class FusedTokenQKAttnOp(torch.autograd.Function):
    """MS_QKFormer Token Q-K scalar attention → single ONNX custom op."""
    forward = staticmethod(_token_qk_attn_forward)

    @staticmethod
    def symbolic(g, q, k, T, num_heads, head_dim, H, W,
                 attn_lif_tau, attn_lif_v_thresh):
        return g.op(
            "FusedTokenQKAttention", q, k,
            T_i=T, num_heads_i=num_heads, head_dim_i=head_dim,
            H_i=H, W_i=W,
            attn_lif_tau_f=attn_lif_tau,
            attn_lif_v_threshold_f=attn_lif_v_thresh,
        )


class FusedDSSAAttnOp(torch.autograd.Function):
    """DSSA deformable attention core → single ONNX custom op."""
    forward = staticmethod(_dssa_attn_forward)

    @staticmethod
    def symbolic(g, y_kv, x_query, scale1, scale2,
                 T, num_heads, head_dim, H_in, W_in,
                 attn_lif_tau, attn_lif_v_thresh):
        return g.op(
            "FusedDSSAAttention", y_kv, x_query, scale1, scale2,
            T_i=T, num_heads_i=num_heads, head_dim_i=head_dim,
            H_in_i=H_in, W_in_i=W_in,
            attn_lif_tau_f=attn_lif_tau,
            attn_lif_v_threshold_f=attn_lif_v_thresh,
        )
