"""Fused spiking neuron ops for ONNX tracing.

Two export modes controlled by `use_native_onnx`:

  - Native ONNX (default): Emit standard ONNX ops (Slice, Mul, Add, Greater,
    Where, Concat) that any inference compiler can fuse natively. No custom
    plugins needed. TRT's Myelin fuses these with surrounding Conv/BN.

  - Plugin mode: Emit single custom ONNX op nodes (FusedLIFNeuron, etc.)
    consumed by platform-specific plugins (TRT IPluginV3, CoreML custom layer).
    Faster kernel but causes reformat overhead due to plugin opacity.

All ops accept 4D input (T*B, C, H, W) with T known at export time.

Platform-agnostic — pure PyTorch, no inference engine dependencies.
"""

import torch

# Export mode: True = standard ONNX ops (Myelin-fusible), False = custom plugin ops
use_native_onnx = True

# Trace-time batch size (set before export, used for static Slice indices)
# Set by TDLTransform._apply_tdl2() based on the export input_shape.
trace_batch_size = None


def _lif_forward(ctx, x_seq, T, tau, v_threshold, v_reset, hard_reset):
    """Shared LIF forward for both native and plugin modes."""
    B = x_seq.shape[0] // T
    spatial_shape = (B,) + tuple(x_seq.shape[1:])
    v = torch.zeros(spatial_shape, device=x_seq.device, dtype=x_seq.dtype)
    spikes = []
    recip_tau = 1.0 / tau
    one_sub_recip = 1.0 - recip_tau
    for t in range(T):
        x_t = x_seq[t * B:(t + 1) * B]
        h = one_sub_recip * v + recip_tau * x_t
        spike = (h >= v_threshold).to(x_seq.dtype)
        if hard_reset:
            v = (1.0 - spike) * h + spike * v_reset
        else:
            v = h - spike * v_threshold
        spikes.append(spike)
    return torch.cat(spikes, dim=0)


class FusedLIFOp(torch.autograd.Function):
    """LIF neuron → native ONNX ops (Myelin-fusible, zero reformats)."""
    forward = staticmethod(_lif_forward)

    @staticmethod
    def symbolic(g, x_seq, T, tau, v_threshold, v_reset, hard_reset):
        return _lif_native_symbolic(g, x_seq, T, tau, v_threshold,
                                     v_reset, hard_reset)


class FusedLIFPluginOp(torch.autograd.Function):
    """LIF neuron → custom ONNX op (plugin-based, for attention paths)."""
    forward = staticmethod(_lif_forward)

    @staticmethod
    def symbolic(g, x_seq, T, tau, v_threshold, v_reset, hard_reset):
        return g.op(
            "FusedLIFNeuron", x_seq,
            T_i=T, tau_f=tau, v_threshold_f=v_threshold,
            v_reset_f=v_reset, hard_reset_i=int(hard_reset),
        )


def _if_forward(ctx, x_seq, T, v_threshold, v_reset, hard_reset):
    B = x_seq.shape[0] // T
    spatial_shape = (B,) + tuple(x_seq.shape[1:])
    v = torch.zeros(spatial_shape, device=x_seq.device, dtype=x_seq.dtype)
    spikes = []
    for t in range(T):
        x_t = x_seq[t * B:(t + 1) * B]
        h = v + x_t
        spike = (h >= v_threshold).to(x_seq.dtype)
        if hard_reset:
            v = (1.0 - spike) * h + spike * v_reset
        else:
            v = h - spike * v_threshold
        spikes.append(spike)
    return torch.cat(spikes, dim=0)


class FusedIFOp(torch.autograd.Function):
    """IF neuron → native ONNX ops."""
    forward = staticmethod(_if_forward)

    @staticmethod
    def symbolic(g, x_seq, T, v_threshold, v_reset, hard_reset):
        return _if_native_symbolic(g, x_seq, T, v_threshold, v_reset, hard_reset)


class FusedIFPluginOp(torch.autograd.Function):
    """IF neuron → custom ONNX op (plugin)."""
    forward = staticmethod(_if_forward)

    @staticmethod
    def symbolic(g, x_seq, T, v_threshold, v_reset, hard_reset):
        return g.op("FusedIFNeuron", x_seq, T_i=T, v_threshold_f=v_threshold,
                     v_reset_f=v_reset, hard_reset_i=int(hard_reset))


def _ms_forward(ctx, x_seq, T, decay, thresh):
    B = x_seq.shape[0] // T
    spatial_shape = (B,) + tuple(x_seq.shape[1:])
    mem = torch.zeros(spatial_shape, device=x_seq.device, dtype=x_seq.dtype)
    spike = torch.zeros(spatial_shape, device=x_seq.device, dtype=x_seq.dtype)
    spikes = []
    for t in range(T):
        x_t = x_seq[t * B:(t + 1) * B]
        mem = mem * decay * (1.0 - spike) + x_t
        spike = (mem >= thresh).to(x_seq.dtype)
        spikes.append(spike)
    return torch.cat(spikes, dim=0)


class FusedMSOp(torch.autograd.Function):
    """MS neuron → native ONNX ops."""
    forward = staticmethod(_ms_forward)

    @staticmethod
    def symbolic(g, x_seq, T, decay, thresh):
        return _ms_native_symbolic(g, x_seq, T, decay, thresh)


class FusedMSPluginOp(torch.autograd.Function):
    """MS neuron → custom ONNX op (plugin)."""
    forward = staticmethod(_ms_forward)

    @staticmethod
    def symbolic(g, x_seq, T, decay, thresh):
        return g.op("FusedMSNeuron", x_seq, T_i=T, decay_f=decay, thresh_f=thresh)


def _ilif_forward(ctx, x_seq, T, decay, max_level):
    """I-LIF forward: multi-level spike output via round(clamp(mem, 0, max_level))."""
    B = x_seq.shape[0] // T
    spatial_shape = (B,) + tuple(x_seq.shape[1:])
    mem = torch.zeros(spatial_shape, device=x_seq.device, dtype=x_seq.dtype)
    spike = torch.zeros(spatial_shape, device=x_seq.device, dtype=x_seq.dtype)
    spikes = []
    for t in range(T):
        x_t = x_seq[t * B:(t + 1) * B]
        mem = decay * (mem - spike) + x_t
        spike = torch.round(torch.clamp(mem, 0, max_level))
        spikes.append(spike)
    return torch.cat(spikes, dim=0)


class FusedILIFOp(torch.autograd.Function):
    """I-LIF neuron → native ONNX ops (Round + Clip)."""
    forward = staticmethod(_ilif_forward)

    @staticmethod
    def symbolic(g, x_seq, T, decay, max_level):
        return _ilif_native_symbolic(g, x_seq, T, decay, max_level)


class FusedILIFPluginOp(torch.autograd.Function):
    """I-LIF neuron → custom ONNX op (plugin)."""
    forward = staticmethod(_ilif_forward)

    @staticmethod
    def symbolic(g, x_seq, T, decay, max_level):
        return g.op("FusedILIFNeuron", x_seq,
                    T_i=T, decay_f=decay, max_level_i=max_level)


# ---------------------------------------------------------------------------
# Native ONNX symbolic implementations
# ---------------------------------------------------------------------------
# These emit standard ONNX ops (Slice, Mul, Add, Greater, Where, Concat)
# that Myelin/CoreML/Vela can fuse natively — no custom plugin boundary.
# ---------------------------------------------------------------------------

def _const(g, val, dtype=torch.float32):
    """Create a scalar ONNX constant."""
    return g.op("Constant", value_t=torch.tensor([val], dtype=dtype))


def _get_B_node(g, x_seq, T):
    """Get B as an ONNX node: B = shape(x)[0] / T. Returns (B_node, B_int_or_None)."""
    import sengine_edge.tdl.neuron_ops as _self

    # Try static B first
    B_int = None
    if _self.trace_batch_size is not None:
        B_int = _self.trace_batch_size
    else:
        try:
            sizes = x_seq.type().sizes()
            if sizes and sizes[0] is not None:
                B_int = sizes[0] // T
        except Exception:
            pass

    if B_int is not None:
        B_node = g.op("Constant", value_t=torch.tensor([B_int], dtype=torch.long))
        return B_node, B_int
    else:
        # Dynamic B: B = shape[0] / T
        shape = g.op("Shape", x_seq)
        dim0 = g.op("Gather", shape,
                     g.op("Constant", value_t=torch.tensor(0, dtype=torch.long)),
                     axis_i=0)
        dim0 = g.op("Unsqueeze", dim0, axes_i=[0])
        T_node = g.op("Constant", value_t=torch.tensor([T], dtype=torch.long))
        B_node = g.op("Div", dim0, T_node)
        return B_node, None


def _slice_t(g, x_seq, t, B_node, B_int, axes):
    """Slice x_seq[t*B : (t+1)*B] along dim 0."""
    if B_int is not None:
        # Static: use constant indices
        start = g.op("Constant", value_t=torch.tensor([t * B_int], dtype=torch.long))
        end = g.op("Constant", value_t=torch.tensor([(t + 1) * B_int], dtype=torch.long))
    else:
        # Dynamic: compute indices from B node
        t_node = g.op("Constant", value_t=torch.tensor([t], dtype=torch.long))
        t1_node = g.op("Constant", value_t=torch.tensor([t + 1], dtype=torch.long))
        start = g.op("Mul", B_node, t_node)
        end = g.op("Mul", B_node, t1_node)
    return g.op("Slice", x_seq, start, end, axes)


def _lif_native_symbolic(g, x_seq, T, tau, v_threshold, v_reset, hard_reset):
    """Emit LIF as unrolled standard ONNX ops for T timesteps."""
    B_node, B_int = _get_B_node(g, x_seq, T)

    recip_tau = 1.0 / tau
    one_sub = 1.0 - recip_tau
    c_recip = _const(g, recip_tau)
    c_one_sub = _const(g, one_sub)
    c_vth = _const(g, v_threshold)
    c_vrst = _const(g, v_reset)
    c_one = _const(g, 1.0)
    c_zero = _const(g, 0.0)
    axes = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))

    x_0 = _slice_t(g, x_seq, 0, B_node, B_int, axes)
    v = g.op("Mul", x_0, c_zero)

    spike_list = []
    for t in range(T):
        x_t = _slice_t(g, x_seq, t, B_node, B_int, axes)
        h = g.op("Add",
                 g.op("Mul", c_one_sub, v),
                 g.op("Mul", c_recip, x_t))
        cmp = g.op("GreaterOrEqual", h, c_vth)
        spike = g.op("Cast", cmp, to_i=1)
        if hard_reset:
            v = g.op("Add",
                     g.op("Mul", g.op("Sub", c_one, spike), h),
                     g.op("Mul", spike, c_vrst))
        else:
            v = g.op("Sub", h, g.op("Mul", spike, c_vth))
        spike_list.append(spike)

    return g.op("Concat", *spike_list, axis_i=0)


def _if_native_symbolic(g, x_seq, T, v_threshold, v_reset, hard_reset):
    """Emit IF as unrolled standard ONNX ops."""
    B_node, B_int = _get_B_node(g, x_seq, T)

    c_vth = _const(g, v_threshold)
    c_vrst = _const(g, v_reset)
    c_one = _const(g, 1.0)
    c_zero = _const(g, 0.0)
    axes = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))

    x_0 = _slice_t(g, x_seq, 0, B_node, B_int, axes)
    v = g.op("Mul", x_0, c_zero)

    spike_list = []
    for t in range(T):
        x_t = _slice_t(g, x_seq, t, B_node, B_int, axes)
        h = g.op("Add", v, x_t)
        cmp = g.op("GreaterOrEqual", h, c_vth)
        spike = g.op("Cast", cmp, to_i=1)
        if hard_reset:
            v = g.op("Add",
                     g.op("Mul", g.op("Sub", c_one, spike), h),
                     g.op("Mul", spike, c_vrst))
        else:
            v = g.op("Sub", h, g.op("Mul", spike, c_vth))
        spike_list.append(spike)

    return g.op("Concat", *spike_list, axis_i=0)


def _ms_native_symbolic(g, x_seq, T, decay, thresh):
    """Emit MSNeuron as unrolled standard ONNX ops."""
    B_node, B_int = _get_B_node(g, x_seq, T)

    c_decay = _const(g, decay)
    c_thresh = _const(g, thresh)
    c_one = _const(g, 1.0)
    c_zero = _const(g, 0.0)
    axes = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))

    x_0 = _slice_t(g, x_seq, 0, B_node, B_int, axes)
    mem = g.op("Mul", x_0, c_zero)
    spike = g.op("Mul", x_0, c_zero)

    spike_list = []
    for t in range(T):
        x_t = _slice_t(g, x_seq, t, B_node, B_int, axes)
        mem = g.op("Add",
                   g.op("Mul", g.op("Mul", mem, c_decay),
                        g.op("Sub", c_one, spike)),
                   x_t)
        cmp = g.op("GreaterOrEqual", mem, c_thresh)
        spike = g.op("Cast", cmp, to_i=1)
        spike_list.append(spike)

    return g.op("Concat", *spike_list, axis_i=0)


def _ilif_native_symbolic(g, x_seq, T, decay, max_level):
    """Emit I-LIF neuron as unrolled standard ONNX ops (Round + Clip)."""
    B_node, B_int = _get_B_node(g, x_seq, T)

    c_decay = _const(g, decay)
    c_zero = _const(g, 0.0)
    c_max = _const(g, float(max_level))
    axes = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))

    x_0 = _slice_t(g, x_seq, 0, B_node, B_int, axes)
    mem = g.op("Mul", x_0, c_zero)
    spike = g.op("Mul", x_0, c_zero)

    spike_list = []
    for t in range(T):
        x_t = _slice_t(g, x_seq, t, B_node, B_int, axes)
        # mem = decay * (mem - spike) + x_t
        mem = g.op("Add",
                   g.op("Mul", c_decay, g.op("Sub", mem, spike)),
                   x_t)
        # spike = round(clamp(mem, 0, max_level))
        clipped = g.op("Clip", mem, c_zero, c_max)
        spike = g.op("Round", clipped)
        spike_list.append(spike)

    return g.op("Concat", *spike_list, axis_i=0)
