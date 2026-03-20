"""Fused Triton kernels for SNN neuron dynamics (inference-only).

Fuses the entire LIF/IF temporal loop into a single GPU kernel launch:
  - Each program instance handles a contiguous block of spatial neurons
  - Loops over T timesteps inside the kernel, carrying membrane potential in registers
  - Eliminates T * (3-5) separate kernel launches from the Python loop

Supported neuron types:
  - LIF (Leaky Integrate-and-Fire): v = v * decay + x * (1 - decay),  decay = 1 - 1/tau
  - IF  (Integrate-and-Fire):       v = v + x

Reset modes:
  - Hard reset (v_reset is not None): v = (1 - spike) * v + spike * v_reset
  - Soft reset (v_reset is None):     v = v - spike * v_threshold

Usage:
    from iengine.triton_sparse.neuron_kernel import fused_lif_forward, fused_if_forward

    # x_seq: (T, N) flattened input, spike_seq: (T, N) output
    spike_seq = fused_lif_forward(x_seq, tau=2.0, v_threshold=1.0, v_reset=0.0)
    spike_seq = fused_if_forward(x_seq, v_threshold=1.0, v_reset=0.0)

Design follows spikingjelly's FPTT (Forward Pass Through Time) strategy,
implemented in Triton instead of CuPy for portability and auto-tuning.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def _lif_fptt_hard_reset_kernel(
    # Pointers
    x_seq_ptr,      # input:  (T, N) flattened
    spike_seq_ptr,  # output: (T, N) flattened
    # Scalars
    decay: tl.constexpr,       # 1 - 1/tau
    one_minus_decay: tl.constexpr,  # 1/tau (for x scaling)
    v_threshold: tl.constexpr,
    v_reset: tl.constexpr,
    T: tl.constexpr,           # number of timesteps
    N,                         # number of spatial elements (B*C*H*W)
    # Block size
    BLOCK_N: tl.constexpr,
):
    """Fused LIF forward with hard reset. One program per spatial block."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    # Initialize membrane potential
    v = tl.zeros((BLOCK_N,), dtype=tl.float32) + v_reset

    for t in range(T):
        # Load input for this timestep
        x = tl.load(x_seq_ptr + t * N + offs, mask=mask, other=0.0).to(tl.float32)

        # Charge: v = decay * v + (1 - decay) * x
        #       = v * (1 - 1/tau) + x / tau
        v = decay * v + one_minus_decay * x

        # Fire: spike = (v >= v_threshold)
        spike = (v >= v_threshold).to(tl.float32)

        # Hard reset: v = (1 - spike) * v + spike * v_reset
        v = (1.0 - spike) * v + spike * v_reset

        # Store spike
        tl.store(spike_seq_ptr + t * N + offs, spike, mask=mask)


@triton.jit
def _lif_fptt_soft_reset_kernel(
    x_seq_ptr,
    spike_seq_ptr,
    decay: tl.constexpr,
    one_minus_decay: tl.constexpr,
    v_threshold: tl.constexpr,
    T: tl.constexpr,
    N,
    BLOCK_N: tl.constexpr,
):
    """Fused LIF forward with soft reset."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    v = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for t in range(T):
        x = tl.load(x_seq_ptr + t * N + offs, mask=mask, other=0.0).to(tl.float32)
        v = decay * v + one_minus_decay * x
        spike = (v >= v_threshold).to(tl.float32)
        # Soft reset: v = v - spike * v_threshold
        v = v - spike * v_threshold
        tl.store(spike_seq_ptr + t * N + offs, spike, mask=mask)


@triton.jit
def _if_fptt_hard_reset_kernel(
    x_seq_ptr,
    spike_seq_ptr,
    v_threshold: tl.constexpr,
    v_reset: tl.constexpr,
    T: tl.constexpr,
    N,
    BLOCK_N: tl.constexpr,
):
    """Fused IF forward with hard reset."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    v = tl.zeros((BLOCK_N,), dtype=tl.float32) + v_reset

    for t in range(T):
        x = tl.load(x_seq_ptr + t * N + offs, mask=mask, other=0.0).to(tl.float32)
        v = v + x
        spike = (v >= v_threshold).to(tl.float32)
        v = (1.0 - spike) * v + spike * v_reset
        tl.store(spike_seq_ptr + t * N + offs, spike, mask=mask)


@triton.jit
def _if_fptt_soft_reset_kernel(
    x_seq_ptr,
    spike_seq_ptr,
    v_threshold: tl.constexpr,
    T: tl.constexpr,
    N,
    BLOCK_N: tl.constexpr,
):
    """Fused IF forward with soft reset."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    v = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for t in range(T):
        x = tl.load(x_seq_ptr + t * N + offs, mask=mask, other=0.0).to(tl.float32)
        v = v + x
        spike = (v >= v_threshold).to(tl.float32)
        v = v - spike * v_threshold
        tl.store(spike_seq_ptr + t * N + offs, spike, mask=mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def fused_lif_forward(
    x_seq: torch.Tensor,
    tau: float = 2.0,
    v_threshold: float = 1.0,
    v_reset: float = 0.0,
) -> torch.Tensor:
    """Fused LIF forward pass over T timesteps.

    Args:
        x_seq: Input tensor of shape (T, N) or (T, B, ...).
               Will be flattened to (T, N) internally.
        tau: Membrane time constant (decay = 1 - 1/tau).
        v_threshold: Firing threshold.
        v_reset: Reset voltage. Use None for soft reset.

    Returns:
        Spike tensor, same shape as x_seq, with binary values {0, 1}.
    """
    orig_shape = x_seq.shape
    T = orig_shape[0]
    x_flat = x_seq.reshape(T, -1).contiguous()
    N = x_flat.shape[1]

    spike_seq = torch.empty_like(x_flat)

    decay = 1.0 - 1.0 / tau
    one_minus_decay = 1.0 / tau

    BLOCK_N = 1024
    grid = ((N + BLOCK_N - 1) // BLOCK_N,)

    if v_reset is not None:
        _lif_fptt_hard_reset_kernel[grid](
            x_flat, spike_seq,
            decay, one_minus_decay,
            v_threshold, v_reset,
            T, N,
            BLOCK_N=BLOCK_N,
        )
    else:
        _lif_fptt_soft_reset_kernel[grid](
            x_flat, spike_seq,
            decay, one_minus_decay,
            v_threshold,
            T, N,
            BLOCK_N=BLOCK_N,
        )

    return spike_seq.reshape(orig_shape)


def fused_if_forward(
    x_seq: torch.Tensor,
    v_threshold: float = 1.0,
    v_reset: float = 0.0,
) -> torch.Tensor:
    """Fused IF forward pass over T timesteps.

    Args:
        x_seq: Input tensor of shape (T, N) or (T, B, ...).
        v_threshold: Firing threshold.
        v_reset: Reset voltage. Use None for soft reset.

    Returns:
        Spike tensor, same shape as x_seq.
    """
    orig_shape = x_seq.shape
    T = orig_shape[0]
    x_flat = x_seq.reshape(T, -1).contiguous()
    N = x_flat.shape[1]

    spike_seq = torch.empty_like(x_flat)

    BLOCK_N = 1024
    grid = ((N + BLOCK_N - 1) // BLOCK_N,)

    if v_reset is not None:
        _if_fptt_hard_reset_kernel[grid](
            x_flat, spike_seq,
            v_threshold, v_reset,
            T, N,
            BLOCK_N=BLOCK_N,
        )
    else:
        _if_fptt_soft_reset_kernel[grid](
            x_flat, spike_seq,
            v_threshold,
            T, N,
            BLOCK_N=BLOCK_N,
        )

    return spike_seq.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Model-level neuron replacement (inference-only)
# ---------------------------------------------------------------------------

def replace_neuron_forward(model, verbose: bool = True) -> int:
    """Replace MultiStepLIF/IF neuron forward methods with fused Triton kernels.

    This is an inference-only optimization. The replacement forward bypasses
    the Python for-loop and runs the entire T-step dynamics in one kernel.

    Args:
        model: nn.Module containing MultiStepLIFNeuron / MultiStepIFNeuron.
        verbose: Print replacement summary.

    Returns:
        Number of neurons replaced.
    """
    from models.neurons import MultiStepLIFNeuron, MultiStepIFNeuron

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, MultiStepLIFNeuron):
            neuron = module.neuron
            tau = float(neuron.tau.item() if isinstance(neuron.tau, torch.Tensor) else neuron.tau)
            v_threshold = float(neuron.v_threshold.item() if isinstance(neuron.v_threshold, torch.Tensor) else neuron.v_threshold)
            v_reset = neuron.v_reset

            def make_lif_forward(t, vth, vr):
                def forward(self, x_seq):
                    return fused_lif_forward(x_seq, tau=t, v_threshold=vth, v_reset=vr)
                return forward

            import types
            module.forward = types.MethodType(make_lif_forward(tau, v_threshold, v_reset), module)
            count += 1
            if verbose:
                reset_mode = 'soft' if v_reset is None else f'hard(v_reset={v_reset})'
                print(f"  [Triton LIF] {name}: tau={tau}, v_th={v_threshold}, {reset_mode}")

        elif isinstance(module, MultiStepIFNeuron):
            neuron = module.neuron
            v_threshold = float(neuron.v_threshold.item() if isinstance(neuron.v_threshold, torch.Tensor) else neuron.v_threshold)
            v_reset = neuron.v_reset

            def make_if_forward(vth, vr):
                def forward(self, x_seq):
                    return fused_if_forward(x_seq, v_threshold=vth, v_reset=vr)
                return forward

            import types
            module.forward = types.MethodType(make_if_forward(v_threshold, v_reset), module)
            count += 1
            if verbose:
                reset_mode = 'soft' if v_reset is None else f'hard(v_reset={v_reset})'
                print(f"  [Triton IF]  {name}: v_th={v_threshold}, {reset_mode}")

    if verbose:
        print(f"  Replaced {count} neuron(s) with fused Triton kernels")
    return count
