"""
Spiking neuron models - standalone implementations without spikingjelly dependency.

Provides LIF (Leaky Integrate-and-Fire) and IF (Integrate-and-Fire) neurons
with surrogate gradient functions for direct training of SNNs.
"""

import torch
import torch.nn as nn
import math


# ---------------------------------------------------------------------------
# Surrogate gradient functions
# ---------------------------------------------------------------------------

class ATan(torch.autograd.Function):
    """Surrogate gradient using arctan function."""
    alpha = 2.0

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.ge(0.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        alpha = ATan.alpha
        grad = alpha / 2 / (1 + (math.pi / 2 * alpha * x).pow(2)) * grad_output
        return grad


class Sigmoid(torch.autograd.Function):
    """Surrogate gradient using sigmoid function."""
    alpha = 4.0

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.ge(0.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        alpha = Sigmoid.alpha
        sgax = (x * alpha).sigmoid()
        grad = grad_output * (1 - sgax) * sgax * alpha
        return grad


class GateGrad(torch.autograd.Function):
    """Surrogate gradient using rectangular window (used in some SNN papers)."""
    lens = 0.5

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.ge(0.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad = grad_output * (x.abs() < GateGrad.lens).float() / (2 * GateGrad.lens)
        return grad


# Convenient wrapper
def heaviside(x, surrogate='atan'):
    """Apply Heaviside step with surrogate gradient."""
    if surrogate == 'atan':
        return ATan.apply(x)
    elif surrogate == 'sigmoid':
        return Sigmoid.apply(x)
    elif surrogate == 'gate':
        return GateGrad.apply(x)
    else:
        raise ValueError(f"Unknown surrogate: {surrogate}")


# ---------------------------------------------------------------------------
# Neuron models
# ---------------------------------------------------------------------------

class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron.

    v[t] = tau * v[t-1] + x[t]
    spike[t] = Heaviside(v[t] - v_threshold)
    v[t] = v[t] * (1 - spike[t])   (hard reset)
      or  v[t] = v[t] - spike[t] * v_threshold  (soft reset)

    Args:
        tau: membrane time constant (decay factor), default 2.0
        v_threshold: firing threshold, default 1.0
        v_reset: reset voltage after spike (None for soft reset), default 0.0
        surrogate: surrogate gradient function name, default 'atan'
        detach_reset: whether to detach reset in backward, default False
    """

    def __init__(self, tau=2.0, v_threshold=1.0, v_reset=0.0,
                 surrogate='atan', detach_reset=False):
        super().__init__()
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate = surrogate
        self.detach_reset = detach_reset
        self.v = 0.0

    def reset(self):
        self.v = 0.0

    def neuronal_charge(self, x):
        if isinstance(self.v, float):
            self.v = torch.zeros_like(x)
        self.v = self.v * (1.0 - 1.0 / self.tau) + x / self.tau

    def neuronal_fire(self):
        return heaviside(self.v - self.v_threshold, self.surrogate)

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike_d * self.v_threshold
        else:
            # hard reset
            self.v = (1.0 - spike_d) * self.v + spike_d * self.v_reset

    def forward(self, x):
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike


class IFNeuron(nn.Module):
    """
    Integrate-and-Fire neuron (no leak).

    v[t] = v[t-1] + x[t]
    spike[t] = Heaviside(v[t] - v_threshold)
    """

    def __init__(self, v_threshold=1.0, v_reset=0.0,
                 surrogate='atan', detach_reset=False):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate = surrogate
        self.detach_reset = detach_reset
        self.v = 0.0

    def reset(self):
        self.v = 0.0

    def forward(self, x):
        if isinstance(self.v, float):
            self.v = torch.zeros_like(x)
        self.v = self.v + x
        spike = heaviside(self.v - self.v_threshold, self.surrogate)

        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            self.v = self.v - spike_d * self.v_threshold
        else:
            self.v = (1.0 - spike_d) * self.v + spike_d * self.v_reset
        return spike


class MultiStepLIFNeuron(nn.Module):
    """
    LIF neuron that processes multiple timesteps at once.

    Input: (T, B, C, ...) or (T*B, C, ...) with T specified
    Output: same shape, binary spikes
    """

    def __init__(self, tau=2.0, v_threshold=1.0, v_reset=0.0,
                 surrogate='atan', detach_reset=False, backend='torch'):
        super().__init__()
        self.neuron = LIFNeuron(tau, v_threshold, v_reset, surrogate, detach_reset)

    def reset(self):
        self.neuron.reset()

    def forward(self, x_seq):
        """x_seq: (T, B, ...) tensor"""
        self.neuron.reset()
        spikes = []
        for t in range(x_seq.shape[0]):
            spikes.append(self.neuron(x_seq[t]))
        return torch.stack(spikes, dim=0)


class MultiStepIFNeuron(nn.Module):
    """IF neuron that processes multiple timesteps at once."""

    def __init__(self, v_threshold=1.0, v_reset=0.0,
                 surrogate='atan', detach_reset=False):
        super().__init__()
        self.neuron = IFNeuron(v_threshold, v_reset, surrogate, detach_reset)

    def reset(self):
        self.neuron.reset()

    def forward(self, x_seq):
        """x_seq: (T, B, ...) tensor"""
        self.neuron.reset()
        spikes = []
        for t in range(x_seq.shape[0]):
            spikes.append(self.neuron(x_seq[t]))
        return torch.stack(spikes, dim=0)


# ---------------------------------------------------------------------------
# Utility: reset all spiking neurons in a model
# ---------------------------------------------------------------------------

def reset_net(net):
    """Reset all spiking neurons in the network."""
    for m in net.modules():
        if hasattr(m, 'reset') and callable(m.reset):
            m.reset()
