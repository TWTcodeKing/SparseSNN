"""Correctness tests for dense Linear+BN+IF/LIF TileLang kernels."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available",
)


def ref_linear_bn_if(inp, weight, state, bn_scale, bn_bias,
                      v_threshold=1.0, v_reset=0.0):
    out = inp.float() @ weight.float()
    bn_out = out * bn_scale.unsqueeze(0) + bn_bias.unsqueeze(0)
    h = state + bn_out
    spike = (h >= v_threshold).float()
    v_new = (1.0 - spike) * h + spike * v_reset
    return spike, v_new


def ref_linear_bn_lif(inp, weight, state, bn_scale, bn_bias,
                       v_threshold=1.0, v_reset=0.0, recip_tau=0.5):
    out = inp.float() @ weight.float()
    bn_out = out * bn_scale.unsqueeze(0) + bn_bias.unsqueeze(0)
    h = (1.0 - recip_tau) * state + recip_tau * bn_out
    spike = (h >= v_threshold).float()
    v_new = (1.0 - spike) * h + spike * v_reset
    return spike, v_new


# (M, K, N, description)
LINEAR_CONFIGS = [
    (64, 128, 128, "square"),
    (128, 256, 64, "wide_to_narrow"),
    (32, 384, 384, "transformer_embed"),
]


class TestLinearBnIF:

    @pytest.mark.parametrize(
        "M,K,N,desc", LINEAR_CONFIGS,
        ids=[c[-1] for c in LINEAR_CONFIGS],
    )
    def test_single_timestep(self, M, K, N, desc):
        from iengine.backends.tilelang.linear_bn_neuron import linear_bn_if_kernel

        torch.manual_seed(42)
        inp = torch.randn(M, K, device="cuda", dtype=torch.float16)
        weight = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.1
        state = torch.zeros(M, N, device="cuda", dtype=torch.float32)
        bn_scale = torch.ones(N, device="cuda", dtype=torch.float32) * 0.5
        bn_bias = torch.randn(N, device="cuda", dtype=torch.float32) * 0.1

        kernel = linear_bn_if_kernel(
            M=M, K=K, N_out=N,
            block_M=64, block_N=64, block_K=32,
            num_stages=2, threads=128,
        )
        spikes_tl = kernel(inp, weight, state, bn_scale, bn_bias)

        spikes_ref, state_ref = ref_linear_bn_if(inp, weight,
            torch.zeros(M, N, device="cuda", dtype=torch.float32),
            bn_scale, bn_bias)

        torch.testing.assert_close(
            spikes_tl.float(), spikes_ref, rtol=1e-2, atol=1e-2,
            msg=f"Linear IF spike mismatch for {desc}",
        )
        torch.testing.assert_close(
            state.float(), state_ref, rtol=1e-2, atol=5e-2,
            msg=f"Linear IF state mismatch for {desc}",
        )


class TestLinearBnLIF:

    def test_single_timestep(self):
        from iengine.backends.tilelang.linear_bn_neuron import linear_bn_lif_kernel

        M, K, N = 64, 128, 128
        recip_tau = 0.5
        torch.manual_seed(42)
        inp = torch.randn(M, K, device="cuda", dtype=torch.float16)
        weight = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.1
        state = torch.zeros(M, N, device="cuda", dtype=torch.float32)
        bn_scale = torch.ones(N, device="cuda", dtype=torch.float32) * 0.5
        bn_bias = torch.randn(N, device="cuda", dtype=torch.float32) * 0.1

        kernel = linear_bn_lif_kernel(
            M=M, K=K, N_out=N,
            block_M=64, block_N=64, block_K=32,
            num_stages=2, threads=128,
            recip_tau=recip_tau,
        )
        spikes_tl = kernel(inp, weight, state, bn_scale, bn_bias)

        spikes_ref, state_ref = ref_linear_bn_lif(inp, weight,
            torch.zeros(M, N, device="cuda", dtype=torch.float32),
            bn_scale, bn_bias, recip_tau=recip_tau)

        torch.testing.assert_close(
            spikes_tl.float(), spikes_ref, rtol=1e-2, atol=1e-2,
        )
        torch.testing.assert_close(
            state.float(), state_ref, rtol=1e-2, atol=5e-2,
        )
