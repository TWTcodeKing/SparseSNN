/**
 * Hand-optimized IF neuron kernel for Green Context memory partition.
 *
 * Vectorized 128-bit loads, minimal overhead, single-pass grid-stride loop.
 * Designed to achieve peak bandwidth on as few as 8 SMs.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Temporal-safe IF neuron: each thread processes ALL T timesteps for one spatial position.
// This guarantees correct temporal ordering (no race condition on membrane state).
// Input layout: (T*B*H*W, F) where spatial = B*H*W*F, T_steps = total / spatial.
__global__ void if_neuron_vec4_kernel(
    const half* __restrict__ input,    // (M, F) FP16, M = T * spatial_positions
    float* __restrict__ membrane,       // (spatial, F) FP32
    half* __restrict__ spikes,          // (M, F) FP16
    int total_elems,                    // T * spatial * F
    int spatial_elems,                  // B * H * W * F
    float v_threshold
) {
    // Each thread handles one spatial element across ALL timesteps
    int s_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (s_idx >= spatial_elems) return;

    int T_steps = total_elems / spatial_elems;
    float v = membrane[s_idx];

    // Process T timesteps sequentially (correct temporal ordering)
    for (int t = 0; t < T_steps; t++) {
        int g_idx = t * spatial_elems + s_idx;
        float x_val = __half2float(input[g_idx]);
        float h = v + x_val;
        float spike = (h >= v_threshold) ? 1.0f : 0.0f;
        v = (1.0f - spike) * h;
        spikes[g_idx] = __float2half(spike);
    }

    membrane[s_idx] = v;
}

// Temporal-safe LIF neuron: each thread processes ALL T timesteps for one spatial position.
__global__ void lif_neuron_vec4_kernel(
    const half* __restrict__ input,
    float* __restrict__ membrane,
    half* __restrict__ spikes,
    int total_elems,
    int spatial_elems,
    float v_threshold,
    float recip_tau
) {
    int s_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (s_idx >= spatial_elems) return;

    int T_steps = total_elems / spatial_elems;
    float decay = 1.0f - recip_tau;
    float v = membrane[s_idx];

    for (int t = 0; t < T_steps; t++) {
        int g_idx = t * spatial_elems + s_idx;
        float x_val = __half2float(input[g_idx]);
        float h = decay * v + recip_tau * x_val;
        float spike = (h >= v_threshold) ? 1.0f : 0.0f;
        v = (1.0f - spike) * h;
        spikes[g_idx] = __float2half(spike);
    }

    membrane[s_idx] = v;
}



// Python binding
torch::Tensor if_neuron_cuda(
    torch::Tensor input,    // (TB*spatial_hw, F) or (M, F) FP16, contiguous
    torch::Tensor membrane, // (spatial_hw, F) or (spatial, F) FP32, contiguous
    float v_threshold = 1.0f
) {
    auto spikes = torch::empty_like(input);
    int total = input.numel();
    int spatial = membrane.numel();

    // Each thread handles one spatial element across ALL T timesteps
    int threads = 256;
    int blocks = (spatial + threads - 1) / threads;

    if_neuron_vec4_kernel<<<blocks, threads>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        membrane.data_ptr<float>(),
        reinterpret_cast<half*>(spikes.data_ptr<at::Half>()),
        total, spatial, v_threshold);

    return spikes;
}

// LIF Python binding
torch::Tensor lif_neuron_cuda(
    torch::Tensor input,
    torch::Tensor membrane,
    float v_threshold = 1.0f,
    float recip_tau = 0.5f
) {
    auto spikes = torch::empty_like(input);
    int total = input.numel();
    int spatial = membrane.numel();

    int threads = 256;
    int blocks = (spatial + threads - 1) / threads;

    lif_neuron_vec4_kernel<<<blocks, threads>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        membrane.data_ptr<float>(),
        reinterpret_cast<half*>(spikes.data_ptr<at::Half>()),
        total, spatial, v_threshold, recip_tau);

    return spikes;
}


PYBIND11_MODULE(if_neuron_ext, m) {
    m.def("if_neuron", &if_neuron_cuda, "Temporal-safe IF neuron",
          py::arg("input"), py::arg("membrane"), py::arg("v_threshold") = 1.0f);
    m.def("lif_neuron", &lif_neuron_cuda, "Temporal-safe LIF neuron with decay",
          py::arg("input"), py::arg("membrane"),
          py::arg("v_threshold") = 1.0f, py::arg("recip_tau") = 0.5f);
}
