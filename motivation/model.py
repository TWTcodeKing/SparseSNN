"""Dummy 5-layer Linear+LIF SNN for TRT profiling motivation experiment.

Architecture: Input(B, 768) -> [Linear(768,768) -> LIF] x 5 -> TemporalMean -> Linear(768,100)
Temporal: T=4 timesteps, baked into the ONNX graph (LIF loop unrolled).
Scale: hidden_dim=768 mimics Spikformer-class feature dimensions.
"""

import torch
import torch.nn as nn


class MotivationSNN(nn.Module):
    """5-layer feed-forward SNN: Linear + LIF, Spikformer-scale."""

    def __init__(self, hidden_dim=768, num_classes=100, T=4, num_layers=5,
                 tau=2.0, v_threshold=1.0):
        super().__init__()
        self.T = T
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.v_th = v_threshold

        self.linears = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """x: (B, hidden_dim) -> (B, num_classes)"""
        # Repeat input for T timesteps: (T, B, hidden_dim)
        x = x.unsqueeze(0).expand(self.T, -1, -1).contiguous()

        for i in range(self.num_layers):
            # --- Stateless: Linear on (T*B, hidden) ---
            x_flat = x.reshape(-1, self.hidden_dim)
            x_flat = self.linears[i](x_flat)
            x = x_flat.reshape(self.T, -1, self.hidden_dim)

            # --- Stateful: LIF neuron (unrolled over T) ---
            v = torch.zeros_like(x[0])          # (B, hidden)
            s0 = self._lif_step(v, x[0])
            v = self._lif_update_v(v, x[0], s0)
            s1 = self._lif_step(v, x[1])
            v = self._lif_update_v(v, x[1], s1)
            s2 = self._lif_step(v, x[2])
            v = self._lif_update_v(v, x[2], s2)
            s3 = self._lif_step(v, x[3])
            x = torch.stack([s0, s1, s2, s3], dim=0)  # (T, B, hidden)

        # Temporal mean + classifier
        x = x.mean(dim=0)     # (B, hidden)
        return self.head(x)

    def _lif_step(self, v, x_t):
        """Single LIF timestep: returns spike."""
        v_new = v * (1.0 - 1.0 / self.tau) + x_t / self.tau
        return (v_new >= self.v_th).to(x_t.dtype)

    def _lif_update_v(self, v, x_t, spike):
        """Update membrane potential with reset."""
        v_new = v * (1.0 - 1.0 / self.tau) + x_t / self.tau
        return v_new * (1.0 - spike)


def export_onnx(output_path="motivation/output/model.onnx", hidden_dim=768,
                batch_size=1, T=4):
    """Export MotivationSNN to ONNX with dynamic batch axis."""
    model = MotivationSNN(hidden_dim=hidden_dim, T=T)
    model.eval()

    dummy = torch.randn(batch_size, hidden_dim)
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported ONNX: {output_path}")

    # Simplify
    try:
        import onnx
        from onnxsim import simplify
        m = onnx.load(output_path)
        m_sim, ok = simplify(m)
        if ok:
            onnx.save(m_sim, output_path)
            print(f"  Simplified OK ({len(m.graph.node)} -> {len(m_sim.graph.node)} nodes)")
        else:
            print("  Simplification failed, keeping original")
    except ImportError:
        print("  onnxsim not available, skipping simplification")

    return output_path


if __name__ == "__main__":
    export_onnx()
