"""TDL Transform Engine — platform-agnostic SNN graph transforms.

Applies three Temporal Dimension Lowering transformations to convert an SNN
model from 5D temporal IR (T, B, C, H, W) to 4D spatial IR (T*B, C, H, W):

  TDL-1: T-Axis Absorption     — patch stateless wrappers to skip 5D reshape
  TDL-2: Stateful Op Extraction — replace neurons with fused custom ops
  TDL-3: Attention Decomposition — replace DSSA with native 4D DSSA4D

After transforms, the model operates entirely in 4D. The temporal dimension
survives only inside fused neuron kernels (T as an attribute, not a tensor dim).

Platform-agnostic — pure PyTorch, no inference engine dependencies.
"""

import torch
import torch.nn as nn

from iengine.tdl.analysis import (
    collect_neuron_params, is_neuron, is_stateless_wrapper, is_spike_attention,
)
from iengine.tdl.neuron_ops import (
    FusedLIFOp, FusedIFOp, FusedMSOp,
    FusedLIFPluginOp, FusedIFPluginOp, FusedMSPluginOp,
)
from iengine.tdl.dssa_4d import DSSA4D


class TDLTransform:
    """Platform-agnostic TDL graph transforms for SNN inference.

    Usage:
        tdl = TDLTransform(model, T=4)
        tdl.apply()           # model now operates in 4D
        torch.onnx.export(model, ...)
        tdl.restore()         # model restored to original 5D
    """

    def __init__(self, model: nn.Module, T: int = None):
        self.model = model
        self.T = T or getattr(model, 'T', 4)
        self._originals = {}     # name → original forward or module
        self._original_fwd = None  # model.forward backup
        self._dssa_replacements = {}  # name → (parent, attr, original_module)

    def apply(self):
        """Apply TDL-1/2/3 transforms + model forward patch."""
        # Classify attention neurons BEFORE TDL-3 replaces modules
        self._attention_neurons = set()
        for name, module in self.model.named_modules():
            if is_spike_attention(module):
                for child_name, _ in module.named_modules():
                    if child_name:
                        self._attention_neurons.add(f"{name}.{child_name}")

        self._apply_tdl1()
        self._apply_tdl2()
        self._apply_tdl3()
        self._patch_model_forward()

    def restore(self):
        """Undo all transforms, restoring the original model."""
        # Restore model forward
        if self._original_fwd is not None:
            self.model.forward = self._original_fwd
            self._original_fwd = None

        # Restore DSSA modules (TDL-3)
        for name, (parent, attr, orig_module) in self._dssa_replacements.items():
            setattr(parent, attr, orig_module)
        self._dssa_replacements.clear()

        # Restore individual module forwards (TDL-1 + TDL-2)
        for name, module in self.model.named_modules():
            if name in self._originals:
                module.forward = self._originals[name]
        self._originals.clear()

    # ------------------------------------------------------------------
    # TDL-1: T-Axis Absorption — patch stateless wrappers for 4D
    # ------------------------------------------------------------------

    def _apply_tdl1(self):
        """Patch all stateless temporal wrappers to operate on 4D directly."""
        # Collect attention module children (don't patch their internals —
        # TDL-3 handles those via DSSA4D replacement)
        attention_children = set()
        for name, module in self.model.named_modules():
            if is_spike_attention(module):
                for child_name, _ in module.named_modules():
                    if child_name:
                        attention_children.add(f"{name}.{child_name}")

        for name, module in self.model.named_modules():
            if name in attention_children:
                continue
            if not is_stateless_wrapper(module):
                continue

            self._originals[name] = module.forward

            # Pattern A: SeqToANNContainer-like (has .module)
            if hasattr(module, 'module'):
                inner = module.module
                # Check for TDBNContainer pattern: Sequential(Conv2d, BN3d)
                if isinstance(inner, nn.Sequential):
                    children = list(inner.children())
                    if (len(children) == 2
                            and isinstance(children[0], nn.Conv2d)
                            and isinstance(children[1], nn.BatchNorm3d)):
                        # Convert BN3d → BN2d for 4D compatibility
                        bn2d = _convert_bn3d_to_bn2d(children[1])
                        conv = children[0]
                        def _make_tdbn_fwd(c, b):
                            def _fwd(x):
                                return b(c(x))
                            return _fwd
                        module.forward = _make_tdbn_fwd(conv, bn2d)
                        continue

                # Standard SeqToANNContainer: just pass through .module
                def _make_passthrough(m):
                    def _fwd(x):
                        return m(x)
                    return _fwd
                module.forward = _make_passthrough(inner)
                continue

            # Pattern B: _MultiStep* (subclass of Conv2d/MaxPool2d/etc)
            for base_cls in (nn.Conv2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Linear):
                if isinstance(module, base_cls) and type(module) is not base_cls:
                    def _make_base_fwd(m, cls):
                        def _fwd(x):
                            return cls.forward(m, x)
                        return _fwd
                    module.forward = _make_base_fwd(module, base_cls)
                    break

            # Pattern C: BN wrapper (has .bn)
            if hasattr(module, 'bn') and isinstance(module.bn, nn.BatchNorm2d):
                if name in self._originals and module.forward != self._originals[name]:
                    continue  # already patched by Pattern B
                def _make_bn_fwd(bn):
                    def _fwd(x):
                        return bn(x)
                    return _fwd
                module.forward = _make_bn_fwd(module.bn)

    # ------------------------------------------------------------------
    # TDL-2: Stateful Operator Extraction — replace neurons with fused ops
    # ------------------------------------------------------------------

    def _apply_tdl2(self):
        """Replace neuron forwards with fused custom ops.

        Hybrid strategy: neurons on simple paths (Conv→Neuron→Conv) use
        native ONNX ops (Myelin-fusible, zero reformats). Neurons inside
        attention blocks use plugin ops (avoid ONNX shape inference issues
        with dynamic batch + attention reshapes).
        """
        neuron_params = collect_neuron_params(self.model)

        for name, module in self.model.named_modules():
            if name not in neuron_params:
                continue

            p = neuron_params[name]
            self._originals[name] = module.forward

            # Choose export mode: native ONNX for simple path, plugin for attention
            in_attention = (name in self._attention_neurons
                            or any(name.startswith(an + '.')
                                   for an in self._attention_neurons))
            from iengine.tdl import neuron_ops as _nops
            use_native = _nops.use_native_onnx and not in_attention

            if p['type'] == 'LIF':
                OpClass = FusedLIFOp if use_native else FusedLIFPluginOp
                def _make_lif(params, cls):
                    def _fwd(x):
                        return cls.apply(x, params['T'], params['tau'],
                                         params['v_threshold'], params['v_reset'],
                                         params['hard_reset'])
                    return _fwd
                module.forward = _make_lif(p, OpClass)

            elif p['type'] == 'IF':
                OpClass = FusedIFOp if use_native else FusedIFPluginOp
                def _make_if(params, cls):
                    def _fwd(x):
                        return cls.apply(x, params['T'],
                                         params['v_threshold'], params['v_reset'],
                                         params['hard_reset'])
                    return _fwd
                module.forward = _make_if(p, OpClass)

            elif p['type'] == 'MS':
                OpClass = FusedMSOp if use_native else FusedMSPluginOp
                def _make_ms(params, cls):
                    def _fwd(x):
                        return cls.apply(x, params['T'],
                                         params['decay'], params['thresh'])
                    return _fwd
                module.forward = _make_ms(p, OpClass)

    # ------------------------------------------------------------------
    # TDL-3: Temporal Attention Decomposition — replace DSSA with DSSA4D
    # ------------------------------------------------------------------

    def _apply_tdl3(self):
        """Replace DSSA attention modules with native 4D DSSA4D."""
        replacements = []
        for name, module in self.model.named_modules():
            if is_spike_attention(module):
                replacements.append((name, module))

        for name, module in replacements:
            dssa4d = DSSA4D.from_dssa(module, self.T)
            parent, attr = _get_parent_and_attr(self.model, name)
            self._dssa_replacements[name] = (parent, attr, module)
            setattr(parent, attr, dssa4d)

    # ------------------------------------------------------------------
    # Model forward patch — 4D input/output
    # ------------------------------------------------------------------

    def _patch_model_forward(self):
        """Patch the model's forward to accept (B,C,H,W) and operate in 4D."""
        self._original_fwd = self.model.forward
        T = self.T
        model = self.model

        # Detect model architecture by structure
        if hasattr(model, 'prologue') and hasattr(model, 'layers'):
            # SpikingResformer pattern
            self._patch_spikingresformer_forward()
        elif (hasattr(model, 'conv1') and hasattr(model, 'bn1')
              and hasattr(model, 'sn1') and hasattr(model, 'layer1')):
            # SEW-ResNet pattern
            self._patch_sewresnet_forward()
        elif (hasattr(model, 'conv1') and hasattr(model, 'conv2_x')
              and hasattr(model, 'sn_out')):
            # MS-ResNet18 pattern
            self._patch_msresnet18_forward()
        elif (hasattr(model, 'conv1') and hasattr(model, 'layer1')
              and hasattr(model, 'sn_out')):
            # MS-ResNet104/Cifar pattern
            self._patch_msresnet104_forward()
        else:
            # Generic fallback — try to detect the pattern
            self._patch_generic_forward()

    def _patch_spikingresformer_forward(self):
        T = self.T
        model = self.model

        def forward_4d(x):
            x = x.repeat(T, 1, 1, 1)
            x = model.prologue(x)
            x = model.layers(x)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            x = model.classifier(x)
            TB = x.shape[0]
            B = TB // T
            return x.view(T, B, -1).mean(0)

        model.forward = forward_4d

    def _patch_sewresnet_forward(self):
        T = self.T
        model = self.model

        def forward_4d(x):
            x = x.repeat(T, 1, 1, 1)
            x = model.conv1(x)
            x = model.bn1(x)
            # sn1 is now fused (4D), no need for unsqueeze/repeat
            x = model.sn1(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)  # (T*B, 512, 1, 1) → (T*B, 512)
            x = model.fc(x)
            TB = x.shape[0]
            B = TB // T
            return x.view(T, B, -1).mean(0)

        model.forward = forward_4d

    def _patch_msresnet18_forward(self):
        import torch.nn.functional as F
        T = self.T
        model = self.model

        def forward_4d(x):
            x = x.repeat(T, 1, 1, 1)
            x = model.conv1(x)
            x = model.conv2_x(x)
            x = model.conv3_x(x)
            x = model.conv4_x(x)
            x = model.conv5_x(x)
            x = model.sn_out(x)
            # Temporal mean then spatial pool
            TB = x.shape[0]
            B = TB // T
            x = x.view(T, B, *x.shape[1:]).mean(0)
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
            x = model.fc(x)
            return x

        model.forward = forward_4d

    def _patch_msresnet104_forward(self):
        import torch.nn.functional as F
        T = self.T
        model = self.model

        def forward_4d(x):
            x = x.repeat(T, 1, 1, 1)
            x = model.conv1(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.sn_out(x)
            TB = x.shape[0]
            B = TB // T
            x = x.view(T, B, *x.shape[1:]).mean(0)
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
            x = model.fc(x)
            return x

        model.forward = forward_4d

    def _patch_generic_forward(self):
        """Fallback: wrap original forward with 4D input handling."""
        T = self.T
        orig = self._original_fwd

        def forward_4d(x):
            x = x.repeat(T, 1, 1, 1)
            return orig(x)

        self.model.forward = forward_4d



# ---------------------------------------------------------------------------
# High-level export API
# ---------------------------------------------------------------------------

def export_with_fused_neurons(
    model: nn.Module,
    onnx_path: str,
    input_shape: tuple = (1, 3, 32, 32),
    opset: int = 17,
    dynamic_batch: bool = True,
    flatten_temporal: bool = True,
    verbose: bool = True,
):
    """Export SNN model to ONNX with TDL transforms applied.

    Args:
        model:            PyTorch SNN model.
        onnx_path:        Output ONNX file path.
        input_shape:      (B, C, H, W) for tracing.
        opset:            ONNX opset version.
        dynamic_batch:    Enable dynamic batch dim.
        flatten_temporal: Apply TDL-1/2/3 transforms (4D batched mode).
        verbose:          Print progress.
    """
    from pathlib import Path
    from models.neurons import reset_net

    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    T = getattr(model, 'T', 4)

    tdl = TDLTransform(model, T)
    neuron_params = collect_neuron_params(model)

    if verbose:
        n_lif = sum(1 for p in neuron_params.values() if p['type'] == 'LIF')
        n_if = sum(1 for p in neuron_params.values() if p['type'] == 'IF')
        n_ms = sum(1 for p in neuron_params.values() if p['type'] == 'MS')
        parts = [f"{n_lif} LIF"] if n_lif else []
        parts += [f"{n_if} IF"] if n_if else []
        parts += [f"{n_ms} MS"] if n_ms else []
        print(f"  Found {' + '.join(parts)} neurons")

    if flatten_temporal:
        if verbose:
            print(f"  Applying TDL-1/2/3 batched mode (T={T})")
        tdl.apply()
    else:
        if verbose:
            print(f"  Legacy 5D export mode")
        tdl = None

    # Set trace_batch_size for native ONNX symbolic
    import iengine.tdl.neuron_ops as _nops
    if dynamic_batch:
        _nops.trace_batch_size = None  # Force dynamic Slice (Shape/Div)
    else:
        _nops.trace_batch_size = input_shape[0]  # Static B

    device = next(model.parameters()).device
    model.eval()
    reset_net(model)
    dummy = torch.randn(*input_shape, device=device)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }

    if verbose:
        print(f"  Exporting ONNX: input={list(input_shape)}")

    # Bypass peephole crash (known torch issue with fused custom ops)
    _C = torch._C
    orig_peephole = _C._jit_pass_peephole
    def _safe_peephole(g, f=False):
        try:
            return orig_peephole(g, f)
        except IndexError:
            pass
    _C._jit_pass_peephole = _safe_peephole

    try:
        with torch.no_grad():
            torch.onnx.export(
                model, dummy, onnx_path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                opset_version=opset,
                do_constant_folding=False,
            )
    finally:
        _C._jit_pass_peephole = orig_peephole
        _nops.trace_batch_size = None
        reset_net(model)
        if tdl is not None:
            tdl.restore()

    if verbose:
        import os
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  Saved: {onnx_path} ({size_mb:.1f} MB)")

    return onnx_path


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _get_parent_and_attr(model, name):
    """Get (parent_module, attribute_name) for a named submodule."""
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def _convert_bn3d_to_bn2d(bn3d):
    """Convert BatchNorm3d to BatchNorm2d (safe at eval time)."""
    bn2d = nn.BatchNorm2d(
        bn3d.num_features, eps=bn3d.eps, momentum=bn3d.momentum,
        affine=bn3d.affine, track_running_stats=bn3d.track_running_stats,
    )
    if bn3d.affine:
        bn2d.weight.data.copy_(bn3d.weight.data)
        bn2d.bias.data.copy_(bn3d.bias.data)
    if bn3d.track_running_stats:
        bn2d.running_mean.copy_(bn3d.running_mean)
        bn2d.running_var.copy_(bn3d.running_var)
        bn2d.num_batches_tracked.copy_(bn3d.num_batches_tracked)
    return bn2d
