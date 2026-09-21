"""TDL Transform Engine — platform-agnostic SNN graph transforms.

Applies three Temporal Dimension Lowering transformations to convert an SNN
model from 5D temporal IR (T, B, C, H, W) to 4D spatial IR (T*B, C, H, W):

  TDL-1: T-Axis Absorption     — patch stateless wrappers to skip 5D reshape
  TDL-2: Stateful Op Extraction — replace neurons with fused custom ops
  TDL-3: Attention Decomposition — replace attention blocks with 4D variants

Supported attention types:
  - DSSA (SpikingResformer) → DSSA4D
  - SSA  (SpikFormer)       → SpikformerSSA4D + SpikformerMLP4D
  - SSA  (MaxFormer)        → MaxFormerSSA4D
  - Token_QK_Attention      → TokenQKA4D

After transforms, the model operates entirely in 4D. The temporal dimension
survives only inside fused neuron kernels (T as an attribute, not a tensor dim).

Platform-agnostic — pure PyTorch, no inference engine dependencies.
"""

import torch
import torch.nn as nn

from sengine_edge.tdl.analysis import (
    collect_neuron_params, is_neuron, is_stateless_wrapper,
    is_spike_attention, is_spikformer_mlp, get_attention_type,
)
from sengine_edge.tdl.neuron_ops import (
    FusedLIFOp, FusedIFOp, FusedMSOp,
    FusedLIFPluginOp, FusedIFPluginOp, FusedMSPluginOp,
    FusedILIFOp, FusedILIFPluginOp,
)
from sengine_edge.tdl.dssa_4d import DSSA4D
from sengine_edge.tdl.ssa_4d import (
    SpikformerSSA4D, SpikformerMLP4D, MaxFormerSSA4D, TokenQKA4D,
)


class TDLTransform:
    """Platform-agnostic TDL graph transforms for SNN inference.

    Usage:
        tdl = TDLTransform(model, T=4)
        tdl.apply()           # model now operates in 4D
        torch.onnx.export(model, ...)
        tdl.restore()         # model restored to original 5D
    """

    def __init__(self, model: nn.Module, T: int = None,
                 force_native_onnx: bool = False):
        self.model = model
        self.T = T or getattr(model, 'T', 4)
        self.force_native_onnx = force_native_onnx
        self._originals = {}     # name → original forward or module
        self._original_fwd = None  # model.forward backup
        self._dssa_replacements = {}  # name → (parent, attr, original_module)

    def apply(self):
        """Apply TDL-1/2/3 transforms + model forward patch."""
        # Classify attention + SpikFormer MLP neurons BEFORE TDL-3 replaces modules
        self._attention_neurons = set()
        for name, module in self.model.named_modules():
            if is_spike_attention(module) or is_spikformer_mlp(module):
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
        if hasattr(self.model, '_tdl_orig_forward'):
            # Undo the top-level patch made by _wrap_bare_convs_for_tdl
            self.model.forward = self.model._tdl_orig_forward
            del self.model._tdl_orig_forward
            self.model._tdl_forward_4d = False

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
        # Collect attention + SpikFormer MLP children (don't patch their
        # internals — TDL-3 handles those via 4D module replacement)
        attention_children = set()
        for name, module in self.model.named_modules():
            if is_spike_attention(module) or is_spikformer_mlp(module):
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

            # Choose export mode: native ONNX for simple path, plugin for attention.
            # force_native_onnx=True overrides this for TRT export (no plugins).
            in_attention = (name in self._attention_neurons
                            or any(name.startswith(an + '.')
                                   for an in self._attention_neurons))
            from sengine_edge.tdl import neuron_ops as _nops
            use_native = _nops.use_native_onnx and (self.force_native_onnx
                                                     or not in_attention)

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

            elif p['type'] == 'ILIF':
                OpClass = FusedILIFOp if use_native else FusedILIFPluginOp
                def _make_ilif(params, cls):
                    def _fwd(x):
                        return cls.apply(x, params['T'],
                                         params['decay'], params['max_level'])
                    return _fwd
                module.forward = _make_ilif(p, OpClass)

    # ------------------------------------------------------------------
    # TDL-3: Temporal Attention Decomposition — replace with 4D variants
    # ------------------------------------------------------------------

    def _apply_tdl3(self):
        """Replace attention (+ SpikFormer MLP) modules with 4D variants."""
        # Collect attention modules
        attn_replacements = []
        for name, module in self.model.named_modules():
            atype = get_attention_type(module)
            if atype is not None:
                attn_replacements.append((name, module, atype))

        for name, module, atype in attn_replacements:
            if atype == 'dssa':
                replacement = DSSA4D.from_dssa(module, self.T)
            elif atype == 'spikformer_ssa':
                replacement = SpikformerSSA4D.from_ssa(module, self.T)
            elif atype == 'maxformer_ssa':
                replacement = MaxFormerSSA4D.from_ssa(module, self.T)
            elif atype == 'token_qka':
                replacement = TokenQKA4D.from_qka(module, self.T)
            else:
                continue
            parent, attr = _get_parent_and_attr(self.model, name)
            self._dssa_replacements[name] = (parent, attr, module)
            setattr(parent, attr, replacement)

        # Also replace SpikFormer MLP modules (Linear + BN1d + LIF in 5D)
        mlp_replacements = []
        for name, module in self.model.named_modules():
            if is_spikformer_mlp(module):
                mlp_replacements.append((name, module))

        for name, module in mlp_replacements:
            replacement = SpikformerMLP4D.from_mlp(module, self.T)
            parent, attr = _get_parent_and_attr(self.model, name)
            self._dssa_replacements[name] = (parent, attr, module)
            setattr(parent, attr, replacement)

    # ------------------------------------------------------------------
    # Model forward patch — 4D input/output
    # ------------------------------------------------------------------

    def _patch_model_forward(self):
        """Patch the model's forward to accept (B,C,H,W) and operate in 4D."""
        self._original_fwd = self.model.forward
        T = self.T
        model = self.model

        # Detect model architecture by structure
        if hasattr(model, 'patch_embed') and hasattr(model, 'block'):
            # SpikFormer pattern: SPS patch_embed + ModuleList block
            self._patch_spikformer_forward()
        elif hasattr(model, 'patch_embed1') and hasattr(model, 'stage3'):
            # MaxFormer / MS_QKFormer pattern: hierarchical stages
            self._patch_maxformer_forward()
        elif hasattr(model, 'prologue') and hasattr(model, 'layers'):
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
        elif (hasattr(model, 'backbone') and hasattr(model, 'neck')
              and hasattr(model, 'detect')):
            # Detection model pattern (ems_yolo)
            self._patch_detection_forward()
        elif (hasattr(model, 'stem') and hasattr(model, 'neck')
              and hasattr(model, 'detect')):
            # Detection model pattern (spike_yolo — no separate backbone)
            self._patch_detection_forward()
        else:
            # Generic fallback — try to detect the pattern
            self._patch_generic_forward()

    def _patch_spikformer_forward(self):
        """4D forward for SpikFormer.

        Original flow: (B,C,H,W) → repeat → (T,B,C,H,W) → SPS → (T,B,N,C)
                        → Blocks(SSA+MLP in 5D) → mean(N) → mean(T) → head

        4D flow: (B,C,H,W) → repeat → (T*B,C,H,W) → SPS(4D) → (T*B,N,C)
                  → Blocks(SSA4D+MLP4D in 3D) → mean(N) → reshape → mean(T) → head

        SPS is already 4D after TDL-1 (all Conv2d+BN2d+MaxPool are stateless
        wrappers), but we need to rewrite its forward to skip the 5D reshapes
        and flatten(-2).transpose(-1,-2) at the end.
        """
        T = self.T
        model = self.model
        sps = model.patch_embed

        # Patch SPS forward for 4D
        def sps_4d(x):
            # x: (T*B, C, H, W)
            x = sps.proj_conv(x)
            x = sps.proj_bn(x)
            x = sps.proj_lif(x)

            x = sps.proj_conv1(x)
            x = sps.proj_bn1(x)
            x = sps.proj_lif1(x)

            x = sps.proj_conv2(x)
            x = sps.proj_bn2(x)
            x = sps.proj_lif2(x)
            x = sps.maxpool2(x)

            x = sps.proj_conv3(x)
            x = sps.proj_bn3(x)
            x = sps.proj_lif3(x)
            x = sps.maxpool3(x)

            x_feat = x
            x = sps.rpe_conv(x)
            x = sps.rpe_bn(x)
            x = sps.rpe_lif(x)
            x = x + x_feat

            # (T*B, C, H', W') → (T*B, N, C) where N = H'*W'
            # Use explicit reshape with Python ints for constant ONNX target.
            C_dim = int(x.shape[1])
            N_dim = int(x.shape[2]) * int(x.shape[3])
            x = x.reshape(-1, C_dim, N_dim).transpose(1, 2)
            return x

        self._originals['patch_embed'] = sps.forward
        sps.forward = sps_4d

        # Patch Block forward for 4D (SSA4D and MLP4D already replaced by TDL-3)
        for i, blk in enumerate(model.block):
            blk_name = f'block.{i}'
            self._originals[blk_name] = blk.forward
            def _make_blk_fwd(b):
                def _fwd(x):
                    # x: (T*B, N, C) — SSA4D and MLP4D operate directly on this
                    x = x + b.attn(x)
                    x = x + b.mlp(x)
                    return x
                return _fwd
            blk.forward = _make_blk_fwd(blk)

        def forward_4d(x):
            # x: (B, C, H, W)
            x = x.repeat(T, 1, 1, 1)       # (T*B, C, H, W)
            x = model.patch_embed(x)         # (T*B, N, C)
            for blk in model.block:
                x = blk(x)                   # (T*B, N, C)
            # Token pool + temporal mean + classifier.
            # Reshape (TB, N, C) → (TB, C, sqrt(N), sqrt(N)) as 4D
            # so the parser sees standard GlobalAvgPool → Flatten → Gemm → TemporalMean.
            C_dim = int(x.shape[2])
            N_dim = int(x.shape[1])
            # N should be a perfect square (H'*W' from patch embed)
            import math
            side = int(math.isqrt(N_dim))
            x = x.transpose(1, 2).reshape(-1, C_dim, side, side)   # (TB, C, H', W')
            x = torch.nn.functional.adaptive_avg_pool2d(x, 1)       # (TB, C, 1, 1)
            x = x.flatten(1)                                         # (TB, C)
            TB = x.shape[0]
            B = TB // T
            x = x.view(T, B, -1).mean(0)           # (B, C) — mean over T
            x = model.head(x)                       # (B, num_classes)
            return x

        model.forward = forward_4d

    def _patch_maxformer_forward(self):
        """4D forward for MaxFormer / MaxFormerCifar / MaxFormerDVS / MS_QKFormer.

        These share the same hierarchical structure:
          patch_embed1 → stage1 → patch_embed2 → stage2 → patch_embed3 → stage3
          → head_lif → head → mean(T)

        All stages operate on (T, B, C, H, W). In 4D mode, they operate on
        (T*B, C, H, W). The S_MLP and embedding modules use Conv2d+BN2d which
        are already patched by TDL-1. The SSA/QKA blocks are replaced by TDL-3.

        S_MLP needs its own 4D patch since it has 5D neuron + residual patterns.
        """
        T = self.T
        model = self.model

        # Patch S_MLP modules for 4D
        for name, module in model.named_modules():
            if not (hasattr(module, 'fc1_conv') and hasattr(module, 'fc1_bn')
                    and hasattr(module, 'fc1_lif') and hasattr(module, 'fc2_conv')):
                continue
            self._originals[name] = module.forward
            def _make_smlp_fwd(m):
                def _fwd(x):
                    # x: (T*B, C, H, W)
                    identity = x
                    x = m.fc1_lif(x)
                    x = m.fc1_conv(x)
                    x = m.fc1_bn(x)
                    if m.res:
                        x = identity + x
                        identity = x
                    x = m.fc2_lif(x)
                    x = m.fc2_conv(x)
                    x = m.fc2_bn(x)
                    x = x + identity
                    return x
                return _fwd
            module.forward = _make_smlp_fwd(module)

        # Patch Block_DWC for 4D
        for name, module in model.named_modules():
            if not (hasattr(module, 'conv') and hasattr(module, 'conv_bn')
                    and hasattr(module, 'conv_neuron') and hasattr(module, 'mlp')):
                continue
            self._originals[name] = module.forward
            def _make_dwc_fwd(m):
                def _fwd(x):
                    # x: (T*B, C, H, W)
                    identity = x
                    x = m.conv_neuron(x)
                    x = m.conv(x)
                    x = m.conv_bn(x)
                    x = x + identity
                    x = m.mlp(x)
                    return x
                return _fwd
            module.forward = _make_dwc_fwd(module)

        # Patch Block_SSA for 4D (SSA already replaced by MaxFormerSSA4D)
        for name, module in model.named_modules():
            if (hasattr(module, 'attn') and hasattr(module, 'mlp')
                    and not hasattr(module, 'conv')
                    and isinstance(module.attn, MaxFormerSSA4D)):
                self._originals[name] = module.forward
                def _make_bssa_fwd(m):
                    def _fwd(x):
                        x = m.attn(x)   # MaxFormerSSA4D includes residual
                        x = m.mlp(x)    # S_MLP 4D includes residual
                        return x
                    return _fwd
                module.forward = _make_bssa_fwd(module)

        # Patch Block_QKA for 4D (QKA already replaced by TokenQKA4D)
        for name, module in model.named_modules():
            if (hasattr(module, 'attn') and hasattr(module, 'mlp')
                    and isinstance(module.attn, TokenQKA4D)):
                self._originals[name] = module.forward
                def _make_bqka_fwd(m):
                    def _fwd(x):
                        x = m.attn(x)   # TokenQKA4D includes residual
                        x = m.mlp(x)
                        return x
                    return _fwd
                module.forward = _make_bqka_fwd(module)

        # Patch Block_Max for 4D (MaxPool mixer)
        for name, module in model.named_modules():
            if hasattr(module, 'pool') and hasattr(module, 'mlp') and not hasattr(module, 'attn'):
                self._originals[name] = module.forward
                def _make_bmax_fwd(m):
                    def _fwd(x):
                        x = m.pool(x)
                        x = m.mlp(x)
                        return x
                    return _fwd
                module.forward = _make_bmax_fwd(module)

        # Patch Block_identity for 4D (MLP only)
        for name, module in model.named_modules():
            if (hasattr(module, 'mlp') and not hasattr(module, 'attn')
                    and not hasattr(module, 'pool') and not hasattr(module, 'conv')):
                cls_name = type(module).__name__
                if cls_name == 'Block_identity':
                    self._originals[name] = module.forward
                    def _make_bid_fwd(m):
                        def _fwd(x):
                            return m.mlp(x)
                        return _fwd
                    module.forward = _make_bid_fwd(module)

        # Patch embedding modules for 4D
        self._patch_maxformer_embeds()

        def forward_4d(x):
            # x: (B, C, H, W)
            x = x.repeat(T, 1, 1, 1)             # (T*B, C, H, W)
            x = model.patch_embed1(x)
            for blk in model.stage1:
                x = blk(x)
            x = model.patch_embed2(x)
            for blk in model.stage2:
                x = blk(x)
            x = model.patch_embed3(x)
            for blk in model.stage3:
                x = blk(x)
            # Global average pool: (T*B, C, H, W) → (T*B, C, 1, 1) → (T*B, C)
            x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
            x = model.head_lif(x)
            x = model.head(x)
            TB = x.shape[0]
            B = TB // T
            return x.view(T, B, -1).mean(0)

        model.forward = forward_4d

    def _patch_maxformer_embeds(self):
        """Patch MaxFormer embedding modules for 4D operation."""
        model = self.model

        for name, module in model.named_modules():
            cls_name = type(module).__name__

            if cls_name == 'Embed':
                self._originals[name] = module.forward
                def _make_embed_fwd(m):
                    def _fwd(x, dual=False):
                        if not m.shortcut:
                            x = m.embed_lif(x)
                        x_feat = x
                        x = m.embed_conv(x)
                        x = m.embed_bn(x)
                        if dual:
                            return x, x_feat
                        return x
                    return _fwd
                module.forward = _make_embed_fwd(module)

            elif cls_name == 'MaxEmbed':
                self._originals[name] = module.forward
                def _make_maxembed_fwd(m):
                    def _fwd(x, dual=False):
                        if not m.shortcut:
                            x = m.embed_lif(x)
                        x_feat = x
                        x = m.embed_conv(x)
                        x = m.embed_bn(x)
                        x = m.maxpool(x)
                        if dual:
                            return x, x_feat
                        return x
                    return _fwd
                module.forward = _make_maxembed_fwd(module)

            elif cls_name == 'EmbedOrigImageNet':
                self._originals[name] = module.forward
                def _make_eoi_fwd(m):
                    def _fwd(x):
                        x = m.embed1(x)
                        x, x_feat = m.embed2(x, dual=True)
                        x = m.embed3(x)
                        x_feat = m.embed4(x_feat)
                        return x + x_feat
                    return _fwd
                module.forward = _make_eoi_fwd(module)

            elif cls_name == 'EmbedOrig':
                self._originals[name] = module.forward
                def _make_eo_fwd(m):
                    def _fwd(x):
                        x = m.embed1(x)
                        x, x_feat = m.embed2(x, dual=True)
                        x_feat = m.embed3(x_feat)
                        return x + x_feat
                    return _fwd
                module.forward = _make_eo_fwd(module)

            elif cls_name == 'EmbedMax':
                self._originals[name] = module.forward
                def _make_em_fwd(m):
                    def _fwd(x):
                        x, x_feat = m.max_embed1(x, dual=True)
                        x = m.embed1(x)
                        x_feat = m.max_embed2(x_feat)
                        return x + x_feat
                    return _fwd
                module.forward = _make_em_fwd(module)

            elif cls_name == 'EmbedMaxPlus':
                self._originals[name] = module.forward
                def _make_emp_fwd(m):
                    def _fwd(x):
                        x = m.proj_conv(x)
                        x = m.proj_bn(x)
                        x = m.max_embed1(x)
                        x, x_feat = m.max_embed2(x, dual=True)
                        x = m.max_embed3(x)
                        x_feat = m.embed1(x_feat)
                        return x + x_feat
                    return _fwd
                module.forward = _make_emp_fwd(module)

            elif cls_name == 'PatchEmbedInitMaxPool':
                self._originals[name] = module.forward
                def _make_peimp_fwd(m):
                    def _fwd(x):
                        x = m.embed1.embed_conv(x)
                        x = m.embed1.embed_bn(x)
                        x = m.maxpool1(x)
                        x = m.lif1(x)
                        x_feat = x
                        x = m.embed2.embed_conv(x)
                        x = m.embed2.embed_bn(x)
                        x = m.maxpool2(x)
                        x = m.lif2(x)
                        x = m.embed3.embed_conv(x)
                        x = m.embed3.embed_bn(x)
                        x_feat = m.embed4.embed_conv(x_feat)
                        x_feat = m.embed4.embed_bn(x_feat)
                        return x + x_feat
                    return _fwd
                module.forward = _make_peimp_fwd(module)

            elif cls_name == 'Embed1Max':
                self._originals[name] = module.forward
                def _make_e1m_fwd(m):
                    def _fwd(x):
                        x, x_feat = m.max_embed1(x, dual=True)
                        x = m.embed1(x)
                        x_feat = m.max_embed2(x_feat)
                        return x + x_feat
                    return _fwd
                module.forward = _make_e1m_fwd(module)

            elif cls_name == 'Embed1MaxCifar':
                self._originals[name] = module.forward
                def _make_e1mc_fwd(m):
                    def _fwd(x):
                        x, x_feat = m.embed1(x, dual=True)
                        x = m.max_embed1(x)
                        x_feat = m.embed2(x_feat)
                        return x + x_feat
                    return _fwd
                module.forward = _make_e1mc_fwd(module)

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

    def _patch_detection_forward(self):
        """4D forward for detection models (ems_yolo, spike_yolo).

        Original: (B,C,H,W) → repeat → (T,B,C,H,W) → backbone → neck → mean(T) per feature → detect
        4D:       (B,C,H,W) → tile → (T*B,C,H,W) → backbone(4D) → neck(4D) → reshape+mean per feature → detect
        """
        T = self.T
        model = self.model

        # Replace TDBNContainer's BN3d with BN2d for 4D compatibility.
        self._convert_tdbn_to_4d(model)

        # Patch the neck's forward to work on 4D tensors
        self._patch_neck_for_4d(model.neck)

        has_backbone = hasattr(model, 'backbone')

        # Patch C2fSpike blocks for 4D (they have T,B,C,H,W = x.shape unpack)
        self._patch_c2fspike_for_4d(model)

        def forward_4d(x):
            # x: (B, C, H, W) → tile → (T*B, C, H, W)
            x = x.repeat(T, 1, 1, 1)

            if has_backbone:
                # ems_yolo: backbone returns [p3, p4, p5]
                features = model.backbone(x)
            else:
                # spike_yolo: stem + stages inline
                x = model.stem_neuron(model.stem(x))
                x = model.stage1(x)
                x = model.stage2(x)
                p3 = x
                x = model.stage3(x)
                p4 = x
                x = model.stage4(x)
                p5 = x
                features = [p3, p4, p5]

            # Neck (4D)
            features = model.neck(features)

            # Per-feature temporal mean: (T*B, C, H, W) → (B, C, H, W)
            out_features = []
            for f in features:
                TB = f.shape[0]
                B = TB // T
                out_features.append(f.reshape(T, B, *f.shape[1:]).mean(0))
            return model.detect(out_features)

        model.forward = forward_4d

    def _patch_c2fspike_for_4d(self, module):
        """Patch C2fSpike blocks: replace 5D T,B,C,H,W unpack + dim=2 cat with 4D."""
        try:
            from models.spike_yolo import C2fSpike
        except ImportError:
            return
        for name, child in module.named_modules():
            if isinstance(child, C2fSpike):
                self._originals[name] = child.forward
                def _make_c2f_4d(blk):
                    def _fwd(x):
                        x = blk.cv1(x)
                        C = x.shape[1]  # 4D: (TB, C, H, W)
                        x0 = x[:, :C // 2]
                        x1 = x[:, C // 2:]
                        parts = [x0, x1]
                        for block in blk.blocks:
                            x1 = block(x1)
                            parts.append(x1)
                        out = torch.cat(parts, dim=1)  # dim=1 for 4D channel
                        return blk.cv2(out)
                    return _fwd
                child.forward = _make_c2f_4d(child)

    def _convert_tdbn_to_4d(self, module):
        """Replace BN3d in TDBNContainer with BN2d, patch forward for 4D."""
        from models.msresnet import TDBNContainer
        for name, child in module.named_children():
            if isinstance(child, TDBNContainer):
                conv = child.module[0]
                bn3d = child.module[1]
                # Create BN2d with same weights as BN3d
                bn2d = nn.BatchNorm2d(bn3d.num_features)
                bn2d.weight = bn3d.weight
                bn2d.bias = bn3d.bias
                bn2d.running_mean = bn3d.running_mean
                bn2d.running_var = bn3d.running_var
                bn2d.num_batches_tracked = bn3d.num_batches_tracked
                bn2d.eps = bn3d.eps
                bn2d.momentum = bn3d.momentum
                child.module[1] = bn2d
                # Patch forward: just Conv2d + BN2d on (T*B, C, H, W)
                def _make_4d_fwd(c, b):
                    def _fwd(x):
                        return b(c(x))
                    return _fwd
                child.forward = _make_4d_fwd(conv, bn2d)
            else:
                self._convert_tdbn_to_4d(child)

    def _patch_neck_for_4d(self, neck):
        """Patch FPN/PANet neck to accept list of 4D tensors.

        Original necks expect (T,B,C,H,W) features, do flatten(0,1)/view(T,B,...).
        In 4D mode, input is (T*B,C,H,W) — flatten is no-op, cat uses dim=1.
        """
        import torch.nn.functional as _F
        # EMSFPN pattern (ems_yolo)
        if hasattr(neck, 'lateral5') and hasattr(neck, 'smooth4'):
            def _emsfpn_4d(features):
                p3, p4, p5 = features
                p5_up = neck.sn5(neck.lateral5(p5))
                p5_up = _F.interpolate(p5_up, size=p4.shape[2:], mode='nearest')
                p4 = neck.sn4(neck.smooth4(p4 + p5_up))
                p4_up = neck.sn4_lat(neck.lateral4(p4))
                p4_up = _F.interpolate(p4_up, size=p3.shape[2:], mode='nearest')
                p3 = neck.sn3(neck.smooth3(p3 + p4_up))
                return [p3, p4, p5]
            neck.forward = _emsfpn_4d
        # SpikePANet pattern (spike_yolo)
        elif hasattr(neck, 'up5') and hasattr(neck, 'td4'):
            def _spikepan_4d(features):
                p3, p4, p5 = features
                # Top-down
                p5_up = neck.up5(p5)
                p5_up = _F.interpolate(p5_up, size=p4.shape[2:], mode='nearest')
                p4 = neck.td4(torch.cat([p4, p5_up], dim=1))  # dim=1 for NCHW 4D
                p4_up = neck.up4(p4)
                p4_up = _F.interpolate(p4_up, size=p3.shape[2:], mode='nearest')
                p3 = neck.td3(torch.cat([p3, p4_up], dim=1))
                # Bottom-up
                p3_down = neck.down3(p3)
                p4 = neck.bu4(torch.cat([p4, p3_down], dim=1))
                p4_down = neck.down4(p4)
                p5 = neck.bu5(torch.cat([p5, p4_down], dim=1))
                return [p3, p4, p5]
            neck.forward = _spikepan_4d

    def _patch_generic_forward(self):
        """Fallback: wrap original forward with 4D input handling."""
        T = self.T
        orig = self._original_fwd
        if getattr(self.model, '_tdl_forward_4d', False):
            # _wrap_bare_convs_for_tdl already installed a 4D forward that
            # tiles the input itself (VGG style). Wrapping it again would tile
            # T*T times and export a graph with a T*B "batch".
            return

        def forward_4d(x):
            x = x.repeat(T, 1, 1, 1)
            return orig(x)

        self.model.forward = forward_4d



# ---------------------------------------------------------------------------
# Pre-processing: wrap bare Conv+BN for TDL compatibility
# ---------------------------------------------------------------------------

def _wrap_bare_convs_for_tdl(model: nn.Module, T: int, verbose: bool = False):
    """Patch DVS model blocks that do manual 5D→4D reshaping.

    DVS model variants (MaxFormerDVS, etc.) don't use SeqToANNContainer.
    Their blocks manually reshape: x.flatten(0,1) → Conv → reshape(T,B,...).

    For ONNX export with TDL, the model's forward() must accept 4D input
    (T already absorbed into batch). This function patches the model's
    top-level forward to:
      1. NOT do x.unsqueeze(0).repeat(T,...) (the 5D expansion)
      2. Pass 4D (T*B, C, H, W) input directly

    The blocks' manual flatten(0,1) becomes a no-op on already-4D input
    (flatten(0,1) on 4D (TB,C,H,W) is still (TB,C,H,W)).
    The reshape(T,B,...) after Conv still works if T and B are tracked.

    The key issue: reshape(T, B, C, H, W) with hardcoded T fails on 4D.
    We patch blocks to use reshape(-1, B, C, H, W) which infers T.
    """
    import types

    n_patched = 0

    # Patch the model's top-level forward to skip 5D expansion
    orig_forward = model.forward

    def _patched_forward(self, x):
        # Skip the 5D expansion: input is already (B, C, H, W) from ONNX
        # We need to tile it T times: (B,C,H,W) → (T*B,C,H,W)
        if len(x.shape) == 4:
            x = x.repeat(self.T, 1, 1, 1)  # (T*B, C, H, W)
        return self.forward_features(x)

    # Only patch if the model has the 5D-expansion pattern in forward
    import inspect
    try:
        src = inspect.getsource(model.forward)
    except (TypeError, OSError):
        return

    if '.repeat(self.T,' not in src and 'unsqueeze(0)' not in src:
        return

    # This model expands 4D input to 5D in forward. We need to patch
    # forward to accept 4D (T already absorbed by TDL-1).

    # Case 1: Model has forward_features (MaxFormer-DVS style)
    # Case 2: Model passes directly to self.features (VGG style)
    # General approach: patch forward to tile input T times as 4D,
    # and handle temporal mean at the end.

    # Patch blocks that do manual T,B,C,H,W = x.shape reshaping
    for name, module in model.named_modules():
        try:
            msrc = inspect.getsource(module.forward)
        except (TypeError, OSError):
            continue

        if 'T, B, C, H, W = x.shape' not in msrc:
            continue

        orig_fwd = module.forward

        def _make_patched_block_fwd(orig, t_val):
            def _patched(x):
                if len(x.shape) == 4:
                    TB, C, H, W = x.shape
                    B = TB // t_val
                    x = x.reshape(t_val, B, C, H, W)
                result = orig(x)
                if len(result.shape) == 5:
                    T2, B2 = result.shape[0], result.shape[1]
                    result = result.reshape(T2 * B2, *result.shape[2:])
                return result
            return _patched

        module.forward = _make_patched_block_fwd(orig_fwd, T)
        n_patched += 1

    # Patch the top-level forward to accept 4D input
    orig_forward = model.forward

    if hasattr(model, 'forward_features'):
        # MaxFormer-DVS style: forward → forward_features → stages
        orig_ff = model.forward_features

        def _new_forward_features(x):
            x = model.patch_embed1(x)
            for blk in model.stage1:
                x = blk(x)
            x = model.patch_embed2(x)
            for blk in model.stage2:
                x = blk(x)
            if len(x.shape) == 4:
                x = x.mean(dim=[2, 3])
            elif len(x.shape) == 5:
                x = x.flatten(3).mean(3)
            return x

        model.forward_features = _new_forward_features

        def _new_forward(x):
            if len(x.shape) == 4:
                x = x.repeat(T, 1, 1, 1)
            x = model.forward_features(x)
            x = model.head_lif(x)
            if hasattr(model, 'head'):
                x = model.head(x)
            TB_val = x.shape[0]; B_val = TB_val // T
            x = x.reshape(T, B_val, *x.shape[1:]).mean(0)
            return x

        if not hasattr(model, '_tdl_orig_forward'):
            model._tdl_orig_forward = orig_forward
        model._tdl_forward_4d = True
        model.forward = _new_forward

    elif hasattr(model, 'features') and hasattr(model, 'classifier'):
        # VGG style: forward → features → temporal_mean → pool → classifier
        def _new_forward(x):
            if len(x.shape) == 4:
                x = x.repeat(T, 1, 1, 1)  # (B,C,H,W) → (T*B,C,H,W)
            # features: SeqToANNContainer handles T-merged batch via TDL-1
            x = model.features(x)
            # Temporal mean: (T*B, C, H, W) → (B, C, H, W)
            if len(x.shape) == 5:
                x = x.mean(dim=0)
            else:
                TB_val = x.shape[0]; B_val = TB_val // T
                x = x.reshape(T, B_val, *x.shape[1:]).mean(0)
            import torch.nn.functional as F
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.flatten(1)
            x = model.classifier(x)
            return x

        if not hasattr(model, '_tdl_orig_forward'):
            model._tdl_orig_forward = orig_forward
        model._tdl_forward_4d = True
        model.forward = _new_forward
        n_patched += 1  # count top-level forward as patched

    if n_patched > 0 and verbose:
        print(f"  Patched {n_patched} blocks/forward for 4D TDL-compatible export")


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
    force_native_onnx: bool = True,
    verbose: bool = True,
):
    """Export SNN model to ONNX with TDL transforms applied.

    Args:
        model:             PyTorch SNN model.
        onnx_path:         Output ONNX file path.
        input_shape:       (B, C, H, W) for tracing.
        opset:             ONNX opset version.
        dynamic_batch:     Enable dynamic batch dim.
        flatten_temporal:  Apply TDL-1/2/3 transforms (4D batched mode).
        force_native_onnx: Force all neurons (including attention) to use
                           native ONNX ops instead of plugin ops. Required
                           for TRT (no FusedLIFNeuron plugin registered).
        verbose:           Print progress.
    """
    from pathlib import Path
    from models.neurons import reset_net

    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    T = getattr(model, 'T', 4)

    tdl = TDLTransform(model, T, force_native_onnx=force_native_onnx)
    neuron_params = collect_neuron_params(model)

    if verbose:
        n_lif = sum(1 for p in neuron_params.values() if p['type'] == 'LIF')
        n_if = sum(1 for p in neuron_params.values() if p['type'] == 'IF')
        n_ms = sum(1 for p in neuron_params.values() if p['type'] == 'MS')
        n_ilif = sum(1 for p in neuron_params.values() if p['type'] == 'ILIF')
        parts = [f"{n_lif} LIF"] if n_lif else []
        parts += [f"{n_if} IF"] if n_if else []
        parts += [f"{n_ms} MS"] if n_ms else []
        parts += [f"{n_ilif} ILIF"] if n_ilif else []
        print(f"  Found {' + '.join(parts)} neurons")

    if flatten_temporal:
        # Pre-process: wrap bare Conv+BN in SeqToANNContainer for models
        # that do manual T*B reshaping (DVS variants).
        _wrap_bare_convs_for_tdl(model, T, verbose=verbose)
        if verbose:
            print(f"  Applying TDL-1/2/3 batched mode (T={T})")
        tdl.apply()
    else:
        if verbose:
            print(f"  Legacy 5D export mode")
        tdl = None

    # Set trace_batch_size for native ONNX symbolic
    import sengine_edge.tdl.neuron_ops as _nops
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
            export_kwargs = dict(
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                opset_version=opset,
                do_constant_folding=False,
            )
            try:
                torch.onnx.export(model, dummy, onnx_path, dynamo=False, **export_kwargs)
            except TypeError:
                torch.onnx.export(model, dummy, onnx_path, **export_kwargs)
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
