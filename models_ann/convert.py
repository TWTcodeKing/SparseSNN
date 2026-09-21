"""SNN → ANN model converter.

Converts any SNN model from models/ into an equivalent ANN model by:
  1. Replacing spiking neurons (LIF/IF/MS/ILIF/PLIF) with nn.ReLU
  2. Unwrapping SeqToANNContainer (extracts inner module)
  3. Patching forward() to remove T-dimension handling
  4. Removing reset_net() / membrane state

The resulting model accepts standard (B, C, H, W) input (vision) or
(B, L) input (NLP), with no temporal dimension.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Neuron detection ──

_NEURON_CLASS_NAMES = {
    'MultiStepLIFNeuron', 'MultiStepIFNeuron', 'MultiStepILIFNeuron',
    'LIFNeuron', 'IFNeuron', 'ILIFNeuron',
    'MSNeuron', 'PLIFNeuron',
    'LIF',  # SpikingResformer's LIF subclass
}


def _is_neuron(module: nn.Module) -> bool:
    return type(module).__name__ in _NEURON_CLASS_NAMES


def _is_seq_container(module: nn.Module) -> bool:
    return type(module).__name__ in ('SeqToANNContainer', 'SeqToANNContainerT')


# ── In-place conversion ──

def _replace_neurons_inplace(model: nn.Module):
    """Replace all spiking neurons with ReLU."""
    for name, child in model.named_children():
        if _is_neuron(child):
            setattr(model, name, nn.ReLU(inplace=True))
        else:
            _replace_neurons_inplace(child)


def _unwrap_seq_containers_inplace(model: nn.Module):
    """Replace SeqToANNContainer with its inner module."""
    for name, child in model.named_children():
        if _is_seq_container(child):
            inner = child.module
            setattr(model, name, inner)
            _unwrap_seq_containers_inplace(inner)
        else:
            _unwrap_seq_containers_inplace(child)


def _replace_tdbn_inplace(model: nn.Module):
    """Replace TDBNContainer (Conv2d + BN3d) with Sequential(Conv2d, BN2d)."""
    for name, child in model.named_children():
        cls_name = type(child).__name__
        if cls_name == 'TDBNContainer':
            inner = child.module
            children_list = list(inner.children())
            if len(children_list) == 2:
                conv = children_list[0]
                bn3d = children_list[1]
                if isinstance(bn3d, nn.BatchNorm3d):
                    bn2d = nn.BatchNorm2d(bn3d.num_features)
                    bn2d.weight = bn3d.weight
                    bn2d.bias = bn3d.bias
                    bn2d.running_mean = bn3d.running_mean
                    bn2d.running_var = bn3d.running_var
                    bn2d.num_batches_tracked = bn3d.num_batches_tracked
                    setattr(model, name, nn.Sequential(conv, bn2d))
                    continue
        _replace_tdbn_inplace(child)


def _replace_avgpool3d_inplace(model: nn.Module):
    """Replace AvgPool3d with AvgPool2d (drop temporal dim)."""
    for name, child in model.named_children():
        if isinstance(child, nn.AvgPool3d):
            # AvgPool3d kernel (1, kH, kW), stride (1, sH, sW) → AvgPool2d kernel (kH,kW), stride (sH,sW)
            ks = child.kernel_size
            st = child.stride
            if isinstance(ks, (tuple, list)) and len(ks) == 3:
                setattr(model, name, nn.AvgPool2d(kernel_size=ks[1:], stride=st[1:]))
            else:
                setattr(model, name, nn.AvgPool2d(kernel_size=ks, stride=st))
        else:
            _replace_avgpool3d_inplace(child)


def _patch_msresnet_blocks_inplace(model: nn.Module):
    """Patch MS-ResNet BasicBlock forward methods to remove 5D permute."""
    import types
    for name, child in model.named_modules():
        cls_name = type(child).__name__
        if cls_name in ('BasicBlock18', 'BasicBlockCifar', 'BasicBlock104'):
            def _make_block_fwd(blk):
                def forward(self, x):
                    out = self.conv_bn1(self.sn1(x))
                    out = self.conv_bn2(self.sn2(out))
                    sc = self.shortcut(x) if len(self.shortcut) > 0 else x
                    return out + sc
                return forward
            child.forward = types.MethodType(_make_block_fwd(child), child)


def _patch_multistep_ops_inplace(model: nn.Module):
    """Patch _MultiStep* ops (SpikingResformer) to use base class forward directly."""
    import types
    for name, child in model.named_modules():
        cls_name = type(child).__name__
        if cls_name in ('_MultiStepConv2d', 'Conv3x3', 'Conv1x1'):
            def _mk(m):
                def fwd(self, x):
                    return nn.Conv2d.forward(self, x)
                return fwd
            child.forward = types.MethodType(_mk(child), child)
        elif cls_name == '_MultiStepMaxPool2d':
            def _mk(m):
                def fwd(self, x):
                    return nn.MaxPool2d.forward(self, x)
                return fwd
            child.forward = types.MethodType(_mk(child), child)
        elif cls_name == '_MultiStepAvgPool2d':
            def _mk(m):
                def fwd(self, x):
                    return nn.AdaptiveAvgPool2d.forward(self, x)
                return fwd
            child.forward = types.MethodType(_mk(child), child)
        elif cls_name == '_MultiStepLinear':
            def _mk(m):
                def fwd(self, x):
                    return nn.Linear.forward(self, x)
                return fwd
            child.forward = types.MethodType(_mk(child), child)

        # BN wrapper (SpikingResformer): flatten(0,1) assumes 5D → just use inner bn
        elif cls_name == 'BN' and hasattr(child, 'bn') and isinstance(child.bn, nn.BatchNorm2d):
            def _mk(m):
                def fwd(self, x):
                    return self.bn(x)
                return fwd
            child.forward = types.MethodType(_mk(child), child)

        # DSSA: deformable spike self-attention → standard 4D attention
        elif cls_name == 'DSSA':
            def _mk(m):
                def fwd(self, x):
                    # x: (B, C, H, W) — no T
                    B, C, H, W = x.shape
                    x_feat = x
                    x = self.activation_in(x)

                    y = self.W(x)
                    y = self.norm(y)
                    y = y.reshape(B, self.num_heads, 2 * C // self.num_heads, -1)
                    y1 = y[:, :, :C // self.num_heads, :]
                    y2 = y[:, :, C // self.num_heads:, :]
                    x = x.reshape(B, self.num_heads, C // self.num_heads, -1)

                    scale1 = 1.0 / (C // self.num_heads) ** 0.5
                    attn = torch.matmul(y1.transpose(-2, -1), x) * scale1
                    attn = self.activation_attn(attn)

                    scale2 = 1.0 / self.lenth ** 0.5
                    out = torch.matmul(y2, attn) * scale2
                    out = out.reshape(B, C, H, W)
                    out = self.activation_out(out)

                    out = self.Wproj(out)
                    out = self.norm_proj(out)
                    return out + x_feat
                return fwd
            child.forward = types.MethodType(_mk(child), child)


def _patch_maxformer_submodules_inplace(model: nn.Module):
    """Patch all MaxFormer submodules to remove 5D T-dimension handling."""
    import types

    for name, child in model.named_modules():
        cls_name = type(child).__name__

        # Embed classes: all do T,B,C,H,W = x.shape
        if cls_name == 'Embed':
            def _mk(m):
                def fwd(self, x, dual=False):
                    if not self.shortcut:
                        x = self.embed_lif(x)
                    x_feat = x
                    x = self.embed_conv(x)
                    x = self.embed_bn(x)
                    return (x, x_feat) if dual else x
                return fwd
            child.forward = types.MethodType(_mk(child), child)

        elif cls_name == 'MaxEmbed':
            def _mk(m):
                def fwd(self, x, dual=False):
                    if not self.shortcut:
                        x = self.embed_lif(x)
                    x_feat = x
                    x = self.embed_conv(x)
                    x = self.embed_bn(x)
                    x = self.maxpool(x)
                    return (x, x_feat) if dual else x
                return fwd
            child.forward = types.MethodType(_mk(child), child)

        elif cls_name == 'EmbedOrigImageNet':
            def _mk(m):
                def fwd(self, x):
                    x = self.embed1(x)
                    x, x_feat = self.embed2(x, dual=True)
                    x = self.embed3(x)
                    x_feat = self.embed4(x_feat)
                    return x + x_feat
                return fwd
            child.forward = types.MethodType(_mk(child), child)

        elif cls_name == 'EmbedOrig':
            def _mk(m):
                def fwd(self, x):
                    x = self.embed1(x)
                    x, x_feat = self.embed2(x, dual=True)
                    x_feat = self.embed3(x_feat)
                    return x + x_feat
                return fwd
            child.forward = types.MethodType(_mk(child), child)

        elif cls_name == 'EmbedMax':
            def _mk(m):
                def fwd(self, x):
                    x, x_feat = self.max_embed1(x, dual=True)
                    x = self.embed1(x)
                    x_feat = self.max_embed2(x_feat)
                    return x + x_feat
                return fwd
            child.forward = types.MethodType(_mk(child), child)

        elif cls_name == 'Embed1Max':
            def _mk(m):
                def fwd(self, x):
                    x, x_feat = self.max_embed1(x, dual=True)
                    x = self.embed1(x)
                    x_feat = self.max_embed2(x_feat)
                    return x + x_feat
                return fwd
            child.forward = types.MethodType(_mk(child), child)

        elif cls_name == 'Embed1MaxCifar':
            def _mk(m):
                def fwd(self, x):
                    x, x_feat = self.embed1(x, dual=True)
                    x = self.max_embed1(x)
                    x_feat = self.embed2(x_feat)
                    return x + x_feat
                return fwd
            child.forward = types.MethodType(_mk(child), child)

        elif cls_name == 'EmbedMaxPlus':
            def _mk(m):
                def fwd(self, x):
                    x = self.proj_conv(x)
                    x = self.proj_bn(x)
                    x = self.max_embed1(x)
                    x, x_feat = self.max_embed2(x, dual=True)
                    x = self.max_embed3(x)
                    x_feat = self.embed1(x_feat)
                    return x + x_feat
                return fwd
            child.forward = types.MethodType(_mk(child), child)

        elif cls_name == 'PatchEmbedInitMaxPool':
            def _mk(m):
                def fwd(self, x):
                    x = self.embed1.embed_conv(x)
                    x = self.embed1.embed_bn(x)
                    x = self.maxpool1(x)
                    x = self.lif1(x)
                    x_feat = x
                    x = self.embed2.embed_conv(x)
                    x = self.embed2.embed_bn(x)
                    x = self.maxpool2(x)
                    x = self.lif2(x)
                    x = self.embed3.embed_conv(x)
                    x = self.embed3.embed_bn(x)
                    x_feat = self.embed4.embed_conv(x_feat)
                    x_feat = self.embed4.embed_bn(x_feat)
                    return x + x_feat
                return fwd
            child.forward = types.MethodType(_mk(child), child)

        # S_MLP: spiking MLP with residuals
        elif cls_name == 'S_MLP':
            def _mk(m):
                def fwd(self, x):
                    identity = x
                    x = self.fc1_lif(x)
                    x = self.fc1_conv(x)
                    x = self.fc1_bn(x)
                    if self.res:
                        x = identity + x
                        identity = x
                    x = self.fc2_lif(x)
                    x = self.fc2_conv(x)
                    x = self.fc2_bn(x)
                    return x + identity
                return fwd
            child.forward = types.MethodType(_mk(child), child)

        # Block classes
        elif cls_name == 'Block_DWC':
            def _mk(m):
                def fwd(self, x):
                    identity = x
                    x = self.conv_neuron(x)
                    x = self.conv(x)
                    x = self.conv_bn(x)
                    x = x + identity
                    x = self.mlp(x)
                    return x
                return fwd
            child.forward = types.MethodType(_mk(child), child)

        elif cls_name == 'Block_Max':
            def _mk(m):
                def fwd(self, x):
                    x = self.pool(x)
                    x = self.mlp(x)
                    return x
                return fwd
            child.forward = types.MethodType(_mk(child), child)

        elif cls_name == 'Block_identity':
            def _mk(m):
                def fwd(self, x):
                    return self.mlp(x)
                return fwd
            child.forward = types.MethodType(_mk(child), child)

        # MaxFormer SSA: Conv1d attention with 5D reshapes → patch for 4D
        elif cls_name == 'SSA' and hasattr(child, 'q_conv') and hasattr(child, 'x_lif'):
            def _mk(m):
                def fwd(self, x):
                    # x: (B, C, H, W)
                    B, C, H, W = x.shape
                    identity = x
                    x = self.x_lif(x)
                    N = H * W
                    x_flat = x.reshape(B, C, N)  # (B, C, N)

                    q = self.q_lif(self.q_bn(self.q_conv(x_flat)))
                    q = q.transpose(1, 2).reshape(B, N, self.num_heads, C // self.num_heads
                                                  ).permute(0, 2, 1, 3)
                    k = self.k_lif(self.k_bn(self.k_conv(x_flat)))
                    k = k.transpose(1, 2).reshape(B, N, self.num_heads, C // self.num_heads
                                                  ).permute(0, 2, 1, 3)
                    v = self.v_lif(self.v_bn(self.v_conv(x_flat)))
                    v = v.transpose(1, 2).reshape(B, N, self.num_heads, C // self.num_heads
                                                  ).permute(0, 2, 1, 3)

                    attn = (q @ (k.transpose(-2, -1) @ v)) * self.scale
                    x = attn.transpose(2, 3).reshape(B, C, N)
                    x = self.attn_lif(x)
                    x = self.proj_bn(self.proj_conv(x)).reshape(B, C, H, W)
                    return x + identity
                return fwd
            child.forward = types.MethodType(_mk(child), child)

        elif cls_name == 'Block_SSA':
            def _mk(m):
                def fwd(self, x):
                    x = self.attn(x)
                    x = self.mlp(x)
                    return x
                return fwd
            child.forward = types.MethodType(_mk(child), child)

        elif cls_name == 'Block_QKA':
            def _mk(m):
                def fwd(self, x):
                    x = self.attn(x)
                    x = self.mlp(x)
                    return x
                return fwd
            child.forward = types.MethodType(_mk(child), child)


# ── Forward patching ──

def _patch_sewresnet_forward(model):
    """Patch SEWResNet forward: remove T dimension."""
    def forward(self, x):
        # x: (B, C, H, W) — no T dimension
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
    import types
    model.forward = types.MethodType(forward, model)


def _patch_dvs_sewresnet_forward(model):
    """Patch DVSSEWResNet forward: input is (T*B, C, H, W), no temporal ops."""
    # Replace Flatten(2) (for 5D) with Flatten(1) (for 4D)
    conv_layers = list(model.conv.children())
    for i, layer in enumerate(conv_layers):
        if isinstance(layer, nn.Flatten) and layer.start_dim == 2:
            conv_layers[i] = nn.Flatten(1)
    model.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        # x: (T*B, C, H, W) — TB already fused by caller
        x = self.conv(x)   # Sequential: Conv+BN+ReLU... → (T*B, features)
        return self.out(x)  # Linear → (T*B, num_classes)
    import types
    model.forward = types.MethodType(forward, model)


def _patch_sewresnet_cifar_forward(model):
    """Patch SEWResNetCifar forward."""
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        return self.fc(x)
    import types
    model.forward = types.MethodType(forward, model)


def _patch_msresnet18_forward(model):
    """Patch MSResNet18 forward."""
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.sn_out(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        return self.fc(x)
    import types
    model.forward = types.MethodType(forward, model)


def _patch_msresnet104_forward(model):
    """Patch MSResNet104 forward."""
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.sn_out(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        return self.fc(x)
    import types
    model.forward = types.MethodType(forward, model)


def _patch_msresnet_cifar_forward(model):
    """Patch MSResNetCifar forward."""
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.sn_out(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        return self.fc(x)
    import types
    model.forward = types.MethodType(forward, model)


def _patch_snnvgg_forward(model):
    """Patch SNNVGG forward."""
    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        return self.classifier(x)
    import types
    model.forward = types.MethodType(forward, model)


def _patch_spikformer_submodules(model):
    """Patch all SpikFormer submodules (SPS, SSA, MLP, Block) for 4D input."""
    import types

    # Patch SPS (Spiking Patch Splitting) → standard patch embed
    sps = model.patch_embed
    def sps_forward(self, x):
        # x: (B, C, H, W)
        x = self.proj_conv(x)
        x = self.proj_bn(x)
        x = self.proj_lif(x)
        x = self.proj_conv1(x)
        x = self.proj_bn1(x)
        x = self.proj_lif1(x)
        x = self.proj_conv2(x)
        x = self.proj_bn2(x)
        x = self.proj_lif2(x)
        x = self.maxpool2(x)
        x = self.proj_conv3(x)
        x = self.proj_bn3(x)
        x = self.proj_lif3(x)
        x = self.maxpool3(x)
        x_feat = x
        x = self.rpe_conv(x)
        x = self.rpe_bn(x)
        x = self.rpe_lif(x)
        x = x + x_feat
        # (B, C, H', W') → (B, N, C)
        B, C = x.shape[:2]
        return x.flatten(2).transpose(1, 2)
    sps.forward = types.MethodType(sps_forward, sps)

    # Patch SSA (attention) for 3D (B, N, D) input
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if cls_name == 'SSA':
            def _make_ssa_fwd(m):
                def ssa_forward(self, x):
                    B, N, C = x.shape
                    x_for_qkv = x.flatten(0, 1) if x.dim() == 3 else x
                    q = self.q_lif(self.q_bn(self.q_linear(x_for_qkv).reshape(B, N, C)
                                             .transpose(-1, -2)).transpose(-1, -2))
                    k = self.k_lif(self.k_bn(self.k_linear(x_for_qkv).reshape(B, N, C)
                                             .transpose(-1, -2)).transpose(-1, -2))
                    v = self.v_lif(self.v_bn(self.v_linear(x_for_qkv).reshape(B, N, C)
                                             .transpose(-1, -2)).transpose(-1, -2))
                    q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
                    k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
                    v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
                    attn = (q @ k.transpose(-2, -1)) * self.scale
                    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    x = self.attn_lif(x)
                    x = self.proj_lif(self.proj_bn(self.proj_linear(
                        x.flatten(0, 1)).reshape(B, N, C).transpose(-1, -2)).transpose(-1, -2))
                    return x
                return ssa_forward
            module.forward = types.MethodType(_make_ssa_fwd(module), module)

        elif cls_name == 'MLP' and hasattr(module, 'fc1_linear'):
            def _make_mlp_fwd(m):
                def mlp_forward(self, x):
                    B, N, C = x.shape
                    x = self.fc1_lif(self.fc1_bn(self.fc1_linear(
                        x.flatten(0, 1)).reshape(B, N, -1).transpose(-1, -2)).transpose(-1, -2))
                    x = self.fc2_lif(self.fc2_bn(self.fc2_linear(
                        x.flatten(0, 1)).reshape(B, N, -1).transpose(-1, -2)).transpose(-1, -2))
                    return x
                return mlp_forward
            module.forward = types.MethodType(_make_mlp_fwd(module), module)

        elif cls_name == 'Block' and hasattr(module, 'attn') and hasattr(module, 'mlp'):
            def _make_block_fwd(m):
                def block_forward(self, x):
                    x = x + self.attn(x)
                    x = x + self.mlp(x)
                    return x
                return block_forward
            module.forward = types.MethodType(_make_block_fwd(module), module)

    # Patch top-level forward
    def forward(self, x):
        x = self.patch_embed(x)
        for blk in self.block:
            x = blk(x)
        x = x.mean(dim=1)
        return self.head(x)
    model.forward = types.MethodType(forward, model)


def _patch_maxformer_forward(model):
    """Patch MaxFormer forward: remove T."""
    def forward(self, x):
        x = self.patch_embed1(x)
        for blk in self.stage1:
            x = blk(x)
        x = self.patch_embed2(x)
        for blk in self.stage2:
            x = blk(x)
        x = self.patch_embed3(x)
        for blk in self.stage3:
            x = blk(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.head_lif(x)
        return self.head(x)
    import types
    model.forward = types.MethodType(forward, model)


def _patch_spikingresformer_forward(model):
    """Patch SpikingResformer forward."""
    def forward(self, x):
        x = self.prologue(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
    import types
    model.forward = types.MethodType(forward, model)


def _patch_spike_bert_forward(model):
    """Patch SpikeBERT and all submodules: remove T dimension."""
    import types

    # Patch SpikeBertSSA
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if cls_name == 'SpikeBertSSA':
            def _make_ssa(m):
                def fwd(self, x):
                    B, L, D = x.shape
                    q = self.q_lif(self.q_ln(self.q_linear(x)))
                    k = self.k_lif(self.k_ln(self.k_linear(x)))
                    v = self.v_lif(self.v_ln(self.v_linear(x)))
                    q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
                    k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
                    v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
                    attn = (q @ k.transpose(-2, -1)) * self.scale
                    x = (attn @ v).transpose(1, 2).reshape(B, L, D)
                    x = self.attn_lif(x)
                    x = self.proj_lif(self.proj_ln(self.proj(x)))
                    return x
                return fwd
            module.forward = types.MethodType(_make_ssa(module), module)

        elif cls_name == 'SpikeBertMLP':
            def _make_mlp(m):
                def fwd(self, x):
                    x = self.lif1(self.ln1(self.fc1(x)))
                    x = self.lif2(self.ln2(self.fc2(x)))
                    return x
                return fwd
            module.forward = types.MethodType(_make_mlp(module), module)

        elif cls_name == 'SpikeBertBlock':
            def _make_blk(m):
                def fwd(self, x):
                    x = x + self.attn(x)
                    x = x + self.mlp(x)
                    return x
                return fwd
            module.forward = types.MethodType(_make_blk(module), module)

    def forward(self, x):
        B, L = x.shape
        x = x.long().clamp(0, self.embedding.num_embeddings - 1)
        x = self.embedding(x) + self.pos_embedding[:, :L, :]
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        return self.head(self.head_ln(x))
    model.forward = types.MethodType(forward, model)


def _patch_detection_forward(model):
    """Patch detection model (SpikeYOLO/EMS-YOLO) forward: remove T."""
    has_backbone = hasattr(model, 'backbone')
    def forward(self, x):
        if has_backbone:
            features = self.backbone(x)
        else:
            x = self.stem_neuron(self.stem(x))
            x = self.stage1(x)
            x = self.stage2(x)
            p3 = x
            x = self.stage3(x)
            p4 = x
            x = self.stage4(x)
            p5 = x
            features = [p3, p4, p5]
        features = self.neck(features)
        return self.detect(features)
    import types
    model.forward = types.MethodType(forward, model)


# ── Architecture detection + patching ──

def _detect_and_patch_forward(model):
    """Detect model architecture and patch forward() for ANN (no T dim)."""
    cls_name = type(model).__name__

    if cls_name == 'SEWResNet':
        _patch_sewresnet_forward(model)
    elif cls_name == 'DVSSEWResNet':
        _patch_dvs_sewresnet_forward(model)
    elif cls_name == 'SEWResNetCifar':
        _patch_sewresnet_cifar_forward(model)
    elif cls_name == 'MSResNet18':
        _patch_msresnet18_forward(model)
    elif cls_name == 'MSResNet104':
        _patch_msresnet104_forward(model)
    elif cls_name == 'MSResNetCifar':
        _patch_msresnet_cifar_forward(model)
    elif cls_name == 'SNNVGG':
        _patch_snnvgg_forward(model)
    elif cls_name == 'SpikeBERT':
        _patch_spike_bert_forward(model)
    # Transformers: detect by structure
    elif hasattr(model, 'patch_embed') and hasattr(model, 'block'):
        _patch_spikformer_submodules(model)
    elif hasattr(model, 'patch_embed1') and hasattr(model, 'stage3'):
        _patch_maxformer_forward(model)
    elif hasattr(model, 'prologue') and hasattr(model, 'layers'):
        _patch_spikingresformer_forward(model)
    elif hasattr(model, 'detect'):
        _patch_detection_forward(model)
    else:
        # Generic: try removing T from existing forward
        pass  # leave forward as-is, caller handles


# ── Public API ──

def convert_snn_to_ann(model: nn.Module) -> nn.Module:
    """Convert an SNN model to ANN in-place.

    1. Replaces spiking neurons with ReLU
    2. Unwraps SeqToANNContainer
    3. Converts TDBNContainer (BN3d → BN2d)
    4. Patches forward() to remove T dimension

    Returns the modified model (same object, modified in-place).
    """
    _replace_tdbn_inplace(model)
    _replace_avgpool3d_inplace(model)
    _unwrap_seq_containers_inplace(model)
    _replace_neurons_inplace(model)
    _patch_multistep_ops_inplace(model)
    _patch_msresnet_blocks_inplace(model)
    _patch_maxformer_submodules_inplace(model)
    _detect_and_patch_forward(model)
    # Remove T attribute to signal ANN mode
    if hasattr(model, 'T'):
        delattr(model, 'T')
    return model


def build_ann_model(model_name: str = None, config: str = None,
                    num_classes: int = 100, in_channels: int = 3,
                    T: int = 4, **kwargs) -> nn.Module:
    """Build an ANN equivalent of an SNN model.

    Args:
        model_name: ResNet/VGG model name (e.g. 'sew_resnet18')
        config: YAML config path for transformer models
        num_classes: Number of output classes
        in_channels: Input channels
        T: Original T (used for reference, not in the model)

    Returns:
        ANN model that accepts (B, C, H, W) without T dimension.
    """
    from tengine.utils import build_model, build_model_from_config, load_model_config

    if config:
        import yaml
        cfg = load_model_config(config)
        cfg['num_classes'] = num_classes
        cfg['in_channels'] = in_channels
        cfg['T'] = T  # needed for construction, removed after
        if 'img_size' in kwargs:
            cfg['img_size'] = kwargs['img_size']
        model = build_model_from_config(cfg)
    elif model_name:
        model = build_model(model_name, T=T, num_classes=num_classes,
                            in_channels=in_channels)
    else:
        raise ValueError("Provide model_name or config")

    return convert_snn_to_ann(model)
