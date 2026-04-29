"""MaxFormer / MS_QKFormer DAG extraction."""

from sengine.tdl.analysis import collect_neuron_params
from sengine.tdl.graph_ir import (
    OperatorDAG, _conv2d_params, _conv2d_out_shape, _pool_out_shape, _intify,
    _add_conv_bn, _add_neuron, _add_bn, _add_matmul, _add_pool,
)

# Reuse QKFormer's patch embed extractor for PatchEmbedInitMaxPool
from sengine.tdl.model_dag.qkformer import _extract_qkformer_patch_embed_init

def _extract_maxformer_s_mlp(dag, mlp, prefix, in_shape, in_node_id, nparams):
    """S_MLP: lif1→conv1(k=1)→bn1 [+mid-res] →lif2→conv2(k=1)→bn2 +outer-res."""
    B, C, H, W = in_shape
    hidden = mlp.c_hidden
    hidden_shape = (B, hidden, H, W)

    lif1_id = _add_neuron(dag, f'{prefix}.fc1_lif', in_shape, nparams, in_node_id)
    p1 = _conv2d_params(mlp.fc1_conv)
    c1_id = dag.add_node(f'{prefix}.fc1_conv', 'conv2d', False, p1, in_shape, hidden_shape)
    dag.add_edge(lif1_id, c1_id)
    bn1_id = _add_bn(dag, mlp.fc1_bn, f'{prefix}.fc1_bn', hidden_shape, c1_id)

    # Mid-residual if in_features == hidden_features
    if mlp.res:
        mid_res = dag.add_node(f'{prefix}.mid_residual', 'add', False, {},
                               hidden_shape, hidden_shape)
        dag.add_edge(bn1_id, mid_res)
        dag.add_edge(in_node_id, mid_res)
        prev_id = mid_res
    else:
        prev_id = bn1_id

    lif2_id = _add_neuron(dag, f'{prefix}.fc2_lif', hidden_shape, nparams, prev_id)
    p2 = _conv2d_params(mlp.fc2_conv)
    c2_id = dag.add_node(f'{prefix}.fc2_conv', 'conv2d', False, p2, hidden_shape, in_shape)
    dag.add_edge(lif2_id, c2_id)
    bn2_id = _add_bn(dag, mlp.fc2_bn, f'{prefix}.fc2_bn', in_shape, c2_id)

    # Outer residual
    if mlp.res:
        out_res = dag.add_node(f'{prefix}.out_residual', 'add', False, {},
                               in_shape, in_shape)
        dag.add_edge(bn2_id, out_res)
        dag.add_edge(mid_res, out_res)
        return out_res
    else:
        out_res = dag.add_node(f'{prefix}.out_residual', 'add', False, {},
                               in_shape, in_shape)
        dag.add_edge(bn2_id, out_res)
        dag.add_edge(in_node_id, out_res)
        return out_res


def _extract_maxformer_dwc_block(dag, blk, prefix, in_shape, in_node_id, nparams):
    """Block_DWC: LIF→DWConv→BN + residual, then S_MLP."""
    lif_id = _add_neuron(dag, f'{prefix}.conv_neuron', in_shape, nparams, in_node_id)
    p = _conv2d_params(blk.conv)
    conv_id = dag.add_node(f'{prefix}.conv', 'conv2d', False, p, in_shape, in_shape)
    dag.add_edge(lif_id, conv_id)
    bn_id = _add_bn(dag, blk.conv_bn, f'{prefix}.conv_bn', in_shape, conv_id)

    res = dag.add_node(f'{prefix}.conv_residual', 'add', False, {}, in_shape, in_shape)
    dag.add_edge(bn_id, res)
    dag.add_edge(in_node_id, res)

    mlp_id = _extract_maxformer_s_mlp(dag, blk.mlp, f'{prefix}.mlp',
                                      in_shape, res, nparams)
    return mlp_id


def _extract_maxformer_ssa(dag, ssa, prefix, in_shape, in_node_id, nparams):
    """SSA (MaxFormer): x_lif → Q/K/V Conv1d→BN→LIF → K^T@V, Q@result → attn_lif → proj + residual."""
    B, C, H, W = in_shape
    seq_shape = (B, C, H * W)

    x_lif_id = _add_neuron(dag, f'{prefix}.x_lif', in_shape, nparams, in_node_id)

    paths = {}
    for name in ['q', 'k', 'v']:
        bn = getattr(ssa, f'{name}_bn')
        p_id = dag.add_node(f'{prefix}.{name}_conv', 'linear', False,
                            {'in_features': C, 'out_features': C}, seq_shape, seq_shape)
        dag.add_edge(x_lif_id, p_id)
        bn_id = _add_bn(dag, bn, f'{prefix}.{name}_bn', seq_shape, p_id)
        lif_id = _add_neuron(dag, f'{prefix}.{name}_lif', seq_shape, nparams, bn_id)
        paths[name] = lif_id

    kv_id = _add_matmul(dag, f'{prefix}.kv_matmul', seq_shape, seq_shape,
                         paths['k'], paths['v'])
    qkv_id = _add_matmul(dag, f'{prefix}.qkv_matmul', seq_shape, seq_shape,
                          paths['q'], kv_id)
    attn_lif_id = _add_neuron(dag, f'{prefix}.attn_lif', seq_shape, nparams, qkv_id)

    proj_id = dag.add_node(f'{prefix}.proj_conv', 'linear', False,
                           {'in_features': C, 'out_features': C}, seq_shape, seq_shape)
    dag.add_edge(attn_lif_id, proj_id)
    proj_bn_id = _add_bn(dag, ssa.proj_bn, f'{prefix}.proj_bn', seq_shape, proj_id)

    # SSA includes internal residual: x = x + identity
    res = dag.add_node(f'{prefix}.residual', 'add', False, {}, in_shape, in_shape)
    dag.add_edge(proj_bn_id, res)
    dag.add_edge(in_node_id, res)
    return res


def _extract_maxformer_ssa_block(dag, blk, prefix, in_shape, in_node_id, nparams):
    """Block_SSA: SSA (with internal residual) → S_MLP."""
    ssa_id = _extract_maxformer_ssa(dag, blk.attn, f'{prefix}.attn',
                                   in_shape, in_node_id, nparams)
    mlp_id = _extract_maxformer_s_mlp(dag, blk.mlp, f'{prefix}.mlp',
                                      in_shape, ssa_id, nparams)
    return mlp_id


def _extract_embed(dag, emb, prefix, in_shape, in_node_id, nparams, has_maxpool=False):
    """Embed or MaxEmbed module. Returns (node_id, output_shape).
    If shortcut=False, includes LIF before conv."""
    cur_id = in_node_id
    if not emb.shortcut and cur_id is not None:
        cur_id = _add_neuron(dag, f'{prefix}.embed_lif', in_shape, nparams, cur_id)
    conv = emb.embed_conv
    p = _conv2d_params(conv)
    out = _conv2d_out_shape(in_shape, p['out_channels'], p['kernel_size'],
                            p['stride'], p['padding'], p['dilation'])
    conv_id = dag.add_node(f'{prefix}.embed_conv', 'conv2d', False, p, in_shape, out)
    if cur_id is not None:
        dag.add_edge(cur_id, conv_id)
    bn_id = _add_bn(dag, emb.embed_bn, f'{prefix}.embed_bn', out, conv_id)

    if has_maxpool:
        pool_id, out = _add_pool(dag, emb.maxpool, f'{prefix}.maxpool', out, bn_id)
        return pool_id, out
    return bn_id, out


def _extract_embed_orig_imagenet(dag, eoi, prefix, in_shape, in_node_id, nparams):
    """EmbedOrigImageNet: embed1→embed2(dual)→embed3 + embed4(shortcut)."""
    B, C, H, W = in_shape
    e1_id, _ = _extract_embed(dag, eoi.embed1, f'{prefix}.embed1',
                              in_shape, in_node_id, nparams)
    s1 = (B, eoi.embed1.embed_conv.out_channels, H // 2, W // 2)
    # Manually set output shape after stride-2 (embed1 already computed it but
    # we need to track for embed2 input)

    e2_id, _ = _extract_embed(dag, eoi.embed2, f'{prefix}.embed2', s1, e1_id, nparams)
    s2 = (B, eoi.embed2.embed_conv.out_channels, H // 4, W // 4)

    e3_id, _ = _extract_embed(dag, eoi.embed3, f'{prefix}.embed3', s2, e2_id, nparams)

    # Shortcut: embed4 from embed2 input (s1 shape), stride-2
    e4_id, _ = _extract_embed(dag, eoi.embed4, f'{prefix}.embed4', s1, e1_id, nparams)

    res_id = dag.add_node(f'{prefix}.residual', 'add', False, {}, s2, s2)
    dag.add_edge(e3_id, res_id)
    dag.add_edge(e4_id, res_id)
    return res_id, s2


def _extract_embed_max(dag, em, prefix, in_shape, in_node_id, nparams):
    """EmbedMax / Embed1Max / Embed1MaxCifar: main path + residual shortcut, add.
    Handles varying internal structure by checking which attributes exist."""
    B, C, H, W = in_shape
    from models.maxformer import Embed1MaxCifar

    if isinstance(em, Embed1MaxCifar):
        # Embed1MaxCifar: embed1(Embed, no maxpool) → max_embed1(MaxEmbed) + embed2(shortcut)
        e1_id, e1_shape = _extract_embed(dag, em.embed1, f'{prefix}.embed1',
                                         in_shape, in_node_id, nparams)
        e1_spatial = (B, em.embed1.embed_conv.out_channels, H, W)
        me1_id, me1_shape = _extract_embed(dag, em.max_embed1, f'{prefix}.max_embed1',
                                           e1_spatial, e1_id, nparams, has_maxpool=True)
        e2_id, e2_shape = _extract_embed(dag, em.embed2, f'{prefix}.embed2',
                                         in_shape, in_node_id, nparams)
        out_shape = me1_shape
    elif hasattr(em, 'max_embed1') and hasattr(em, 'max_embed2'):
        # EmbedMax: MaxEmbed(main)→Embed + MaxEmbed(shortcut)
        me1_id, me1_shape = _extract_embed(dag, em.max_embed1, f'{prefix}.max_embed1',
                                           in_shape, in_node_id, nparams, has_maxpool=True)
        e1_id, e1_shape = _extract_embed(dag, em.embed1, f'{prefix}.embed1',
                                         me1_shape, me1_id, nparams)
        e2_id, _ = _extract_embed(dag, em.max_embed2, f'{prefix}.max_embed2',
                                  in_shape, in_node_id, nparams,
                                  has_maxpool=hasattr(em.max_embed2, 'maxpool'))
        me1_id = e1_id  # main output is after embed1
        out_shape = e1_shape
    else:
        raise ValueError(f"Unknown embed type in _extract_embed_max: {type(em)}")

    res_id = dag.add_node(f'{prefix}.residual', 'add', False, {}, out_shape, out_shape)
    dag.add_edge(me1_id, res_id)
    dag.add_edge(e2_id, res_id)
    return res_id, out_shape


def _extract_patch_embed_init_maxpool(dag, pe, prefix, in_shape, in_node_id, nparams):
    """PatchEmbedInitMaxPool (MS_QKFormer stem):
    embed1.conv→bn→maxpool1→lif1 → embed2.conv→bn→maxpool2→lif2
    → embed3.conv→bn + embed4(residual).conv→bn → add."""
    B, C, H, W = in_shape
    # Stage 1: embed1 conv→bn → maxpool1 → lif1
    e1_conv = pe.embed1.embed_conv
    p1 = _conv2d_params(e1_conv)
    out1 = _conv2d_out_shape(in_shape, p1['out_channels'], p1['kernel_size'],
                             p1['stride'], p1['padding'], p1['dilation'])
    e1_id = dag.add_node(f'{prefix}.embed1.embed_conv', 'conv2d', False, p1, in_shape, out1)
    if in_node_id is not None:
        dag.add_edge(in_node_id, e1_id)
    e1_bn_id = _add_bn(dag, pe.embed1.embed_bn, f'{prefix}.embed1.embed_bn', out1, e1_id)
    mp1_id, mp1_shape = _add_pool(dag, pe.maxpool1, f'{prefix}.maxpool1', out1, e1_bn_id)
    lif1_id = _add_neuron(dag, f'{prefix}.lif1', mp1_shape, nparams, mp1_id)

    # Stage 2: embed2 conv→bn → maxpool2 → lif2
    e2_conv = pe.embed2.embed_conv
    p2 = _conv2d_params(e2_conv)
    out2 = _conv2d_out_shape(mp1_shape, p2['out_channels'], p2['kernel_size'],
                             p2['stride'], p2['padding'], p2['dilation'])
    e2_id = dag.add_node(f'{prefix}.embed2.embed_conv', 'conv2d', False, p2, mp1_shape, out2)
    dag.add_edge(lif1_id, e2_id)
    e2_bn_id = _add_bn(dag, pe.embed2.embed_bn, f'{prefix}.embed2.embed_bn', out2, e2_id)
    mp2_id, mp2_shape = _add_pool(dag, pe.maxpool2, f'{prefix}.maxpool2', out2, e2_bn_id)
    lif2_id = _add_neuron(dag, f'{prefix}.lif2', mp2_shape, nparams, mp2_id)

    # Stage 3: embed3 conv→bn
    e3_conv = pe.embed3.embed_conv
    p3 = _conv2d_params(e3_conv)
    out3 = _conv2d_out_shape(mp2_shape, p3['out_channels'], p3['kernel_size'],
                             p3['stride'], p3['padding'], p3['dilation'])
    e3_id = dag.add_node(f'{prefix}.embed3.embed_conv', 'conv2d', False, p3, mp2_shape, out3)
    dag.add_edge(lif2_id, e3_id)
    e3_bn_id = _add_bn(dag, pe.embed3.embed_bn, f'{prefix}.embed3.embed_bn', out3, e3_id)

    # Residual: embed4 conv→bn from lif1 output
    e4_conv = pe.embed4.embed_conv
    p4 = _conv2d_params(e4_conv)
    out4 = _conv2d_out_shape(mp1_shape, p4['out_channels'], p4['kernel_size'],
                             p4['stride'], p4['padding'], p4['dilation'])
    e4_id = dag.add_node(f'{prefix}.embed4.embed_conv', 'conv2d', False, p4, mp1_shape, out4)
    dag.add_edge(lif1_id, e4_id)
    e4_bn_id = _add_bn(dag, pe.embed4.embed_bn, f'{prefix}.embed4.embed_bn', out4, e4_id)

    add_id = dag.add_node(f'{prefix}.residual', 'add', False, {}, out3, out3)
    dag.add_edge(e3_bn_id, add_id)
    dag.add_edge(e4_bn_id, add_id)
    return add_id, out3


def _extract_embed_orig_cifar(dag, eo, prefix, in_shape, in_node_id, nparams):
    """EmbedOrig (CIFAR): embed1→embed2(dual)→embed3(shortcut), add. No spatial downsampling."""
    B, C, H, W = in_shape
    e1_id, _ = _extract_embed(dag, eo.embed1, f'{prefix}.embed1',
                              in_shape, in_node_id, nparams)
    s1 = (B, eo.embed1.embed_conv.out_channels, H, W)

    e2_id, _ = _extract_embed(dag, eo.embed2, f'{prefix}.embed2', s1, e1_id, nparams)
    s2 = (B, eo.embed2.embed_conv.out_channels, H, W)

    # Shortcut: embed3 from e1 (shortcut=True means no LIF)
    e3_id, _ = _extract_embed(dag, eo.embed3, f'{prefix}.embed3', s1, e1_id, nparams)

    res_id = dag.add_node(f'{prefix}.residual', 'add', False, {}, s2, s2)
    dag.add_edge(e2_id, res_id)
    dag.add_edge(e3_id, res_id)
    return res_id, s2


def extract_maxformer_dag(model, input_shape):
    """MaxFormer / MaxFormerCifar / MS_QKFormer / MS_QKFormerCifar."""
    dag = OperatorDAG()
    nparams = collect_neuron_params(model)

    # Determine patch_embed1 type
    from models.maxformer import EmbedOrigImageNet, EmbedOrig, PatchEmbedInitMaxPool
    pe1 = model.patch_embed1
    if isinstance(pe1, EmbedOrigImageNet):
        cur_id, cur_shape = _extract_embed_orig_imagenet(
            dag, pe1, 'patch_embed1', input_shape, None, nparams)
    elif isinstance(pe1, EmbedOrig):
        cur_id, cur_shape = _extract_embed_orig_cifar(
            dag, pe1, 'patch_embed1', input_shape, None, nparams)
    elif isinstance(pe1, PatchEmbedInitMaxPool):
        cur_id, cur_shape = _extract_patch_embed_init_maxpool(
            dag, pe1, 'patch_embed1', input_shape, None, nparams)
    else:
        raise ValueError(f"Unknown patch_embed1 type: {type(pe1)}")
    from models.maxformer import (Block_DWC, Block_SSA, Block_QKA,
                                   Block_identity, Block_Max,
                                   EmbedMax, Embed1Max, Embed1MaxCifar)

    def _extract_stage_block(blk, prefix):
        nonlocal cur_id
        if isinstance(blk, Block_SSA):
            cur_id = _extract_maxformer_ssa_block(dag, blk, prefix,
                                                  cur_shape, cur_id, nparams)
        elif isinstance(blk, Block_DWC):
            cur_id = _extract_maxformer_dwc_block(dag, blk, prefix,
                                                  cur_shape, cur_id, nparams)
        elif isinstance(blk, Block_QKA):
            # Token_QK_Attention(.attn): proj_lif→Q/K→sum(Q)→attn_lif→mul(K)→proj + residual, then S_MLP
            attn = blk.attn  # Token_QK_Attention
            B_, C_, H_, W_ = cur_shape
            seq_ = (B_, C_, H_ * W_)
            # proj_lif → Q/K paths
            pl_id = _add_neuron(dag, f'{prefix}.attn.proj_lif', cur_shape, nparams, cur_id)
            q_id = dag.add_node(f'{prefix}.attn.q_conv', 'linear', False,
                                {'in_features': C_, 'out_features': C_}, seq_, seq_)
            dag.add_edge(pl_id, q_id)
            q_bn = _add_bn(dag, attn.q_bn, f'{prefix}.attn.q_bn', seq_, q_id)
            q_lif = _add_neuron(dag, f'{prefix}.attn.q_lif', seq_, nparams, q_bn)

            k_id = dag.add_node(f'{prefix}.attn.k_conv', 'linear', False,
                                {'in_features': C_, 'out_features': C_}, seq_, seq_)
            dag.add_edge(pl_id, k_id)
            k_bn = _add_bn(dag, attn.k_bn, f'{prefix}.attn.k_bn', seq_, k_id)
            k_lif = _add_neuron(dag, f'{prefix}.attn.k_lif', seq_, nparams, k_bn)

            attn_lif = _add_neuron(dag, f'{prefix}.attn.attn_lif', seq_, nparams, q_lif)
            mul_id = dag.add_node(f'{prefix}.attn.attn_mul', 'mul', False, {}, seq_, seq_)
            dag.add_edge(attn_lif, mul_id)
            dag.add_edge(k_lif, mul_id)

            proj_id = dag.add_node(f'{prefix}.attn.proj_conv', 'linear', False,
                                   {'in_features': C_, 'out_features': C_}, seq_, seq_)
            dag.add_edge(mul_id, proj_id)
            proj_bn = _add_bn(dag, attn.proj_bn, f'{prefix}.attn.proj_bn', seq_, proj_id)

            res_id = dag.add_node(f'{prefix}.attn_residual', 'add', False, {},
                                  cur_shape, cur_shape)
            dag.add_edge(proj_bn, res_id)
            dag.add_edge(cur_id, res_id)
            cur_id = _extract_maxformer_s_mlp(dag, blk.mlp, f'{prefix}.mlp',
                                              cur_shape, res_id, nparams)
        elif isinstance(blk, Block_identity):
            cur_id = _extract_maxformer_s_mlp(dag, blk.mlp, f'{prefix}.mlp',
                                              cur_shape, cur_id, nparams)
        elif isinstance(blk, Block_Max):
            # MaxPool mixer then MLP
            pool_shape = cur_shape  # MaxPool k=3,s=1,p=1 → same shape
            pool_id = dag.add_node(f'{prefix}.pool', 'maxpool2d', False,
                                   {'kernel_size': 3, 'stride': 1, 'padding': 1},
                                   cur_shape, pool_shape)
            dag.add_edge(cur_id, pool_id)
            cur_id = _extract_maxformer_s_mlp(dag, blk.mlp, f'{prefix}.mlp',
                                              pool_shape, pool_id, nparams)

    def _extract_embed_stage(pe, prefix):
        nonlocal cur_id, cur_shape
        if isinstance(pe, (EmbedMax, Embed1Max)):
            cur_id, cur_shape = _extract_embed_max(dag, pe, prefix,
                                                   cur_shape, cur_id, nparams)
        elif isinstance(pe, Embed1MaxCifar):
            cur_id, cur_shape = _extract_embed_max(dag, pe, prefix,
                                                   cur_shape, cur_id, nparams)
        else:
            raise ValueError(f"Unknown patch embed type: {type(pe)}")

    for i, blk in enumerate(model.stage1):
        _extract_stage_block(blk, f'stage1.{i}')

    _extract_embed_stage(model.patch_embed2, 'patch_embed2')
    for i, blk in enumerate(model.stage2):
        _extract_stage_block(blk, f'stage2.{i}')

    _extract_embed_stage(model.patch_embed3, 'patch_embed3')
    for i, blk in enumerate(model.stage3):
        _extract_stage_block(blk, f'stage3.{i}')

    lif_id = _add_neuron(dag, 'head_lif', cur_shape, nparams, cur_id)
    head = model.head
    head_in = (cur_shape[0], head.in_features)
    head_out = (cur_shape[0], head.out_features)
    head_id = dag.add_node('head', 'linear', False,
                           {'in_features': head.in_features,
                            'out_features': head.out_features},
                           head_in, head_out)
    dag.add_edge(lif_id, head_id)
    return dag

