"""QKFormer DAG extraction."""

from iengine.tdl.analysis import collect_neuron_params
from iengine.tdl.graph_ir import (
    OperatorDAG, _conv2d_params, _conv2d_out_shape, _intify,
    _add_conv_bn, _add_neuron, _add_bn, _add_matmul, _add_pool,
)

def _extract_qkformer_patch_embed_init(dag, pe, prefix, in_shape, in_node_id, nparams):
    """PatchEmbedInit: conv→bn→maxpool→lif → conv1→bn1→maxpool1→lif1
    → conv2→bn2→lif2 + residual(conv_res→bn_res→lif_res)."""
    B, C, H, W = in_shape
    # Main: conv→bn→maxpool→lif
    bn_id, shape = _add_conv_bn(dag, pe.proj_conv, pe.proj_bn,
                                f'{prefix}.proj_conv', f'{prefix}.proj_bn',
                                in_shape, in_node_id)
    pool_id, shape = _add_pool(dag, pe.proj_maxpool, f'{prefix}.proj_maxpool',
                               shape, bn_id)
    lif_id = _add_neuron(dag, f'{prefix}.proj_lif', shape, nparams, pool_id)

    # conv1→bn1→maxpool1→lif1
    bn1_id, shape1 = _add_conv_bn(dag, pe.proj1_conv, pe.proj1_bn,
                                  f'{prefix}.proj1_conv', f'{prefix}.proj1_bn',
                                  shape, lif_id)
    pool1_id, shape1 = _add_pool(dag, pe.proj1_maxpool, f'{prefix}.proj1_maxpool',
                                 shape1, bn1_id)
    lif1_id = _add_neuron(dag, f'{prefix}.proj1_lif', shape1, nparams, pool1_id)

    # conv2→bn2→lif2
    bn2_id, _ = _add_conv_bn(dag, pe.proj2_conv, pe.proj2_bn,
                              f'{prefix}.proj2_conv', f'{prefix}.proj2_bn',
                              shape1, lif1_id)
    lif2_id = _add_neuron(dag, f'{prefix}.proj2_lif', shape1, nparams, bn2_id)

    # Residual: conv_res→bn_res→lif_res (from post-first-pool)
    res_bn_id, _ = _add_conv_bn(dag, pe.proj_res_conv, pe.proj_res_bn,
                                f'{prefix}.proj_res_conv', f'{prefix}.proj_res_bn',
                                shape, lif_id)
    res_lif_id = _add_neuron(dag, f'{prefix}.proj_res_lif', shape1, nparams, res_bn_id)

    add_id = dag.add_node(f'{prefix}.residual', 'add', False, {}, shape1, shape1)
    dag.add_edge(lif2_id, add_id)
    dag.add_edge(res_lif_id, add_id)
    return add_id, shape1


def _extract_qkformer_patch_embed_stage(dag, pe, prefix, in_shape, in_node_id, nparams):
    """PatchEmbedStage: conv3→bn3→maxpool3→lif3 → conv4→bn4→lif4 + residual."""
    B, C, H, W = in_shape
    bn3_id, shape3 = _add_conv_bn(dag, pe.proj3_conv, pe.proj3_bn,
                                  f'{prefix}.proj3_conv', f'{prefix}.proj3_bn',
                                  in_shape, in_node_id)
    pool3_id, shape3 = _add_pool(dag, pe.proj3_maxpool, f'{prefix}.proj3_maxpool',
                                 shape3, bn3_id)
    lif3_id = _add_neuron(dag, f'{prefix}.proj3_lif', shape3, nparams, pool3_id)

    bn4_id, _ = _add_conv_bn(dag, pe.proj4_conv, pe.proj4_bn,
                              f'{prefix}.proj4_conv', f'{prefix}.proj4_bn',
                              shape3, lif3_id)
    lif4_id = _add_neuron(dag, f'{prefix}.proj4_lif', shape3, nparams, bn4_id)

    # Residual from input
    res_bn_id, _ = _add_conv_bn(dag, pe.proj_res_conv, pe.proj_res_bn,
                                f'{prefix}.proj_res_conv', f'{prefix}.proj_res_bn',
                                in_shape, in_node_id)
    res_lif_id = _add_neuron(dag, f'{prefix}.proj_res_lif', shape3, nparams, res_bn_id)

    add_id = dag.add_node(f'{prefix}.residual', 'add', False, {}, shape3, shape3)
    dag.add_edge(lif4_id, add_id)
    dag.add_edge(res_lif_id, add_id)
    return add_id, shape3


def _extract_qkformer_token_qk_attn(dag, attn, prefix, in_shape, in_node_id, nparams):
    """TokenQKAttention: Q/K Conv1d→BN→LIF, sum(Q)→attn_lif, mul(attn,K), proj."""
    B, C, H, W = in_shape
    N = H * W
    seq_shape = (B, C, N)

    # Q path
    q_id = dag.add_node(f'{prefix}.q_conv', 'linear', False,
                        {'in_features': C, 'out_features': C}, seq_shape, seq_shape)
    dag.add_edge(in_node_id, q_id)
    q_bn_id = _add_bn(dag, attn.q_bn, f'{prefix}.q_bn', seq_shape, q_id)
    q_lif_id = _add_neuron(dag, f'{prefix}.q_lif', seq_shape, nparams, q_bn_id)

    # K path
    k_id = dag.add_node(f'{prefix}.k_conv', 'linear', False,
                        {'in_features': C, 'out_features': C}, seq_shape, seq_shape)
    dag.add_edge(in_node_id, k_id)
    k_bn_id = _add_bn(dag, attn.k_bn, f'{prefix}.k_bn', seq_shape, k_id)
    k_lif_id = _add_neuron(dag, f'{prefix}.k_lif', seq_shape, nparams, k_bn_id)

    # attn_lif on summed Q, then mul with K
    attn_lif_id = _add_neuron(dag, f'{prefix}.attn_lif', seq_shape, nparams, q_lif_id)
    mul_id = dag.add_node(f'{prefix}.attn_mul', 'mul', False, {}, seq_shape, seq_shape)
    dag.add_edge(attn_lif_id, mul_id)
    dag.add_edge(k_lif_id, mul_id)

    # proj
    proj_id = dag.add_node(f'{prefix}.proj_conv', 'linear', False,
                           {'in_features': C, 'out_features': C}, seq_shape, seq_shape)
    dag.add_edge(mul_id, proj_id)
    proj_bn_id = _add_bn(dag, attn.proj_bn, f'{prefix}.proj_bn', seq_shape, proj_id)
    proj_lif_id = _add_neuron(dag, f'{prefix}.proj_lif', in_shape, nparams, proj_bn_id)
    return proj_lif_id


def _extract_qkformer_ssa(dag, attn, prefix, in_shape, in_node_id, nparams):
    """SpikingSelfAttention: Q/K/V Conv1d→BN→LIF, K^T@V, Q@result, attn_lif, proj."""
    B, C, H, W = in_shape
    N = H * W
    seq_shape = (B, C, N)

    paths = {}
    for name in ['q', 'k', 'v']:
        conv = getattr(attn, f'{name}_conv')
        bn = getattr(attn, f'{name}_bn')
        lif = getattr(attn, f'{name}_lif')
        p_id = dag.add_node(f'{prefix}.{name}_conv', 'linear', False,
                            {'in_features': C, 'out_features': C}, seq_shape, seq_shape)
        dag.add_edge(in_node_id, p_id)
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
    proj_bn_id = _add_bn(dag, attn.proj_bn, f'{prefix}.proj_bn', seq_shape, proj_id)
    proj_lif_id = _add_neuron(dag, f'{prefix}.proj_lif', in_shape, nparams, proj_bn_id)
    return proj_lif_id


def _extract_qkformer_mlp(dag, mlp, prefix, in_shape, in_node_id, nparams):
    """QKFormer MLP: fc1_conv(Conv2d k=1)→bn→lif → fc2_conv→bn→lif."""
    B, C, H, W = in_shape
    hidden = mlp.c_hidden
    hidden_shape = (B, hidden, H, W)

    c1 = mlp.fc1_conv
    p1 = _conv2d_params(c1)
    c1_id = dag.add_node(f'{prefix}.fc1_conv', 'conv2d', False, p1, in_shape, hidden_shape)
    dag.add_edge(in_node_id, c1_id)
    bn1_id = _add_bn(dag, mlp.fc1_bn, f'{prefix}.fc1_bn', hidden_shape, c1_id)
    lif1_id = _add_neuron(dag, f'{prefix}.fc1_lif', hidden_shape, nparams, bn1_id)

    c2 = mlp.fc2_conv
    p2 = _conv2d_params(c2)
    c2_id = dag.add_node(f'{prefix}.fc2_conv', 'conv2d', False, p2, hidden_shape, in_shape)
    dag.add_edge(lif1_id, c2_id)
    bn2_id = _add_bn(dag, mlp.fc2_bn, f'{prefix}.fc2_bn', in_shape, c2_id)
    lif2_id = _add_neuron(dag, f'{prefix}.fc2_lif', in_shape, nparams, bn2_id)
    return lif2_id


def _extract_qkformer_token_block(dag, blk, prefix, in_shape, in_node_id, nparams):
    """TokenSpikingBlock: x + tssa(x), x + mlp(x)."""
    attn_id = _extract_qkformer_token_qk_attn(dag, blk.tssa, f'{prefix}.tssa',
                                               in_shape, in_node_id, nparams)
    res1 = dag.add_node(f'{prefix}.attn_residual', 'add', False, {}, in_shape, in_shape)
    dag.add_edge(attn_id, res1)
    dag.add_edge(in_node_id, res1)

    mlp_id = _extract_qkformer_mlp(dag, blk.mlp, f'{prefix}.mlp',
                                   in_shape, res1, nparams)
    res2 = dag.add_node(f'{prefix}.mlp_residual', 'add', False, {}, in_shape, in_shape)
    dag.add_edge(mlp_id, res2)
    dag.add_edge(res1, res2)
    return res2


def _extract_qkformer_spiking_block(dag, blk, prefix, in_shape, in_node_id, nparams):
    """SpikingBlock: x + attn(x), x + mlp(x)."""
    attn_id = _extract_qkformer_ssa(dag, blk.attn, f'{prefix}.attn',
                                    in_shape, in_node_id, nparams)
    res1 = dag.add_node(f'{prefix}.attn_residual', 'add', False, {}, in_shape, in_shape)
    dag.add_edge(attn_id, res1)
    dag.add_edge(in_node_id, res1)

    mlp_id = _extract_qkformer_mlp(dag, blk.mlp, f'{prefix}.mlp',
                                   in_shape, res1, nparams)
    res2 = dag.add_node(f'{prefix}.mlp_residual', 'add', False, {}, in_shape, in_shape)
    dag.add_edge(mlp_id, res2)
    dag.add_edge(res1, res2)
    return res2


def extract_qkformer_dag(model, input_shape):
    """QKFormer: PatchEmbedInit → stage1 → PatchEmbedStage → stage2 → PatchEmbedStage → stage3 → head."""
    dag = OperatorDAG()
    nparams = collect_neuron_params(model)

    cur_id, cur_shape = _extract_qkformer_patch_embed_init(
        dag, model.patch_embed1, 'patch_embed1', input_shape, None, nparams)
    for i, blk in enumerate(model.stage1):
        cur_id = _extract_qkformer_token_block(dag, blk, f'stage1.{i}',
                                               cur_shape, cur_id, nparams)

    cur_id, cur_shape = _extract_qkformer_patch_embed_stage(
        dag, model.patch_embed2, 'patch_embed2', cur_shape, cur_id, nparams)
    for i, blk in enumerate(model.stage2):
        cur_id = _extract_qkformer_token_block(dag, blk, f'stage2.{i}',
                                               cur_shape, cur_id, nparams)

    cur_id, cur_shape = _extract_qkformer_patch_embed_stage(
        dag, model.patch_embed3, 'patch_embed3', cur_shape, cur_id, nparams)
    from models.qkformer import SpikingBlock
    for i, blk in enumerate(model.stage3):
        cur_id = _extract_qkformer_spiking_block(dag, blk, f'stage3.{i}',
                                                  cur_shape, cur_id, nparams)

    head = model.head
    head_in = (cur_shape[0], head.in_features)
    head_out = (cur_shape[0], head.out_features)
    head_id = dag.add_node('head', 'linear', False,
                           {'in_features': head.in_features,
                            'out_features': head.out_features},
                           head_in, head_out)
    dag.add_edge(cur_id, head_id)
    return dag


