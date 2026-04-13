"""Spikformer DAG extraction."""

from iengine.tdl.analysis import collect_neuron_params
from iengine.tdl.graph_ir import (
    OperatorDAG, _add_conv_bn, _add_neuron, _add_bn, _add_linear, _add_matmul, _add_pool,
)

def _extract_spikformer_sps(dag, sps, prefix, in_shape, in_node_id, nparams):
    """SPS patch embedding:
    conv→bn→lif → conv1→bn1→lif1 → conv2→bn2→lif2→maxpool2
    → conv3→bn3→lif3→maxpool3 → [rpe_conv→rpe_bn→rpe_lif + residual add]
    """
    B, C, H, W = in_shape
    # Stage 0: conv → bn → lif
    bn_id, shape = _add_conv_bn(dag, sps.proj_conv, sps.proj_bn,
                                f'{prefix}.proj_conv', f'{prefix}.proj_bn',
                                in_shape, in_node_id)
    lif_id = _add_neuron(dag, f'{prefix}.proj_lif', shape, nparams, bn_id)

    # Stage 1: conv1 → bn1 → lif1
    bn1_id, shape = _add_conv_bn(dag, sps.proj_conv1, sps.proj_bn1,
                                 f'{prefix}.proj_conv1', f'{prefix}.proj_bn1',
                                 shape, lif_id)
    lif1_id = _add_neuron(dag, f'{prefix}.proj_lif1', shape, nparams, bn1_id)

    # Stage 2: conv2 → bn2 → lif2 → maxpool2
    bn2_id, shape = _add_conv_bn(dag, sps.proj_conv2, sps.proj_bn2,
                                 f'{prefix}.proj_conv2', f'{prefix}.proj_bn2',
                                 shape, lif1_id)
    lif2_id = _add_neuron(dag, f'{prefix}.proj_lif2', shape, nparams, bn2_id)
    pool2_id, shape = _add_pool(dag, sps.maxpool2, f'{prefix}.maxpool2',
                                shape, lif2_id)

    # Stage 3: conv3 → bn3 → lif3 → maxpool3
    bn3_id, shape = _add_conv_bn(dag, sps.proj_conv3, sps.proj_bn3,
                                 f'{prefix}.proj_conv3', f'{prefix}.proj_bn3',
                                 shape, pool2_id)
    lif3_id = _add_neuron(dag, f'{prefix}.proj_lif3', shape, nparams, bn3_id)
    pool3_id, shape = _add_pool(dag, sps.maxpool3, f'{prefix}.maxpool3',
                                shape, lif3_id)

    # RPE residual: rpe_conv → rpe_bn → rpe_lif, then add with pool3 output
    rpe_bn_id, _ = _add_conv_bn(dag, sps.rpe_conv, sps.rpe_bn,
                                f'{prefix}.rpe_conv', f'{prefix}.rpe_bn',
                                shape, pool3_id)
    rpe_lif_id = _add_neuron(dag, f'{prefix}.rpe_lif', shape, nparams, rpe_bn_id)
    add_id = dag.add_node(f'{prefix}.rpe_residual', 'add', False, {}, shape, shape)
    dag.add_edge(rpe_lif_id, add_id)
    dag.add_edge(pool3_id, add_id)

    # Output shape for transformer: (B, N, C) where N = H*W after pooling
    N = shape[2] * shape[3]
    seq_shape = (shape[0], N, shape[1])
    return add_id, shape, seq_shape


def _extract_spikformer_ssa(dag, ssa, prefix, seq_shape, spatial_shape,
                            in_node_id, nparams):
    """SSA: Q/K/V each = Linear→BN1d→LIF, then Q@K^T→attn@V→attn_lif→proj.
    seq_shape = (B, N, C), spatial_shape = (B, C, H, W)."""
    B, N, C = seq_shape
    head_dim = C // ssa.num_heads

    # Q path: linear → bn → lif
    q_lin_id, _ = _add_linear(dag, ssa.q_linear, f'{prefix}.q_linear',
                              seq_shape, in_node_id)
    q_bn_id = _add_bn(dag, ssa.q_bn, f'{prefix}.q_bn', seq_shape, q_lin_id)
    q_lif_id = _add_neuron(dag, f'{prefix}.q_lif', seq_shape, nparams, q_bn_id)

    # K path
    k_lin_id, _ = _add_linear(dag, ssa.k_linear, f'{prefix}.k_linear',
                              seq_shape, in_node_id)
    k_bn_id = _add_bn(dag, ssa.k_bn, f'{prefix}.k_bn', seq_shape, k_lin_id)
    k_lif_id = _add_neuron(dag, f'{prefix}.k_lif', seq_shape, nparams, k_bn_id)

    # V path
    v_lin_id, _ = _add_linear(dag, ssa.v_linear, f'{prefix}.v_linear',
                              seq_shape, in_node_id)
    v_bn_id = _add_bn(dag, ssa.v_bn, f'{prefix}.v_bn', seq_shape, v_lin_id)
    v_lif_id = _add_neuron(dag, f'{prefix}.v_lif', seq_shape, nparams, v_bn_id)

    # Q @ K^T → attn @ V (two matmuls)
    attn_shape = (B, ssa.num_heads, N, N)
    qk_id = _add_matmul(dag, f'{prefix}.qk_matmul', seq_shape, attn_shape,
                         q_lif_id, k_lif_id)
    out_shape = seq_shape
    av_id = _add_matmul(dag, f'{prefix}.av_matmul', attn_shape, out_shape,
                         qk_id, v_lif_id)

    # attn_lif → proj_linear → proj_bn → proj_lif
    attn_lif_id = _add_neuron(dag, f'{prefix}.attn_lif', seq_shape, nparams, av_id)
    proj_lin_id, _ = _add_linear(dag, ssa.proj_linear, f'{prefix}.proj_linear',
                                 seq_shape, attn_lif_id)
    proj_bn_id = _add_bn(dag, ssa.proj_bn, f'{prefix}.proj_bn', seq_shape, proj_lin_id)
    proj_lif_id = _add_neuron(dag, f'{prefix}.proj_lif', seq_shape, nparams, proj_bn_id)
    return proj_lif_id


def _extract_spikformer_mlp(dag, mlp, prefix, seq_shape, in_node_id, nparams):
    """MLP: fc1_linear→fc1_bn→fc1_lif → fc2_linear→fc2_bn→fc2_lif."""
    B, N, C = seq_shape
    hidden = mlp.c_hidden
    hidden_shape = (B, N, hidden)

    fc1_id, _ = _add_linear(dag, mlp.fc1_linear, f'{prefix}.fc1_linear',
                            seq_shape, in_node_id)
    fc1_bn_id = _add_bn(dag, mlp.fc1_bn, f'{prefix}.fc1_bn', hidden_shape, fc1_id)
    fc1_lif_id = _add_neuron(dag, f'{prefix}.fc1_lif', hidden_shape, nparams, fc1_bn_id)

    fc2_id, _ = _add_linear(dag, mlp.fc2_linear, f'{prefix}.fc2_linear',
                            hidden_shape, fc1_lif_id)
    fc2_bn_id = _add_bn(dag, mlp.fc2_bn, f'{prefix}.fc2_bn', seq_shape, fc2_id)
    fc2_lif_id = _add_neuron(dag, f'{prefix}.fc2_lif', seq_shape, nparams, fc2_bn_id)
    return fc2_lif_id


def _extract_spikformer_block(dag, block, prefix, seq_shape, spatial_shape,
                              in_node_id, nparams):
    """Block: x + SSA(x), then x + MLP(x). Two residual adds."""
    attn_id = _extract_spikformer_ssa(dag, block.attn, f'{prefix}.attn',
                                      seq_shape, spatial_shape, in_node_id, nparams)
    add1 = dag.add_node(f'{prefix}.attn_residual', 'add', False, {},
                        seq_shape, seq_shape)
    dag.add_edge(attn_id, add1)
    dag.add_edge(in_node_id, add1)

    mlp_id = _extract_spikformer_mlp(dag, block.mlp, f'{prefix}.mlp',
                                     seq_shape, add1, nparams)
    add2 = dag.add_node(f'{prefix}.mlp_residual', 'add', False, {},
                        seq_shape, seq_shape)
    dag.add_edge(mlp_id, add2)
    dag.add_edge(add1, add2)
    return add2


def extract_spikformer_dag(model, input_shape):
    """Spikformer: SPS → N blocks → head."""
    dag = OperatorDAG()
    nparams = collect_neuron_params(model)

    sps_id, spatial, seq = _extract_spikformer_sps(
        dag, model.patch_embed, 'patch_embed', input_shape, None, nparams)

    cur_id = sps_id
    for i, blk in enumerate(model.block):
        cur_id = _extract_spikformer_block(dag, blk, f'block.{i}',
                                           seq, spatial, cur_id, nparams)

    # head: Linear(embed_dims, num_classes) — applied after mean over N then T
    head = model.head
    head_in = (seq[0], seq[2])  # (B, C)
    head_out = (seq[0], head.out_features)
    head_id = dag.add_node('head', 'linear', False,
                           {'in_features': head.in_features,
                            'out_features': head.out_features},
                           head_in, head_out)
    dag.add_edge(cur_id, head_id)
    return dag
