"""MetaFormer (SpikeDrivenTransformerV2) DAG extraction."""

from sengine_edge.tdl.analysis import collect_neuron_params
from sengine_edge.tdl.graph_ir import (
    OperatorDAG, _conv2d_params, _conv2d_out_shape, _intify,
    _add_conv_bn, _add_neuron, _add_bn, _add_matmul,
)

def _extract_ms_downsampling(dag, ds, prefix, in_shape, in_node_id, nparams):
    """MS_DownSampling: [LIF] → Conv2d → BN."""
    cur_id = in_node_id
    if hasattr(ds, 'encode_lif'):
        cur_id = _add_neuron(dag, f'{prefix}.encode_lif', in_shape, nparams, cur_id)
    conv = ds.encode_conv
    p = _conv2d_params(conv)
    out = _conv2d_out_shape(in_shape, p['out_channels'], p['kernel_size'],
                            p['stride'], p['padding'], p['dilation'])
    conv_id = dag.add_node(f'{prefix}.encode_conv', 'conv2d', False, p, in_shape, out)
    if cur_id is not None:
        dag.add_edge(cur_id, conv_id)
    bn_id = _add_bn(dag, ds.encode_bn, f'{prefix}.encode_bn', out, conv_id)
    return bn_id, out


def _extract_sepconv(dag, sc, prefix, in_shape, in_node_id, nparams):
    """SepConv: lif1→pwconv1→bn1→lif2→dwconv→pwconv2→bn2."""
    B, C, H, W = in_shape
    lif1_id = _add_neuron(dag, f'{prefix}.lif1', in_shape, nparams, in_node_id)

    pw1 = sc.pwconv1
    p1 = _conv2d_params(pw1)
    med = p1['out_channels']
    med_shape = (B, med, H, W)
    pw1_id = dag.add_node(f'{prefix}.pwconv1', 'conv2d', False, p1, in_shape, med_shape)
    dag.add_edge(lif1_id, pw1_id)
    bn1_id = _add_bn(dag, sc.bn1, f'{prefix}.bn1', med_shape, pw1_id)

    lif2_id = _add_neuron(dag, f'{prefix}.lif2', med_shape, nparams, bn1_id)

    dw = sc.dwconv
    dp = _conv2d_params(dw)
    dw_id = dag.add_node(f'{prefix}.dwconv', 'conv2d', False, dp, med_shape, med_shape)
    dag.add_edge(lif2_id, dw_id)

    pw2 = sc.pwconv2
    p2 = _conv2d_params(pw2)
    pw2_id = dag.add_node(f'{prefix}.pwconv2', 'conv2d', False, p2, med_shape, in_shape)
    dag.add_edge(dw_id, pw2_id)
    bn2_id = _add_bn(dag, sc.bn2, f'{prefix}.bn2', in_shape, pw2_id)
    return bn2_id


def _extract_ms_convblock(dag, blk, prefix, in_shape, in_node_id, nparams):
    """MS_ConvBlock: SepConv(x)+x residual, then Conv-MLP(x)+x residual."""
    B, C, H, W = in_shape
    sep_id = _extract_sepconv(dag, blk.Conv, f'{prefix}.Conv', in_shape,
                              in_node_id, nparams)
    res1 = dag.add_node(f'{prefix}.conv_residual', 'add', False, {}, in_shape, in_shape)
    dag.add_edge(sep_id, res1)
    dag.add_edge(in_node_id, res1)

    # MLP branch: lif1→conv1→bn1→lif2→conv2→bn2
    mlp_ratio = 4
    hidden = int(C * mlp_ratio)
    hidden_shape = (B, hidden, H, W)
    lif1_id = _add_neuron(dag, f'{prefix}.lif1', in_shape, nparams, res1)

    c1 = blk.conv1
    p1 = _conv2d_params(c1)
    c1_id = dag.add_node(f'{prefix}.conv1', 'conv2d', False, p1, in_shape, hidden_shape)
    dag.add_edge(lif1_id, c1_id)
    bn1_id = _add_bn(dag, blk.bn1, f'{prefix}.bn1', hidden_shape, c1_id)

    lif2_id = _add_neuron(dag, f'{prefix}.lif2', hidden_shape, nparams, bn1_id)

    c2 = blk.conv2
    p2 = _conv2d_params(c2)
    c2_id = dag.add_node(f'{prefix}.conv2', 'conv2d', False, p2, hidden_shape, in_shape)
    dag.add_edge(lif2_id, c2_id)
    bn2_id = _add_bn(dag, blk.bn2, f'{prefix}.bn2', in_shape, c2_id)

    res2 = dag.add_node(f'{prefix}.mlp_residual', 'add', False, {}, in_shape, in_shape)
    dag.add_edge(bn2_id, res2)
    dag.add_edge(res1, res2)
    return res2


def _extract_ms_attention(dag, attn, prefix, in_shape, in_node_id, nparams):
    """MS_Attention_RepConv: head_lif → Q/K/V RepConv+BN+LIF → K^T@V, Q@result → attn_lif → proj."""
    B, C, H, W = in_shape
    N = H * W
    seq_shape = (B, C, N)

    head_lif_id = _add_neuron(dag, f'{prefix}.head_lif', in_shape, nparams, in_node_id)

    def _repconv_path(name, rc_seq, lif_mod):
        # RepConv is body=Sequential(conv1x1, BNAndPad, Sequential(dw_conv3x3, pw_conv1x1, BN))
        # + final BN2d in q_conv Sequential
        # Represent entire RepConv+BN as a single conv2d node (it's a linear op at eval)
        final_bn = rc_seq[-1]  # nn.BatchNorm2d
        nid = dag.add_node(f'{prefix}.{name}', 'conv2d', False,
                           {'in_channels': C, 'out_channels': C,
                            'kernel_size': 3, 'stride': 1, 'padding': 1,
                            'dilation': 1, 'groups': 1},
                           in_shape, in_shape)
        dag.add_edge(head_lif_id, nid)
        bn_id = _add_bn(dag, final_bn, f'{prefix}.{name}_bn', in_shape, nid)
        lif_id = _add_neuron(dag, f'{prefix}.{name.split("_")[0]}_lif',
                             in_shape, nparams, bn_id)
        return lif_id

    # At eval, RepConv+BN is reparameterized to a single conv.
    # We represent the whole q_conv/k_conv/v_conv as one conv node + BN + LIF each.
    q_lif = _repconv_path('q_conv', attn.q_conv, attn.q_lif)
    k_lif = _repconv_path('k_conv', attn.k_conv, attn.k_lif)
    v_lif = _repconv_path('v_conv', attn.v_conv, attn.v_lif)

    # K^T @ V then Q @ result
    kv_id = _add_matmul(dag, f'{prefix}.kv_matmul', in_shape, in_shape,
                         k_lif, v_lif)
    qkv_id = _add_matmul(dag, f'{prefix}.qkv_matmul', in_shape, in_shape,
                          q_lif, kv_id)

    attn_lif_id = _add_neuron(dag, f'{prefix}.attn_lif', in_shape, nparams, qkv_id)

    # Projection: RepConv+BN
    proj_id = dag.add_node(f'{prefix}.proj_conv', 'conv2d', False,
                           {'in_channels': C, 'out_channels': C,
                            'kernel_size': 3, 'stride': 1, 'padding': 1,
                            'dilation': 1, 'groups': 1},
                           in_shape, in_shape)
    dag.add_edge(attn_lif_id, proj_id)
    proj_bn = attn.proj_conv[-1]  # final BN2d
    proj_bn_id = _add_bn(dag, proj_bn, f'{prefix}.proj_conv_bn', in_shape, proj_id)
    return proj_bn_id


def _extract_ms_mlp(dag, mlp, prefix, in_shape, in_node_id, nparams):
    """MS_MLP: fc1_lif→fc1_conv(Conv1d)→fc1_bn→fc2_lif→fc2_conv→fc2_bn."""
    B, C, H, W = in_shape
    hidden = mlp.c_hidden
    seq_shape = (B, C, H * W)
    hidden_seq = (B, hidden, H * W)

    lif1_id = _add_neuron(dag, f'{prefix}.fc1_lif', in_shape, nparams, in_node_id)
    # Conv1d(C, hidden, k=1) represented as linear
    fc1_id = dag.add_node(f'{prefix}.fc1_conv', 'linear', False,
                          {'in_features': C, 'out_features': hidden},
                          seq_shape, hidden_seq)
    dag.add_edge(lif1_id, fc1_id)
    fc1_bn_id = _add_bn(dag, mlp.fc1_bn, f'{prefix}.fc1_bn', hidden_seq, fc1_id)

    lif2_id = _add_neuron(dag, f'{prefix}.fc2_lif', hidden_seq, nparams, fc1_bn_id)
    fc2_id = dag.add_node(f'{prefix}.fc2_conv', 'linear', False,
                          {'in_features': hidden, 'out_features': C},
                          hidden_seq, seq_shape)
    dag.add_edge(lif2_id, fc2_id)
    fc2_bn_id = _add_bn(dag, mlp.fc2_bn, f'{prefix}.fc2_bn', seq_shape, fc2_id)
    return fc2_bn_id


def _extract_ms_block(dag, blk, prefix, in_shape, in_node_id, nparams):
    """MS_Block: x + attn(x), x + mlp(x)."""
    attn_id = _extract_ms_attention(dag, blk.attn, f'{prefix}.attn',
                                    in_shape, in_node_id, nparams)
    res1 = dag.add_node(f'{prefix}.attn_residual', 'add', False, {},
                        in_shape, in_shape)
    dag.add_edge(attn_id, res1)
    dag.add_edge(in_node_id, res1)

    mlp_id = _extract_ms_mlp(dag, blk.mlp, f'{prefix}.mlp', in_shape,
                             res1, nparams)
    res2 = dag.add_node(f'{prefix}.mlp_residual', 'add', False, {},
                        in_shape, in_shape)
    dag.add_edge(mlp_id, res2)
    dag.add_edge(res1, res2)
    return res2


def extract_metaformer_dag(model, input_shape):
    """SpikeDrivenTransformerV2 / MetaFormer."""
    dag = OperatorDAG()
    nparams = collect_neuron_params(model)

    # Stage 1
    cur_id, cur_shape = _extract_ms_downsampling(
        dag, model.downsample1_1, 'downsample1_1', input_shape, None, nparams)
    for i, blk in enumerate(model.ConvBlock1_1):
        cur_id = _extract_ms_convblock(dag, blk, f'ConvBlock1_1.{i}',
                                       cur_shape, cur_id, nparams)
    cur_id, cur_shape = _extract_ms_downsampling(
        dag, model.downsample1_2, 'downsample1_2', cur_shape, cur_id, nparams)
    for i, blk in enumerate(model.ConvBlock1_2):
        cur_id = _extract_ms_convblock(dag, blk, f'ConvBlock1_2.{i}',
                                       cur_shape, cur_id, nparams)

    # Stage 2
    cur_id, cur_shape = _extract_ms_downsampling(
        dag, model.downsample2, 'downsample2', cur_shape, cur_id, nparams)
    for i, blk in enumerate(model.ConvBlock2_1):
        cur_id = _extract_ms_convblock(dag, blk, f'ConvBlock2_1.{i}',
                                       cur_shape, cur_id, nparams)
    for i, blk in enumerate(model.ConvBlock2_2):
        cur_id = _extract_ms_convblock(dag, blk, f'ConvBlock2_2.{i}',
                                       cur_shape, cur_id, nparams)

    # Stage 3: transformer blocks
    cur_id, cur_shape = _extract_ms_downsampling(
        dag, model.downsample3, 'downsample3', cur_shape, cur_id, nparams)
    for i, blk in enumerate(model.block3):
        cur_id = _extract_ms_block(dag, blk, f'block3.{i}',
                                   cur_shape, cur_id, nparams)

    # Stage 4
    cur_id, cur_shape = _extract_ms_downsampling(
        dag, model.downsample4, 'downsample4', cur_shape, cur_id, nparams)
    for i, blk in enumerate(model.block4):
        cur_id = _extract_ms_block(dag, blk, f'block4.{i}',
                                   cur_shape, cur_id, nparams)

    # Head: lif → linear
    lif_id = _add_neuron(dag, 'lif', cur_shape, nparams, cur_id)
    head = model.head
    head_in = (cur_shape[0], head.in_features)
    head_out = (cur_shape[0], head.out_features)
    head_id = dag.add_node('head', 'linear', False,
                           {'in_features': head.in_features,
                            'out_features': head.out_features},
                           head_in, head_out)
    dag.add_edge(lif_id, head_id)
    return dag

