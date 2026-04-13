"""SpikingResformer DAG extraction."""

from iengine.tdl.analysis import collect_neuron_params
from iengine.tdl.graph_ir import (
    OperatorDAG, _conv2d_params, _conv2d_out_shape, _pool_out_shape, _intify,
    _add_conv_bn, _add_neuron, _add_bn, _add_matmul,
)

def _extract_dssa(dag, dssa, prefix, in_shape, in_node_id, nparams):
    """DSSA: activation_in → W(Conv)+norm(BN) → split→matmuls→activation_attn
    → matmul → activation_out → Wproj(Conv1x1)+norm_proj(BN) + residual."""
    B, C, H, W = in_shape
    act_in_id = _add_neuron(dag, f'{prefix}.activation_in', in_shape, nparams,
                            in_node_id)

    # W conv (patch_size stride) + norm BN
    w_conv = dssa.W
    p = _conv2d_params(w_conv)
    w_out = _conv2d_out_shape(in_shape, p['out_channels'], p['kernel_size'],
                              p['stride'], p['padding'], p['dilation'])
    w_id = dag.add_node(f'{prefix}.W', 'conv2d', False, p, in_shape, w_out)
    dag.add_edge(act_in_id, w_id)
    norm_id = _add_bn(dag, dssa.norm.bn, f'{prefix}.norm.bn', w_out, w_id)

    # attn matmul: y1^T @ x
    attn_shape = in_shape  # simplified — attention output same spatial dims
    attn_mm_id = _add_matmul(dag, f'{prefix}.attn_matmul', w_out, attn_shape,
                              norm_id, act_in_id)

    act_attn_id = _add_neuron(dag, f'{prefix}.activation_attn', attn_shape,
                              nparams, attn_mm_id)

    # output matmul: y2 @ attn
    out_mm_id = _add_matmul(dag, f'{prefix}.out_matmul', attn_shape, in_shape,
                             norm_id, act_attn_id)

    act_out_id = _add_neuron(dag, f'{prefix}.activation_out', in_shape,
                             nparams, out_mm_id)

    # Projection: Conv1x1 + BN
    proj_conv = dssa.Wproj
    pp = _conv2d_params(proj_conv)
    proj_id = dag.add_node(f'{prefix}.Wproj', 'conv2d', False, pp, in_shape, in_shape)
    dag.add_edge(act_out_id, proj_id)
    proj_bn_id = _add_bn(dag, dssa.norm_proj.bn, f'{prefix}.norm_proj.bn',
                          in_shape, proj_id)

    # Residual: proj + input
    res_id = dag.add_node(f'{prefix}.residual', 'add', False, {}, in_shape, in_shape)
    dag.add_edge(proj_bn_id, res_id)
    dag.add_edge(in_node_id, res_id)
    return res_id


def _extract_gwffn(dag, gwffn, prefix, in_shape, in_node_id, nparams):
    """GWFFN: LIF→Conv1x1→BN (up), LIF→Conv3x3(groups)→BN (conv) + mid-residual,
    LIF→Conv1x1→BN (down) + outer-residual."""
    B, C, H, W = in_shape
    inner = gwffn.up[1].out_channels  # Conv1x1 output channels

    # Up: LIF → Conv1x1 → BN
    up_lif_id = _add_neuron(dag, f'{prefix}.up.0', in_shape, nparams, in_node_id)
    up_conv = gwffn.up[1]
    pp = _conv2d_params(up_conv)
    up_shape = (B, inner, H, W)
    up_conv_id = dag.add_node(f'{prefix}.up.1', 'conv2d', False, pp, in_shape, up_shape)
    dag.add_edge(up_lif_id, up_conv_id)
    up_bn_id = _add_bn(dag, gwffn.up[2].bn, f'{prefix}.up.2.bn', up_shape, up_conv_id)

    # Conv block: LIF → Conv3x3(groups) → BN, then mid-residual add
    conv_seq = gwffn.conv[0]
    conv_lif_id = _add_neuron(dag, f'{prefix}.conv.0.0', up_shape, nparams, up_bn_id)
    conv_inner = conv_seq[1]
    cp = _conv2d_params(conv_inner)
    conv_id = dag.add_node(f'{prefix}.conv.0.1', 'conv2d', False, cp, up_shape, up_shape)
    dag.add_edge(conv_lif_id, conv_id)
    conv_bn_id = _add_bn(dag, conv_seq[2].bn, f'{prefix}.conv.0.2.bn', up_shape, conv_id)

    mid_res_id = dag.add_node(f'{prefix}.conv_residual', 'add', False, {},
                              up_shape, up_shape)
    dag.add_edge(conv_bn_id, mid_res_id)
    dag.add_edge(up_bn_id, mid_res_id)

    # Down: LIF → Conv1x1 → BN
    dn_lif_id = _add_neuron(dag, f'{prefix}.down.0', up_shape, nparams, mid_res_id)
    dn_conv = gwffn.down[1]
    dp = _conv2d_params(dn_conv)
    dn_conv_id = dag.add_node(f'{prefix}.down.1', 'conv2d', False, dp,
                               up_shape, in_shape)
    dag.add_edge(dn_lif_id, dn_conv_id)
    dn_bn_id = _add_bn(dag, gwffn.down[2].bn, f'{prefix}.down.2.bn',
                         in_shape, dn_conv_id)

    # Outer residual
    out_res_id = dag.add_node(f'{prefix}.residual', 'add', False, {},
                              in_shape, in_shape)
    dag.add_edge(dn_bn_id, out_res_id)
    dag.add_edge(in_node_id, out_res_id)
    return out_res_id


def extract_spikingresformer_dag(model, input_shape):
    """SpikingResformer: prologue → stages(DownsampleLayer+DSSA/GWFFN) → avgpool → classifier."""
    dag = OperatorDAG()
    nparams = collect_neuron_params(model)

    # Prologue: _MultiStepConv2d → BN → _MultiStepMaxPool2d
    prologue = model.prologue
    conv0 = prologue[0]  # _MultiStepConv2d (subclass of Conv2d)
    p0 = _conv2d_params(conv0)
    shape = _conv2d_out_shape(input_shape, p0['out_channels'], p0['kernel_size'],
                              p0['stride'], p0['padding'], p0['dilation'])
    conv0_id = dag.add_node('prologue.0', 'conv2d', False, p0, input_shape, shape)
    bn0_id = _add_bn(dag, prologue[1].bn, 'prologue.1.bn', shape, conv0_id)

    pool0 = prologue[2]  # _MultiStepMaxPool2d
    ks = _intify(pool0.kernel_size)
    st = _intify(pool0.stride)
    pa = _intify(pool0.padding)
    pool_shape = _pool_out_shape(shape, ks, st, pa)
    pool0_id = dag.add_node('prologue.2', 'maxpool2d', False,
                            {'kernel_size': ks, 'stride': st, 'padding': pa},
                            shape, pool_shape)
    dag.add_edge(bn0_id, pool0_id)

    cur_id, cur_shape = pool0_id, pool_shape

    # Stages
    from models.spikingresformer import DSSA, GWFFN, DownsampleLayer
    for si, stage in enumerate(model.layers):
        for li, layer in enumerate(stage):
            lprefix = f'layers.{si}.{li}'
            if isinstance(layer, DownsampleLayer):
                ds_lif = _add_neuron(dag, f'{lprefix}.activation', cur_shape,
                                     nparams, cur_id)
                ds_conv = layer.conv
                dp = _conv2d_params(ds_conv)
                ds_out = _conv2d_out_shape(cur_shape, dp['out_channels'],
                                           dp['kernel_size'], dp['stride'],
                                           dp['padding'], dp['dilation'])
                ds_conv_id = dag.add_node(f'{lprefix}.conv', 'conv2d', False,
                                          dp, cur_shape, ds_out)
                dag.add_edge(ds_lif, ds_conv_id)
                ds_bn_id = _add_bn(dag, layer.norm.bn, f'{lprefix}.norm.bn',
                                    ds_out, ds_conv_id)
                cur_id, cur_shape = ds_bn_id, ds_out
            elif isinstance(layer, DSSA):
                cur_id = _extract_dssa(dag, layer, lprefix, cur_shape,
                                       cur_id, nparams)
            elif isinstance(layer, GWFFN):
                cur_id = _extract_gwffn(dag, layer, lprefix, cur_shape,
                                        cur_id, nparams)

    # avgpool → classifier
    avg_shape = (cur_shape[0], cur_shape[1], 1, 1)
    avg_id = dag.add_node('avgpool', 'avgpool', False,
                          {'output_size': 1}, cur_shape, avg_shape)
    dag.add_edge(cur_id, avg_id)

    clf = model.classifier
    fc_in = (avg_shape[0], clf.in_features)
    fc_out = (avg_shape[0], clf.out_features)
    fc_id = dag.add_node('classifier', 'linear', False,
                         {'in_features': clf.in_features,
                          'out_features': clf.out_features},
                         fc_in, fc_out)
    dag.add_edge(avg_id, fc_id)
    return dag
