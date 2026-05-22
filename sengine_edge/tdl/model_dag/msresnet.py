"""MS-ResNet DAG extraction (MSResNet18, MSResNet104, MSResNetCifar)."""

import torch.nn as nn

from sengine_edge.tdl.analysis import collect_neuron_params
from sengine_edge.tdl.graph_ir import (
    OperatorDAG, _conv2d_params, _conv2d_out_shape, _intify,
    _add_conv_bn, _add_neuron, _add_residual,
)

def _extract_ms_basic_block18(dag, block, prefix, in_shape, in_node_id,
                              neuron_params):
    """MS-ResNet BasicBlock18: sn1→conv_bn1→sn2→conv_bn2→[add]

    Pre-activation pattern: neuron comes before conv.
    """
    sn1_id = _add_neuron(dag, f"{prefix}.sn1", in_shape, neuron_params,
                         in_node_id)

    bn1_id, shape = _add_conv_bn(
        dag, block.conv_bn1.module[0], block.conv_bn1.module[1],
        f"{prefix}.conv_bn1.module.0", f"{prefix}.conv_bn1.module.1",
        in_shape, sn1_id)

    sn2_id = _add_neuron(dag, f"{prefix}.sn2", shape, neuron_params, bn1_id)

    bn2_id, out_shape = _add_conv_bn(
        dag, block.conv_bn2.module[0], block.conv_bn2.module[1],
        f"{prefix}.conv_bn2.module.0", f"{prefix}.conv_bn2.module.1",
        shape, sn2_id)

    # Shortcut
    if isinstance(block.shortcut, nn.Sequential) and len(block.shortcut) == 0:
        shortcut_id = in_node_id
    else:
        # TDBNContainer shortcut
        from models.msresnet import TDBNContainer
        if isinstance(block.shortcut, TDBNContainer):
            sc_bn_id, _ = _add_conv_bn(
                dag, block.shortcut.module[0], block.shortcut.module[1],
                f"{prefix}.shortcut.module.0", f"{prefix}.shortcut.module.1",
                in_shape, in_node_id)
            shortcut_id = sc_bn_id
        else:
            shortcut_id = in_node_id

    res_id = _add_residual(dag, f"{prefix}.residual", 'ADD',
                           out_shape, bn2_id, shortcut_id)
    return res_id, out_shape


def _extract_ms_bottleneck(dag, block, prefix, in_shape, in_node_id,
                           neuron_params):
    """MS-ResNet BottleneckBlock: sn1→conv_bn1→sn2→conv_bn2→sn3→conv_bn3→[add]"""
    sn1_id = _add_neuron(dag, f"{prefix}.sn1", in_shape, neuron_params,
                         in_node_id)

    bn1_id, shape = _add_conv_bn(
        dag, block.conv_bn1.module[0], block.conv_bn1.module[1],
        f"{prefix}.conv_bn1.module.0", f"{prefix}.conv_bn1.module.1",
        in_shape, sn1_id)

    sn2_id = _add_neuron(dag, f"{prefix}.sn2", shape, neuron_params, bn1_id)

    bn2_id, shape = _add_conv_bn(
        dag, block.conv_bn2.module[0], block.conv_bn2.module[1],
        f"{prefix}.conv_bn2.module.0", f"{prefix}.conv_bn2.module.1",
        shape, sn2_id)

    sn3_id = _add_neuron(dag, f"{prefix}.sn3", shape, neuron_params, bn2_id)

    bn3_id, out_shape = _add_conv_bn(
        dag, block.conv_bn3.module[0], block.conv_bn3.module[1],
        f"{prefix}.conv_bn3.module.0", f"{prefix}.conv_bn3.module.1",
        shape, sn3_id)

    # Shortcut
    from models.msresnet import TDBNContainer
    if isinstance(block.shortcut, TDBNContainer):
        sc_bn_id, _ = _add_conv_bn(
            dag, block.shortcut.module[0], block.shortcut.module[1],
            f"{prefix}.shortcut.module.0", f"{prefix}.shortcut.module.1",
            in_shape, in_node_id)
        shortcut_id = sc_bn_id
    else:
        shortcut_id = in_node_id

    res_id = _add_residual(dag, f"{prefix}.residual", 'ADD',
                           out_shape, bn3_id, shortcut_id)
    return res_id, out_shape


def _extract_ms_basic_block_cifar(dag, block, prefix, in_shape, in_node_id,
                                  neuron_params):
    """MS-ResNet BasicBlockCifar/BasicBlock104: same pre-activation pattern,
    shortcut may use AvgPool3d (spatial downsampling) + TDBNContainer."""
    sn1_id = _add_neuron(dag, f"{prefix}.sn1", in_shape, neuron_params,
                         in_node_id)

    bn1_id, shape = _add_conv_bn(
        dag, block.conv_bn1.module[0], block.conv_bn1.module[1],
        f"{prefix}.conv_bn1.module.0", f"{prefix}.conv_bn1.module.1",
        in_shape, sn1_id)

    sn2_id = _add_neuron(dag, f"{prefix}.sn2", shape, neuron_params, bn1_id)

    bn2_id, out_shape = _add_conv_bn(
        dag, block.conv_bn2.module[0], block.conv_bn2.module[1],
        f"{prefix}.conv_bn2.module.0", f"{prefix}.conv_bn2.module.1",
        shape, sn2_id)

    # Shortcut: either identity, or AvgPool3d + TDBNContainer
    if isinstance(block.shortcut, nn.Sequential) and len(block.shortcut) > 0:
        # AvgPool3d(1, stride, stride) → per-timestep equivalent to spatial downsampling
        pool3d = block.shortcut[0]  # nn.AvgPool3d
        spatial_stride = _intify(pool3d.stride[1:]) if isinstance(pool3d.stride, tuple) else pool3d.stride
        pool_shape = (in_shape[0], in_shape[1],
                      in_shape[2] // spatial_stride,
                      in_shape[3] // spatial_stride)
        pool_id = dag.add_node(f"{prefix}.shortcut.0", 'avgpool', False,
                               {'stride': spatial_stride}, in_shape, pool_shape)
        dag.add_edge(in_node_id, pool_id)

        tdbn = block.shortcut[1]  # TDBNContainer
        sc_bn_id, _ = _add_conv_bn(
            dag, tdbn.module[0], tdbn.module[1],
            f"{prefix}.shortcut.1.module.0", f"{prefix}.shortcut.1.module.1",
            pool_shape, pool_id)
        shortcut_id = sc_bn_id
    else:
        shortcut_id = in_node_id

    res_id = _add_residual(dag, f"{prefix}.residual", 'ADD',
                           out_shape, bn2_id, shortcut_id)
    return res_id, out_shape


def _extract_ms_layer(dag, layer, prefix, in_shape, in_node_id,
                      neuron_params):
    """Extract a nn.Sequential of MS-ResNet blocks."""
    from models.msresnet import (BasicBlock18, BottleneckBlock,
                                 BasicBlock104, BasicBlockCifar)

    cur_id, cur_shape = in_node_id, in_shape
    for i, block in enumerate(layer):
        block_prefix = f"{prefix}.{i}"
        if isinstance(block, BottleneckBlock):
            cur_id, cur_shape = _extract_ms_bottleneck(
                dag, block, block_prefix, cur_shape, cur_id, neuron_params)
        elif isinstance(block, (BasicBlock104, BasicBlockCifar)):
            cur_id, cur_shape = _extract_ms_basic_block_cifar(
                dag, block, block_prefix, cur_shape, cur_id, neuron_params)
        elif isinstance(block, BasicBlock18):
            cur_id, cur_shape = _extract_ms_basic_block18(
                dag, block, block_prefix, cur_shape, cur_id, neuron_params)
        else:
            raise ValueError(f"Unknown MS-ResNet block type: {type(block)}")
    return cur_id, cur_shape


def extract_msresnet18_dag(model, input_shape: tuple) -> OperatorDAG:
    """Extract DAG from MSResNet18 (conv1→conv2_x..conv5_x→sn_out→fc)."""
    dag = OperatorDAG()
    nparams = collect_neuron_params(model)

    # Stem: TDBNContainer
    bn_id, shape = _add_conv_bn(
        dag, model.conv1.module[0], model.conv1.module[1],
        'conv1.module.0', 'conv1.module.1', input_shape)

    # 4 conv stages
    cur_id, cur_shape = bn_id, shape
    for name in ['conv2_x', 'conv3_x', 'conv4_x', 'conv5_x']:
        cur_id, cur_shape = _extract_ms_layer(
            dag, getattr(model, name), name, cur_shape, cur_id, nparams)

    # sn_out neuron → temporal mean → avgpool → fc
    sn_out_id = _add_neuron(dag, 'sn_out', cur_shape, nparams, cur_id)

    avg_shape = (cur_shape[0], cur_shape[1], 1, 1)
    avg_id = dag.add_node('avgpool', 'avgpool', False,
                          {'output_size': 1}, cur_shape, avg_shape)
    dag.add_edge(sn_out_id, avg_id)

    fc = model.fc
    fc_in = (avg_shape[0], fc.in_features)
    fc_out = (avg_shape[0], fc.out_features)
    fc_id = dag.add_node('fc', 'linear', False,
                         {'in_features': fc.in_features,
                          'out_features': fc.out_features},
                         fc_in, fc_out)
    dag.add_edge(avg_id, fc_id)
    return dag


def extract_msresnet_cifar_dag(model, input_shape: tuple) -> OperatorDAG:
    """Extract DAG from MSResNetCifar (conv1→layer1..3→sn_out→fc)."""
    dag = OperatorDAG()
    nparams = collect_neuron_params(model)

    # Stem: TDBNContainer
    bn_id, shape = _add_conv_bn(
        dag, model.conv1.module[0], model.conv1.module[1],
        'conv1.module.0', 'conv1.module.1', input_shape)

    # 3 layers
    cur_id, cur_shape = bn_id, shape
    for name in ['layer1', 'layer2', 'layer3']:
        cur_id, cur_shape = _extract_ms_layer(
            dag, getattr(model, name), name, cur_shape, cur_id, nparams)

    # sn_out
    sn_out_id = _add_neuron(dag, 'sn_out', cur_shape, nparams, cur_id)

    avg_shape = (cur_shape[0], cur_shape[1], 1, 1)
    avg_id = dag.add_node('avgpool', 'avgpool', False,
                          {'output_size': 1}, cur_shape, avg_shape)
    dag.add_edge(sn_out_id, avg_id)

    fc = model.fc
    fc_in = (avg_shape[0], fc.in_features)
    fc_out = (avg_shape[0], fc.out_features)
    fc_id = dag.add_node('fc', 'linear', False,
                         {'in_features': fc.in_features,
                          'out_features': fc.out_features},
                         fc_in, fc_out)
    dag.add_edge(avg_id, fc_id)
    return dag


def extract_msresnet104_dag(model, input_shape: tuple) -> OperatorDAG:
    """Extract DAG from MSResNet104 (3-conv stem→conv2_x..5_x→sn_out→fc)."""
    dag = OperatorDAG()
    nparams = collect_neuron_params(model)

    # 3-conv stem: conv1.module = Sequential(Conv, Conv, Conv, BN3d)
    stem = model.conv1.module
    shape = input_shape
    prev_id = None
    for i in range(3):
        conv = stem[i]
        p = _conv2d_params(conv)
        out = _conv2d_out_shape(shape, p['out_channels'], p['kernel_size'],
                                p['stride'], p['padding'], p['dilation'])
        conv_id = dag.add_node(f'conv1.module.{i}', 'conv2d', False, p,
                               shape, out)
        if prev_id is not None:
            dag.add_edge(prev_id, conv_id)
        prev_id = conv_id
        shape = out

    # BN3d at stem[3]
    bn3d = stem[3]
    bn_id = dag.add_node('conv1.module.3', 'bn2d', False,
                         {'num_features': bn3d.num_features}, shape, shape)
    dag.add_edge(prev_id, bn_id)

    # 4 conv stages
    cur_id, cur_shape = bn_id, shape
    for name in ['conv2_x', 'conv3_x', 'conv4_x', 'conv5_x']:
        cur_id, cur_shape = _extract_ms_layer(
            dag, getattr(model, name), name, cur_shape, cur_id, nparams)

    # sn_out → avgpool → fc
    sn_out_id = _add_neuron(dag, 'sn_out', cur_shape, nparams, cur_id)

    avg_shape = (cur_shape[0], cur_shape[1], 1, 1)
    avg_id = dag.add_node('avgpool', 'avgpool', False,
                          {'output_size': 1}, cur_shape, avg_shape)
    dag.add_edge(sn_out_id, avg_id)

    fc = model.fc
    fc_in = (avg_shape[0], fc.in_features)
    fc_out = (avg_shape[0], fc.out_features)
    fc_id = dag.add_node('fc', 'linear', False,
                         {'in_features': fc.in_features,
                          'out_features': fc.out_features},
                         fc_in, fc_out)
    dag.add_edge(avg_id, fc_id)
    return dag
