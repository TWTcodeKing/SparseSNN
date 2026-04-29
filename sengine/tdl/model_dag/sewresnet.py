"""SEW-ResNet DAG extraction (ImageNet + CIFAR variants)."""

from sengine.tdl.analysis import collect_neuron_params
from sengine.tdl.graph_ir import (
    OperatorDAG, _conv2d_params, _conv2d_out_shape, _pool_out_shape, _intify,
    _add_conv_bn, _add_neuron, _add_residual,
)


def _extract_basic_block(dag, block, prefix, in_shape, in_node_id, nparams):
    """BasicBlock: conv1→sn1→conv2→sn2→[residual]"""
    bn1_id, shape = _add_conv_bn(
        dag, block.conv1.module[0], block.conv1.module[1],
        f"{prefix}.conv1.module.0", f"{prefix}.conv1.module.1",
        in_shape, in_node_id)
    sn1_id = _add_neuron(dag, f"{prefix}.sn1", shape, nparams, bn1_id)

    bn2_id, shape = _add_conv_bn(
        dag, block.conv2.module[0], block.conv2.module[1],
        f"{prefix}.conv2.module.0", f"{prefix}.conv2.module.1",
        shape, sn1_id)
    sn2_id = _add_neuron(dag, f"{prefix}.sn2", shape, nparams, bn2_id)

    if block.downsample is not None:
        ds = block.downsample
        ds_bn_id, _ = _add_conv_bn(
            dag, ds[0].module[0], ds[0].module[1],
            f"{prefix}.downsample.0.module.0",
            f"{prefix}.downsample.0.module.1",
            in_shape, in_node_id)
        shortcut_id = _add_neuron(dag, f"{prefix}.downsample.1", shape,
                                  nparams, ds_bn_id)
    else:
        shortcut_id = in_node_id

    connect_f = getattr(block, 'connect_f', 'ADD')
    res_id = _add_residual(dag, f"{prefix}.residual", connect_f,
                           shape, sn2_id, shortcut_id)
    return res_id, shape


def _extract_bottleneck(dag, block, prefix, in_shape, in_node_id, nparams):
    """Bottleneck: conv1→sn1→conv2→sn2→conv3→sn3→[residual]"""
    bn1_id, shape = _add_conv_bn(
        dag, block.conv1.module[0], block.conv1.module[1],
        f"{prefix}.conv1.module.0", f"{prefix}.conv1.module.1",
        in_shape, in_node_id)
    sn1_id = _add_neuron(dag, f"{prefix}.sn1", shape, nparams, bn1_id)

    bn2_id, shape = _add_conv_bn(
        dag, block.conv2.module[0], block.conv2.module[1],
        f"{prefix}.conv2.module.0", f"{prefix}.conv2.module.1",
        shape, sn1_id)
    sn2_id = _add_neuron(dag, f"{prefix}.sn2", shape, nparams, bn2_id)

    bn3_id, shape = _add_conv_bn(
        dag, block.conv3.module[0], block.conv3.module[1],
        f"{prefix}.conv3.module.0", f"{prefix}.conv3.module.1",
        shape, sn2_id)
    sn3_id = _add_neuron(dag, f"{prefix}.sn3", shape, nparams, bn3_id)

    if block.downsample is not None:
        ds = block.downsample
        ds_bn_id, _ = _add_conv_bn(
            dag, ds[0].module[0], ds[0].module[1],
            f"{prefix}.downsample.0.module.0",
            f"{prefix}.downsample.0.module.1",
            in_shape, in_node_id)
        shortcut_id = _add_neuron(dag, f"{prefix}.downsample.1", shape,
                                  nparams, ds_bn_id)
    else:
        shortcut_id = in_node_id

    connect_f = getattr(block, 'connect_f', 'ADD')
    res_id = _add_residual(dag, f"{prefix}.residual", connect_f,
                           shape, sn3_id, shortcut_id)
    return res_id, shape


def _extract_layer(dag, layer, prefix, in_shape, in_node_id, nparams):
    from models.sewresnet import Bottleneck
    cur_id, cur_shape = in_node_id, in_shape
    for i, block in enumerate(layer):
        fn = _extract_bottleneck if isinstance(block, Bottleneck) else _extract_basic_block
        cur_id, cur_shape = fn(dag, block, f"{prefix}.{i}", cur_shape, cur_id, nparams)
    return cur_id, cur_shape


def extract_sewresnet_dag(model, input_shape):
    """SEW-ResNet ImageNet: conv1→bn1→sn1→maxpool→4 layers→avgpool→fc."""
    dag = OperatorDAG()
    nparams = collect_neuron_params(model)

    bn1_id, shape = _add_conv_bn(dag, model.conv1, model.bn1, 'conv1', 'bn1', input_shape)
    sn1_id = _add_neuron(dag, 'sn1', shape, nparams, bn1_id)

    pool = model.maxpool.module
    pool_shape = _pool_out_shape(shape, _intify(pool.kernel_size),
                                 _intify(pool.stride), _intify(pool.padding))
    pool_id = dag.add_node('maxpool.module', 'maxpool2d', False,
                           {'kernel_size': _intify(pool.kernel_size),
                            'stride': _intify(pool.stride),
                            'padding': _intify(pool.padding)}, shape, pool_shape)
    dag.add_edge(sn1_id, pool_id)

    cur_id, cur_shape = pool_id, pool_shape
    for name in ['layer1', 'layer2', 'layer3', 'layer4']:
        cur_id, cur_shape = _extract_layer(dag, getattr(model, name), name,
                                           cur_shape, cur_id, nparams)

    avg_shape = (cur_shape[0], cur_shape[1], 1, 1)
    avg_id = dag.add_node('avgpool.module', 'avgpool', False, {'output_size': 1},
                          cur_shape, avg_shape)
    dag.add_edge(cur_id, avg_id)

    fc = model.fc
    fc_id = dag.add_node('fc', 'linear', False,
                         {'in_features': fc.in_features, 'out_features': fc.out_features},
                         (avg_shape[0], fc.in_features), (avg_shape[0], fc.out_features))
    dag.add_edge(avg_id, fc_id)
    return dag


def extract_sewresnet_cifar_dag(model, input_shape):
    """SEW-ResNet CIFAR: conv1→bn1→sn1→3 layers→avgpool→fc."""
    dag = OperatorDAG()
    nparams = collect_neuron_params(model)

    bn1_id, shape = _add_conv_bn(dag, model.conv1, model.bn1, 'conv1', 'bn1', input_shape)
    sn1_id = _add_neuron(dag, 'sn1', shape, nparams, bn1_id)

    cur_id, cur_shape = sn1_id, shape
    for name in ['layer1', 'layer2', 'layer3']:
        cur_id, cur_shape = _extract_layer(dag, getattr(model, name), name,
                                           cur_shape, cur_id, nparams)

    avg_shape = (cur_shape[0], cur_shape[1], 1, 1)
    avg_id = dag.add_node('avgpool.module', 'avgpool', False, {'output_size': 1},
                          cur_shape, avg_shape)
    dag.add_edge(cur_id, avg_id)

    fc = model.fc
    fc_id = dag.add_node('fc', 'linear', False,
                         {'in_features': fc.in_features, 'out_features': fc.out_features},
                         (avg_shape[0], fc.in_features), (avg_shape[0], fc.out_features))
    dag.add_edge(avg_id, fc_id)
    return dag
