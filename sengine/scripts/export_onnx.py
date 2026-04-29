#!/usr/bin/env python3
"""Export SNN models to plugin-mode ONNX for sengine.

Exports ONNX with FusedIFNeuron/FusedLIFNeuron custom ops, which is
directly parseable by sengine's ONNXParser.

Usage:
    # ResNet (by model name)
    python sengine/scripts/export_onnx.py --model sew_resnet18 --dataset imagenet --T 4

    # Transformer (by config YAML)
    python sengine/scripts/export_onnx.py --config configs/spikformer/spikformer_8_384.yaml \
        --dataset cifar100 --T 4 --img-size 224

    # With checkpoint
    python sengine/scripts/export_onnx.py --model sew_resnet34 --dataset cifar100 \
        --checkpoint obc_pt/sew_resnet34_cifar100_sbc_2_4_global.pth
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models.neurons import reset_net


def export_model(model_name: str, output_dir: str, T: int = 4,
                 dataset: str = 'imagenet', img_size: int | None = None,
                 config: str | None = None, checkpoint: str | None = None):
    """Export one model to plugin-mode ONNX."""
    device = torch.device('cuda:0')
    os.makedirs(output_dir, exist_ok=True)

    if img_size is None:
        img_size = 32 if 'cifar' in dataset else 224

    num_classes = 100 if 'cifar100' in dataset else (10 if 'cifar10' in dataset else 1000)

    print(f"\n=== Exporting {model_name} (T={T}, img={img_size}x{img_size}) ===")

    # Build model
    if config:
        import yaml
        from tengine.utils import build_model_from_config
        with open(config) as f:
            cfg = yaml.safe_load(f)
        cfg['num_classes'] = num_classes
        cfg['T'] = T
        cfg['img_size'] = img_size
        model = build_model_from_config(cfg)
    else:
        from tengine.utils import build_model
        model = build_model(model_name, T=T, num_classes=num_classes, in_channels=3)

    model = model.to(device).eval()

    # Load checkpoint
    if checkpoint and os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded checkpoint: {checkpoint}")
    else:
        print(f"  No checkpoint (random weights)")

    input_shape = (1, 3, img_size, img_size)
    tag = f"{model_name}_{dataset}"
    plugin_path = os.path.join(output_dir, f"{tag}_plugin.onnx")

    # Export plugin-mode ONNX
    import sengine.tdl.neuron_ops as nops
    nops.use_native_onnx = False
    try:
        from sengine.tdl.transforms import export_with_fused_neurons
        export_with_fused_neurons(
            model, plugin_path, input_shape=input_shape, opset=17,
            dynamic_batch=False, verbose=True)
        print(f"  Saved: {plugin_path} ({os.path.getsize(plugin_path)/1e6:.1f} MB)")
    except Exception as e:
        print(f"  Export FAILED: {e}")
        plugin_path = None
    finally:
        nops.use_native_onnx = True
        reset_net(model)

    return plugin_path


def main():
    parser = argparse.ArgumentParser(description="Export SNN models to plugin-mode ONNX")
    parser.add_argument('--model', type=str, default='sew_resnet18',
                        help='Model name (e.g., sew_resnet18, sew_resnet34)')
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config for transformer models')
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='sengine/exports')
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=None)
    args = parser.parse_args()

    export_model(
        model_name=args.model, output_dir=args.output_dir,
        T=args.T, dataset=args.dataset, img_size=args.img_size,
        config=args.config, checkpoint=args.checkpoint,
    )


if __name__ == '__main__':
    main()
