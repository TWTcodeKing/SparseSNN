"""Export Spikformer-1-512 to ONNX and build TRT engine.

Usage:
    # CIFAR-100 (32x32)
    python motivation/prepare_spikformer.py

    # ImageNet (224x224)
    python motivation/prepare_spikformer.py --dataset imagenet --max-batch 32
"""

import argparse
import os
import sys
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

OUT = "motivation/output"


def export_onnx(dataset, img_size, num_classes, max_batch):
    from tengine.utils import build_model_from_config
    from iengine.backends.tensorrt.export import export_onnx as _export
    from models.neurons import reset_net

    with open("configs/spikformer/spikformer_1_512.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg.update(num_classes=num_classes, in_channels=3, T=4, img_size=img_size)

    model = build_model_from_config(cfg).cuda().eval()
    reset_net(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Spikformer-1-512: {n_params:,} params ({n_params*2/1024/1024:.1f} MB FP16)")
    print(f"  dataset={dataset}, img={img_size}x{img_size}, classes={num_classes}")

    tag = f"spikformer_1_512_{dataset}"
    onnx_path = os.path.join(OUT, f"{tag}.onnx")
    _export(model, onnx_path, input_shape=(1, 3, img_size, img_size), opset=17,
            dynamic_batch=True, simplify=True, verbose=True)
    reset_net(model)
    return onnx_path, tag


def build_trt(onnx_path, tag, max_batch):
    from iengine.backends.tensorrt.builder import build_engine

    engine_path = os.path.join(OUT, f"{tag}.engine")
    build_engine(
        onnx_path, engine_path,
        fp16=True, sparse=False,
        min_batch=1, opt_batch=max(1, max_batch // 2), max_batch=max_batch,
        workspace_gb=8.0, verbose=True,
    )
    return engine_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar100",
                        choices=["cifar100", "imagenet"])
    parser.add_argument("--max-batch", type=int, default=8)
    args = parser.parse_args()

    if args.dataset == "imagenet":
        img_size, num_classes = 224, 1000
    else:
        img_size, num_classes = 32, 100

    os.makedirs(OUT, exist_ok=True)
    onnx_path, tag = export_onnx(args.dataset, img_size, num_classes, args.max_batch)
    engine_path = build_trt(onnx_path, tag, args.max_batch)
    print(f"\nReady: {engine_path}")


if __name__ == "__main__":
    main()
