"""Benchmark SNN models using ONNX Runtime (CUDA EP).

Exports models to standard ONNX (no custom ops) and runs inference with
ONNXRuntime's CUDAExecutionProvider. Provides a baseline for comparing
against sengine, TensorRT, and torch.compile.

Usage:
    python scripts/bench_onnxruntime_latency.py \
        --model sew_resnet18 --dataset cifar100 --T 4 --batch-sizes 1,4,8,16

    python scripts/bench_onnxruntime_latency.py \
        --config configs/maxformer/maxformer_10_512.yaml --dataset imagenet \
        --T 4 --batch-sizes 1,4
"""

import argparse
import os
import sys
import time
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tengine.utils import (
    build_model, build_model_from_config, load_model_config, get_dataset_config,
)
from models.neurons import reset_net


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SNN with ONNX Runtime")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="ResNet model name")
    group.add_argument("--config", type=str, help="Transformer YAML config path")

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--T", type=int, default=4, help="Temporal steps")
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-sizes", type=str, default="1,4,8,16")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--gpu-ids", type=int, default=0)
    parser.add_argument("--onnx-dir", type=str, default="onnxrt_exports",
                        help="Directory for ONNX model files")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 input for ORT inference")
    parser.add_argument("--graph-opt", type=str, default="all",
                        choices=["disabled", "basic", "extended", "all"],
                        help="ORT graph optimization level")
    return parser.parse_args()


def build_snn_model(args, ds_cfg, device):
    """Build SNN model from --model or --config."""
    img_size = args.img_size or ds_cfg['img_size']

    if args.model:
        model = build_model(
            args.model,
            T=args.T,
            num_classes=ds_cfg['num_classes'],
            in_channels=ds_cfg['in_channels'],
        )
    else:
        config = load_model_config(args.config)
        config.update(ds_cfg)
        config['T'] = args.T
        config['img_size'] = img_size
        model = build_model_from_config(config)

    model = model.to(device).eval()

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
        model.load_state_dict(state_dict, strict=False)

    return model, img_size


def export_onnx(model, input_shape, onnx_path, opset, device):
    """Export model to standard ONNX (no custom ops).

    SNN models have internal reshapes that merge (T, B) dimensions, so the
    batch dimension cannot be dynamic. Export must be done per batch size.
    """
    if os.path.exists(onnx_path):
        print(f"  [ort] Reusing existing ONNX: {onnx_path}")
        return

    print(f"  [ort] Exporting ONNX (opset={opset}, shape={input_shape})...")
    # Export on CPU to avoid GPU OOM on large batch sizes.
    # ONNX export only traces the graph — no GPU compute needed.
    model_cpu = model.cpu()
    dummy_input = torch.randn(*input_shape, device='cpu')
    reset_net(model_cpu)

    # Force legacy TorchScript exporter (dynamo exporter has issues with
    # opset downgrade and dynamic_axes on internal reshape ops)
    torch.onnx.export(
        model_cpu, dummy_input, onnx_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )
    model.to(device)  # move back to GPU

    # Simplify with onnxsim if available
    try:
        import onnx
        from onnxsim import simplify
        onnx_model = onnx.load(onnx_path)
        simplified, ok = simplify(onnx_model)
        if ok:
            onnx.save(simplified, onnx_path)
            print(f"  [ort] Simplified ONNX saved: {onnx_path}")
        else:
            print(f"  [ort] Simplification failed, using original")
    except ImportError:
        pass  # onnxsim not installed, use unsimplified
    except Exception as e:
        print(f"  [ort] Simplification error: {e}, using original")


def convert_onnx_to_fp16(onnx_path):
    """Convert entire ONNX model to FP16 (weights, I/O, Cast ops).

    Uses direct conversion instead of onnxconverter_common to avoid
    mixed fp32/fp16 graphs that cause ORT allocator errors.
    """
    fp16_path = onnx_path.replace(".onnx", "_fp16.onnx")
    if os.path.exists(fp16_path):
        print(f"  [ort] Reusing existing FP16 ONNX: {fp16_path}")
        return fp16_path

    import onnx
    from onnx import numpy_helper, TensorProto

    print(f"  [ort] Converting model to FP16 (full conversion)...")
    model = onnx.load(onnx_path, load_external_data=True)

    # Convert all float32 initializers to float16
    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            tensor = numpy_helper.to_array(init)
            new_tensor = numpy_helper.from_array(
                tensor.astype(np.float16), name=init.name
            )
            init.CopyFrom(new_tensor)

    # Fix Cast ops: FLOAT -> FLOAT16
    for node in model.graph.node:
        if node.op_type == 'Cast':
            for attr in node.attribute:
                if attr.name == 'to' and attr.i == TensorProto.FLOAT:
                    attr.i = TensorProto.FLOAT16
        # Fix scalar tensor attributes in nodes (e.g. Constant ops)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.ByteSize() > 0:
                if attr.t.data_type == TensorProto.FLOAT:
                    tensor = numpy_helper.to_array(attr.t)
                    new_tensor = numpy_helper.from_array(tensor.astype(np.float16))
                    attr.t.CopyFrom(new_tensor)

    # Convert graph I/O types
    for inp in model.graph.input:
        if inp.type.tensor_type.elem_type == TensorProto.FLOAT:
            inp.type.tensor_type.elem_type = TensorProto.FLOAT16
    for out in model.graph.output:
        if out.type.tensor_type.elem_type == TensorProto.FLOAT:
            out.type.tensor_type.elem_type = TensorProto.FLOAT16

    # Strip stale value_info to avoid shape/type checker conflicts
    del model.graph.value_info[:]

    onnx.save(model, fp16_path)
    print(f"  [ort] FP16 model saved: {fp16_path}")
    return fp16_path


def create_ort_session(onnx_path, device_id, graph_opt):
    """Create ONNX Runtime inference session with CUDA EP."""
    import platform
    import onnxruntime as ort

    opt_map = {
        "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }

    # ORT_ENABLE_ALL includes x86-specific passes that crash on ARM (Jetson)
    if platform.machine() == 'aarch64' and graph_opt == 'all':
        print("  [ort] WARNING: downgrading graph_opt 'all' -> 'extended' on aarch64 "
              "(ORT x86-specific passes crash on ARM)")
        graph_opt = 'extended'

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = opt_map[graph_opt]

    providers = [
        ('CUDAExecutionProvider', {
            'device_id': device_id,
            'arena_extend_strategy': 'kSameAsRequested',
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
        }),
        'CPUExecutionProvider',
    ]

    session = ort.InferenceSession(onnx_path, sess_opts, providers=providers)
    return session


def benchmark_ort(session, input_array, warmup, iters):
    """Measure ORT inference latency."""
    import onnxruntime as ort

    input_name = session.get_inputs()[0].name
    io_binding = session.io_binding()

    # Create ORT tensor on GPU
    ort_input = ort.OrtValue.ortvalue_from_numpy(input_array, 'cuda', 0)

    # Warmup
    for _ in range(warmup):
        io_binding.bind_ortvalue_input(input_name, ort_input)
        io_binding.bind_output(session.get_outputs()[0].name, 'cuda')
        session.run_with_iobinding(io_binding)
    io_binding.synchronize_outputs()

    # Timed iterations
    times = []
    for _ in range(iters):
        io_binding.bind_ortvalue_input(input_name, ort_input)
        io_binding.bind_output(session.get_outputs()[0].name, 'cuda')

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        session.run_with_iobinding(io_binding)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return np.array(times)


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_ids}")
    torch.cuda.set_device(device)

    import onnxruntime as ort
    print(f"[ort] ONNX Runtime version: {ort.__version__}")
    print(f"[ort] Available providers: {ort.get_available_providers()}")

    ds_cfg = get_dataset_config(args.dataset)
    model, img_size = build_snn_model(args, ds_cfg, device)
    model_name = args.model or os.path.splitext(os.path.basename(args.config))[0]

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    os.makedirs(args.onnx_dir, exist_ok=True)

    # SNN models have internal (T,B)->T*B reshapes with baked-in constants,
    # so we must export a separate ONNX per batch size.
    for bs in batch_sizes:
        onnx_path = os.path.join(
            args.onnx_dir, f"{model_name}_{args.dataset}_T{args.T}_B{bs}.onnx"
        )
        export_shape = (bs, ds_cfg['in_channels'], img_size, img_size)
        export_onnx(model, export_shape, onnx_path, args.opset, device)

    # Free PyTorch model
    del model
    torch.cuda.empty_cache()

    gpu_name = torch.cuda.get_device_name(device)
    dtype = np.float16 if args.fp16 else np.float32

    # Header
    print()
    print("=" * 80)
    print(f"  ONNX Runtime (CUDA EP) | {model_name} | {args.dataset} | T={args.T}")
    print(f"  GPU: {gpu_name} | dtype: {'FP16' if args.fp16 else 'FP32'} | opt: {args.graph_opt}")
    print("=" * 80)
    print(f"  {'Batch':<10} {'Latency (ms)':>14} {'Std (ms)':>10} {'Throughput':>12}")
    print(f"  {'--------':<10} {'--------------':>14} {'----------':>10} {'--------':>12}")

    results = {}
    for bs in batch_sizes:
        try:
            onnx_path = os.path.join(
                args.onnx_dir, f"{model_name}_{args.dataset}_T{args.T}_B{bs}.onnx"
            )

            # Convert to FP16 if requested
            if args.fp16:
                onnx_path = convert_onnx_to_fp16(onnx_path)

            # Create ORT session for this batch size
            session = create_ort_session(onnx_path, args.gpu_ids, args.graph_opt)

            input_array = np.random.randn(
                bs, ds_cfg['in_channels'], img_size, img_size
            ).astype(dtype)

            times = benchmark_ort(session, input_array, args.warmup, args.iters)
            mean_ms = times.mean()
            std_ms = times.std()
            throughput = bs / (mean_ms / 1000)

            results[bs] = {'mean_ms': mean_ms, 'std_ms': std_ms, 'throughput': throughput}
            print(f"  B={bs:<7} {mean_ms:>11.3f}ms {std_ms:>9.3f}ms {throughput:>9.0f}/s")

            del session

        except Exception as e:
            print(f"  B={bs:<7} FAILED: {e}")

    print("=" * 80)
    print()

    return results


if __name__ == "__main__":
    main()
