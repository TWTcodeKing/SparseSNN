"""
CLI: profile per-channel firing rates and save to .pt file.

Core profiling logic lives in utils.profiling. This script is a thin wrapper
that handles argument parsing, model/data loading, and saving results.
"""

import argparse
import torch
import torch.nn as nn

from utils.profiling import profile_model_firing_rates


def _find_upstream_linear(model: nn.Module, neuron_name: str) -> tuple:
    """Find the Linear layer that feeds into a given neuron layer.

    Looks for patterns like 'block.0.attn.q_lif' -> 'block.0.attn.q_linear'.

    Returns:
        (linear_name, linear_module) or (None, None) if not found.
    """
    candidates = []
    if neuron_name.endswith('_lif'):
        candidates.append(neuron_name.rsplit('_lif', 1)[0] + '_linear')
    elif neuron_name.endswith('.lif'):
        candidates.append(neuron_name.rsplit('.lif', 1)[0] + '.linear')
    if 'neuron' in neuron_name:
        candidates.append(neuron_name.replace('neuron', 'linear'))

    modules_dict = dict(model.named_modules())
    for cand in candidates:
        if cand in modules_dict and isinstance(modules_dict[cand], nn.Linear):
            return cand, modules_dict[cand]
    return None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Profile per-channel firing rates of spiking neuron layers')
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config file for transformer models')
    parser.add_argument('--model', type=str, default=None,
                        help='ResNet model name (e.g. sew_resnet18)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name: cifar10, cifar100, imagenet, cifar10dvs')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to dataset root directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='GPU IDs (e.g. 0)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for profiling')
    parser.add_argument('--max-batches', type=int, default=50,
                        help='Maximum number of batches to profile '
                             '(ignored if --sample-fraction is set)')
    parser.add_argument('--sample-fraction', type=float, default=None,
                        help='Fraction of validation set to sample (e.g. 0.1 for 10%%). '
                             'Uses stratified sampling to preserve class distribution.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for stratified sampling')
    parser.add_argument('--T', type=int, default=4,
                        help='Number of timesteps')
    parser.add_argument('--output', type=str, default='firing_rates.pt',
                        help='Output file path for firing rates (.pt)')
    args = parser.parse_args()

    from tengine.utils import (
        load_model_config, build_model_from_config, build_model,
        get_dataset_config, build_dataloaders,
    )

    gpu_id = int(args.gpu_ids.split(',')[0])
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    ds_cfg = get_dataset_config(args.dataset)

    if args.config:
        config = load_model_config(args.config)
        config.update(ds_cfg)
        config['T'] = args.T
        model = build_model_from_config(config)
    elif args.model:
        model = build_model(args.model, num_classes=ds_cfg['num_classes'], T=args.T)
    else:
        parser.error('Must specify either --config or --model')

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    model = model.to(device)
    model.eval()

    _, val_loader = build_dataloaders(
        args.dataset, args.data_root, args.batch_size,
        img_size=ds_cfg['img_size'], num_workers=4,
    )

    if args.sample_fraction is not None:
        rates = profile_model_firing_rates(
            model, val_loader, device,
            dataset=val_loader.dataset,
            sample_fraction=args.sample_fraction,
            batch_size=args.batch_size,
            num_workers=4,
            seed=args.seed,
        )
    else:
        rates = profile_model_firing_rates(model, val_loader, device, args.max_batches)

    torch.save({
        'firing_rates': dict(rates),
        'config': {
            'dataset': args.dataset,
            'checkpoint': args.checkpoint,
            'max_batches': args.max_batches,
            'T': args.T,
        },
    }, args.output)
    print(f"\nSaved firing rates to {args.output}")
