uv run python -m vis.firing_rate_profile \
    --config configs/spikformer/spikformer_cifar.yaml \
    --dataset cifar100 --data-root /home/twt/datasets \
    --checkpoint output/spikformer_cifar_cifar100_bs128_lr0.0001_normal/best.pth \
    --sample-fraction 0.1 --gpu-ids 0 \
    --output firing_rates.pt