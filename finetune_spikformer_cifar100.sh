uv run python -m sparse.permutation \
    --config configs/spikformer/spikformer_cifar.yaml \
    --dataset cifar100 --data-root /home/twt/datasets \
    --checkpoint output/spikformer_cifar_cifar100_bs128_lr0.0001_normal/best.pth \
    --rates firing_rates.pt \
    --evaluate --gpu-ids 3 --method permutation \
    --output permuted_model.pth