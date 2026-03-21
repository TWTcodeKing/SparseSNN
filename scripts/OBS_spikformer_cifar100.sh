  uv run python -m sparse.OBS \
    --checkpoint output/spikformer_cifar_cifar100_bs128_lr0.0001_normal/best.pth \
    --config configs/spikformer/spikformer_cifar.yaml \
    --dataset cifar100 --data-root /home/twt/datasets \
    --evaluate --output pruned_neuron.pth --percdamp 0.01
