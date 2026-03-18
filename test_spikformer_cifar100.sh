uv run tengine/test.py --config configs/spikformer/spikformer_cifar.yaml \
    --dataset cifar100 --data-root /home/twt/datasets \
    --checkpoint output/spikformer_cifar_cifar100_bs128_lr0.0001_normal/best.pth --gpu-ids 0