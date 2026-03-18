  python -m sparse.fr_prune \
    --checkpoint output/spikformer_cifar_cifar100_bs128_lr0.0001_normal/best.pth \
    --config configs/spikformer/spikformer_cifar.yaml \
    --dataset cifar100 --data-root /home/twt/datasets \
    --lam 0.7 --scoring multiplicative \
    --evaluate --output pruned_neuron.pth
