  uv run python -m sparse.OBS \
    --checkpoint output/ms_resnet34_cifar100_bs100_lr0.1_normal/best.pth \
    --model ms_resnet34 --T 6 \
    --dataset cifar100 --data-root /home/twt/datasets \
    --evaluate --output msresnet34_pruned_neuron.pth --percdamp 0.01
