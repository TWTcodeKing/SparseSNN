uv run tengine/train.py --config configs/spikformer/spikformer_cifar.yaml \
 --recipe configs/spikformer/recipes/structured_sparse.yaml \
 --dataset cifar100 --data-root /home/twt/datasets/ --gpu-ids 3 \
 --batch-size 128 --epochs 150
