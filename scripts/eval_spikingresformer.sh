uv run tengine/test.py \
    --config configs/spikingresformer/spikingresformer_l.yaml \
    --dataset imagenet --data-root /data/twt/datasets/imagenet \
    --checkpoint checkpoints/spikingresformer/ImageNet_spikingresformer_l.pth \
    --T 4 --batch-size 64 --gpu-ids 0