  # MaxFormer-384 (reported: 77.82%)
#   uv run tengine/test.py --config configs/maxformer/maxformer_10_384.yaml \
#     --dataset imagenet --data-root /data/twt/datasets/imagenet \
#     --checkpoint checkpoints/maxformer/imagenet/maxformer_10_384_T4.pth \
#     --T 4 --batch-size 128 --gpu-ids 0

#   # MaxFormer-512 (reported: 79.86%)
#   uv run tengine/test.py --config configs/maxformer/maxformer_10_512.yaml \
#     --dataset imagenet --data-root /data/twt/datasets/imagenet \
#     --checkpoint checkpoints/maxformer/imagenet/maxformer_10_512_T4.pth \
#     --T 4 --batch-size 128 --gpu-ids 0

#   # MaxFormer-768 (reported: 82.39%)
#   uv run tengine/test.py --config configs/maxformer/maxformer_10_768.yaml \
#     --dataset imagenet --data-root /data/twt/datasets/imagenet \
#     --checkpoint checkpoints/maxformer/imagenet/maxformer_10_768_T4.pth \
#     --T 4 --batch-size 64 --gpu-ids 0

# MS_QKFormer-384 T=4 (reported: 76.48%)
uv run tengine/test.py --config configs/maxformer/ms_qkformer_10_384.yaml \
    --dataset imagenet --data-root /data/twt/datasets/imagenet \
    --checkpoint checkpoints/maxformer/imagenet/ms_qk_384_T4.pth \
    --T 4 --batch-size 64 --gpu-ids 0

# MS_QKFormer-768 T=1 (reported: 77.78%)
uv run tengine/test.py --config configs/maxformer/ms_qkformer_10_768.yaml \
    --dataset imagenet --data-root /data/twt/datasets/imagenet \
    --checkpoint checkpoints/maxformer/imagenet/ms_qk_768_T1.pth \
    --T 1 --batch-size 64 --gpu-ids 0