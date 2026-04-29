# python -m sparse.snn_sbc --config ./configs/spikingresformer/spikingresformer_m.yaml \
#      --batch-size 32 --dense-checkpoint ./checkpoints/spikingresformer/ImageNet_spikingresformer_m.pth \
#     --dataset imagenet --data-root /data/twt/datasets/imagenet/ --gpu-ids 0 --T 4 --calib-batches 400 --img-size 224

# python -m sparse.snn_sbc --config ./configs/spikingresformer/spikingresformer_l.yaml \
#      --batch-size 32 --dense-checkpoint ./checkpoints/spikingresformer/ImageNet_spikingresformer_l.pth \
#     --dataset imagenet --data-root /data/twt/datasets/imagenet/ --gpu-ids 0 --T 6 --calib-batches 400 --img-size 224

# python -m sparse.snn_sbc --config ./configs/spikingresformer/spikingresformer_s.yaml \
#        --dense-checkpoint ./checkpoints/spikingresformer/ImageNet_spikingresformer_s.pth \
#     --dataset imagenet --data-root /data/twt/datasets/imagenet/ --gpu-ids 0 --T 6 --batch-size 8 --calib-batches 1600 --img-size 224
python -m sparse.snn_sbc --config ./configs/spikingresformer/spikingresformer_ti.yaml \
       --dense-checkpoint ./checkpoints/spikingresformer/ImageNet_spikingresformer_ti.pth \
    --dataset imagenet --data-root /data/twt/datasets/imagenet/ --gpu-ids 1 --T 4 --batch-size 8 --calib-batches 1600 --img-size 224

# python -m sparse.snn_sbc --config ./configs/maxformer/maxformer_10_384.yaml \
#      --batch-size 32 --dense-checkpoint ./checkpoints/maxformer/imagenet/maxformer_10_384_T4.pth \
#     --dataset imagenet --data-root /data/twt/datasets/imagenet/ --gpu-ids 2 --T 4 --calib-batches 400 --img-size 224

# python -m sparse.snn_sbc --config ./configs/maxformer/maxformer_10_512.yaml \
#      --batch-size 32 --dense-checkpoint ./checkpoints/maxformer/imagenet/maxformer_10_512_T4.pth \
#     --dataset imagenet --data-root /data/twt/datasets/imagenet/ --gpu-ids 2 --T 4 --calib-batches 400 --img-size 224

# python -m sparse.snn_sbc --config ./configs/maxformer/maxformer_10_768.yaml \
#        --dense-checkpoint ./checkpoints/maxformer/imagenet/maxformer_10_768_T4.pth \
#     --dataset imagenet --data-root /data/twt/datasets/imagenet/ --gpu-ids 2 --T 4 --batch-size 8 --calib-batches 1600 --img-size 224


# python -m sparse.snn_sbc --config ./configs/maxformer/ms_qkformer_10_384.yaml \
#        --dense-checkpoint ./checkpoints/maxformer/imagenet/ms_qk_384_T4.pth \
#     --dataset imagenet --data-root /data/twt/datasets/imagenet/ --gpu-ids 2 --T 4 --batch-size 8 --calib-batches 1600 --img-size 224
