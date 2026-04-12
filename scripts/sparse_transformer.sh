python -m sparse.snn_sbc --config ./configs/spikingresformer/spikingresformer_m.yaml \
     --batch-size 32 --dense-checkpoint ./checkpoints/spikingresformer/ImageNet_spikingresformer_m.pth \
    --dataset imagenet --data-root /data/twt/datasets/imagenet/ --gpu-ids 0 --T 4 --calib-batches 400 --img-size 224

python -m sparse.snn_sbc --config ./configs/spikingresformer/spikingresformer_l.yaml \
     --batch-size 32 --dense-checkpoint ./checkpoints/spikingresformer/ImageNet_spikingresformer_l.pth \
    --dataset imagenet --data-root /data/twt/datasets/imagenet/ --gpu-ids 0 --T 6 --calib-batches 400 --img-size 224

python -m sparse.snn_sbc --config ./configs/spikingresformer/spikingresformer_s.yaml \
       --dense-checkpoint ./checkpoints/spikingresformer/ImageNet_spikingresformer_s.pth \
    --dataset imagenet --data-root /data/twt/datasets/imagenet/ --gpu-ids 0 --T 6 --batch-size 8 --calib-batches 1600 --img-size 224
