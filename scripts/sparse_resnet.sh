# python -m sparse.snn_sbc --model ms_resnet18 \
#      --batch-size 32 --dense-checkpoint ./checkpoints/msresnet/imagenet/resnet18.pth \
#     --dataset imagenet --data-root /data/twt/datasets/imagenet/ --gpu-ids 0 --T 6 --calib-batches 400 --img-size 224

# python -m sparse.snn_sbc --model ms_resnet34 \
#      --batch-size 32 --dense-checkpoint ./checkpoints/msresnet/imagenet/resnet34.pth \
#     --dataset imagenet --data-root /data/twt/datasets/imagenet/ --gpu-ids 0 --T 6 --calib-batches 400 --img-size 224

python -m sparse.snn_sbc --model ms_resnet104 \
       --dense-checkpoint ./checkpoints/msresnet/imagenet/resnet104.pth \
    --dataset imagenet --data-root /data/twt/datasets/imagenet/ --gpu-ids 0 --T 6 --batch-size 8 --calib-batches 1600 --img-size 224
