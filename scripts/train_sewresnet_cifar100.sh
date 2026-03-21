uv run tengine/train.py --model sew_resnet34 --zero-init-residual \
 --recipe configs/sew_resnet/recipes/cifar100.yaml \
 --dataset cifar100 --data-root /home/twt/datasets/ --gpu-ids 1 \
 --batch-size 100 --epochs 100
