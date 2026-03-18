uv run tengine/train.py --model sew_resnet34 --zero-init-residual \
 --recipe configs/sew_resnet/recipes/cifar10-dvs.yaml \
 --dataset cifar10dvs --data-root /home/twt/datasets/cifar10-dvs --gpu-ids 1 \
 --batch-size 32 --epochs 192
