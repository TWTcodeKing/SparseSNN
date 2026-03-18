uv run tengine/train.py --model ms_resnet34 \
 --recipe configs/ms_resnet/recipes/cifar10-dvs.yaml \
 --dataset cifar10dvs --data-root /home/twt/datasets/cifar10-dvs --gpu-ids 1 \
 --batch-size 16 --epochs 192
