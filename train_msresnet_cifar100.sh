uv run tengine/train.py --model ms_resnet34 \
 --recipe configs/ms_resnet/recipes/cifar100.yaml \
 --dataset cifar100 --data-root /home/twt/datasets/ --gpu-ids 3 \
 --batch-size 100 --epochs 100 --opt sgd 
