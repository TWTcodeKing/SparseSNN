uv run python tengine/train.py \
    --model ms_resnet34 \
    --dataset cifar100 --data-root /home/twt/datasets/ --gpu-ids 3 \
    --dynamic-sparse --dyn-n 2 --dyn-m 4 --dyn-grow-ratio 0.5 \
    --opt sgd --momentum 0.9 --sched cosine --epochs 100 --batch-size 128 --T 6 --lr 0.1