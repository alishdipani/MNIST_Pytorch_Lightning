#!/bin/bash

# Setting values of arguments
gpus=1
val_ratio=0.2
batch_size=32
seed=42
epochs=5

python train.py --gpus $gpus \
    --val_ratio $val_ratio \
    --batch_size $batch_size \
    --seed $seed \
    --epochs $epochs
