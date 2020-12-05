#!/bin/bash

# Setting values of arguments

# Training
gpus=1
batch_size=32
epochs=5
num_workers=4

# Dataset
val_ratio=0.2

# Seed
seed=42

# Model
model_name=CNN3

# Experiment
experiment_name=MNIST-Model=${model_name}

python train.py --gpus $gpus \
    --val_ratio $val_ratio \
    --batch_size $batch_size \
    --seed $seed \
    --epochs $epochs \
    --num_workers $num_workers \
    --model_name $model_name \
    --experiment_name $experiment_name
