#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=4,5,6,7
export GPUS_PER_NODE=4

./tools/run_dist_launch.sh $GPUS_PER_NODE \
    python -u main.py \
        --dataset_file vaw \
        --mode supervised \
        --scr_coef 2 \
        --att_loss_coef 1 \
        --dec_layers 3 \
        --epochs 15 \
        --lr_drop 13 \
        --batch_size 4 \
        --output_dir exps/vaw/supervised/