#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=4

./tools/run_dist_launch.sh $GPUS_PER_NODE \
    python -u main.py \
        --dataset_file vaw \
        --mode zero_shot \
        --scr_coef 2 \
        --att_loss_coef 1 \
        --dec_layers 3 \
        --epochs 9 \
        --batch_size 4 \
        --output_dir exps/vaw/zero_shot_/

