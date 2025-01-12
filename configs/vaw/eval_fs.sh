#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=4,5,6,7
export GPUS_PER_NODE=4

./tools/run_dist_launch.sh $GPUS_PER_NODE \
    python -u main.py \
        --dataset_file vaw \
        --mode supervised \
        --eval \
        --zrse \
        --dec_layers 3 \
        --batch_size 4 \
        --pretrained exps/vaw/supervised_/checkpoint.pth \
        --output_dir exps/vaw/supervised__/

