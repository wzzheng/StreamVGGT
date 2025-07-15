#!/bin/bash
set -e

workdir='..'
model_name='StreamVGGT'
ckpt_name='checkpoints'
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
datasets=('sintel' 'bonn' 'kitti' 'nyu')

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/monodepth/${data}_${model_name}"
    echo "$output_dir"
    CUDA_LAUNCH_BLOCKING=1 python ./eval/monodepth/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data"
done

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/monodepth/${data}_${model_name}"
    CUDA_LAUNCH_BLOCKING=1 python ./eval/monodepth/eval_metrics.py \
        --output_dir "$output_dir" \
        --eval_dataset "$data"
done

