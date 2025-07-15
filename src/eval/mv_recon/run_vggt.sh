#!/bin/bash

set -e
workdir='..'
model_name='VGGT'
ckpt_name='model'
model_weights="${workdir}/ckpt/${ckpt_name}.pt"


output_dir="${workdir}/eval_results/mv_recon/${model_name}_${ckpt_name}"
echo "$output_dir"
accelerate launch --num_processes 1 --main_process_port 29602 ./eval/mv_recon/launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --model_name "$model_name" \
     