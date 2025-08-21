#!/bin/bash

# 创建统一的父目录以及子目录
mkdir -p ./exp_NYSE/logs/Tokenizer
mkdir -p ./exp_NYSE/checkpoints/Tokenizer

# 环境设置
export CUDA_VISIBLE_DEVICES="1"
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
source ~/.bashrc

# 固定参数
root_path_name='./datasets'
data_path_name=NYSE.csv
model_id_name=NYSE
data_name=NYSE

wave_length=2
seq_len=36
token_len=96
pred_len=60
vq_model='ResidualVQ'
block_num=2

# 可变参数列表
n_embeds=(64)
d_models=(64)

# 多参数组合实验
for d_model in "${d_models[@]}"; do
    for n_embed in "${n_embeds[@]}"; do

                tag="${data_name}_emb${n_embed}_d${d_model}_wl${wave_length}_bl${block_num}"
                save_path="./exp_NYSE/checkpoints/${tag}"
                log_file="./exp_NYSE/logs/${tag}.log"

                mkdir -p $save_path

                python -u main.py \
                    --is_training 1 \
                    --vq_model $vq_model \
                    --root_path $root_path_name \
                    --data_path $data_path_name \
                    --data $data_name \
                    --wave_length $wave_length \
                    --features M \
                    --token_len $token_len \
                    --n_embed $n_embed \
                    --chan_indep 0 \
                    --enc_in 5 \
                    --pred_len $pred_len \
                    --d_model $d_model \
                    --block_num $block_num \
                    --dropout 0.2 \
                    --num_epoch 30 \
                    --eval_per_epoch \
                    --train_batch_size 16 \
                    --test_batch_size 16 \
                    --save_path $save_path \
                    --lr 0.0001 \
                    > $log_file
    done
done
