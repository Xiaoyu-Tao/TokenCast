#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Tokenizer" ]; then
    mkdir ./logs/Tokenizer
fi

export CUDA_VISIBLE_DEVICES="7"

# 固定参数
root_path_name=/data/tinyy/vqvae/dataset
data_path_name=ETTh1.csv
model_id_name=shuffle
data_name=pretrain

token_len=16
vq_model='ResidualVQ' # VanillaVQ, SimVQ, SimVQ_CNN 
block_num=3
pred_len=720
seq_len=96

# 可变参数列表
n_embed_list=(1024)  # 修改为多个n_embed值
wave_length_list=(4)
d_model_list=(64)

for n_embed in "${n_embed_list[@]}"; do
  for wave_length in "${wave_length_list[@]}"; do
    for d_model in "${d_model_list[@]}"; do

      save_path='shuffle/ETTh1_'$n_embed'_entropy'
      log_name=$model_id_name'_tok'$token_len'_emb'$n_embed'_wl'$wave_length'_dm'$d_model'_bl'$block_num'_'$vq_model'_'$pred_len'_seq_len'$seq_len'_dependent_entropy_shuffle.log'
      echo "[🚀] Running: n_embed=$n_embed, wave_length=$wave_length, d_model=$d_model"

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
        --chan_indep 1 \
        --enc_in 7 \
        --eval_per_epoch \
        --d_model $d_model \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --block_num $block_num \
        --dropout 0.2 \
        --num_epoch 50 \
        --train_batch_size 128 \
        --test_batch_size 128 \
        --lr 0.0005 \
        --chan_indep 1 \
        --save_path $save_path \
        > logs/Tokenizer/$log_name

    done
  done
done
