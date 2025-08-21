#!/bin/bash

# ==== 主目录配置 ====
output_base="exp_FRED_MD"  # 🧩 统一父目录

# ==== 基本配置 ====
root_path_name=/data/tinyy/first/CrossTimeNet/dataset
data_path_name=FRED-MD.csv
model_id_name=FRED-MD
data_name=FRED-MD

wave_length=4
seq_len=36
token_len=96
pred_len=60
vq_model='ResidualVQ'
block_num=2

# ==== 搜索空间 ====
n_embeds=(64 128 256)
d_models=(96 128)
entropy_penalties=(0.5)
entropy_temps=(0.5)

for n_embed in "${n_embeds[@]}"; do
for d_model in "${d_models[@]}"; do
for entropy_penalty in "${entropy_penalties[@]}"; do
for entropy_temp in "${entropy_temps[@]}"; do

# 🏷️ 唯一标识每组参数
tag="emb${n_embed}_d${d_model}_ep${entropy_penalty}_et${entropy_temp}_wl${wave_length}_bl${block_num}_${vq_model}"

# 🗂️ 构建日志和模型保存路径
log_dir="${output_base}/logs/${model_id_name}/${tag}"
ckpt_dir="${output_base}/checkpoints/${model_id_name}/${tag}"
# mkdir -p "$log_dir"
mkdir -p "$ckpt_dir"

log_file="${log_dir}.log"


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
    --enc_in 107 \
    --pred_len $pred_len \
    --d_model $d_model \
    --block_num $block_num \
    --dropout 0.2 \
    --num_epoch 30 \
    --entropy_penalty $entropy_penalty \
    --entropy_temp $entropy_temp \
    --eval_per_epoch \
    --train_batch_size 4 \
    --test_batch_size 4 \
    --lr 0.0001 \
    --save_path $ckpt_dir \
    > $log_file

done
done
done
done
