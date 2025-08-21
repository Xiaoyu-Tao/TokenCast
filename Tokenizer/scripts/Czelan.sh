#!/bin/bash

# ========== 环境设置 ==========
export CUDA_VISIBLE_DEVICES="3"
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
source ~/.bashrc


# ========== 数据与基本配置 ==========
root_path_name='./datasets'
data_path_name=CzeLan.csv
model_id_name=CzeLan
data_name=CzeLan

wave_length=4
seq_len=96
token_len=96
pred_len=192
vq_model='ResidualVQ'  # VanillaVQ, SimVQ
block_num=2
enc_in=11
train_batch_size=256
test_batch_size=256
num_epoch=30
lr=0.001

n_embeds=(64 128 256 512)
d_models=(64)

# ========== 日志与检查点目录结构 ==========
BASE_LOG_DIR=$model_id_name/logs
BASE_CKPT_DIR=$model_id_name/checkpoints
mkdir -p $BASE_LOG_DIR
mkdir -p $BASE_CKPT_DIR

# ========== 启动网格搜索训练 ==========
for n_embed in "${n_embeds[@]}"; do
for d_model in "${d_models[@]}"; do

    # 保存路径（用于模型、日志）
    combo_name="emb${n_embed}_d${d_model}_wl${wave_length}_bl${block_num}_${vq_model}"
    save_path="${BASE_CKPT_DIR}/${combo_name}"
    log_file="${BASE_LOG_DIR}/${combo_name}.log"
    mkdir -p "$save_path"

    echo "🚀 Running config: $combo_name"
    
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
        --enc_in $enc_in \
        --pred_len $pred_len \
        --d_model $d_model \
        --block_num $block_num \
        --dropout 0.2 \
        --num_epoch $num_epoch \
        --eval_per_epoch \
        --train_batch_size $train_batch_size \
        --test_batch_size $test_batch_size \
        --save_path "$save_path" \
        --lr $lr \
        > "$log_file"
        done
done
