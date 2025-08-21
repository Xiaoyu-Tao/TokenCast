#!/bin/bash

# ========== 实验日志与模型保存目录 ==========
LOG_DIR="log_vq_sft"
mkdir -p "$LOG_DIR"

BASE_CHECKPOINT_DIR="checkpoints_vq_sft"
mkdir -p "$BASE_CHECKPOINT_DIR"

# ========== 你可以自由修改的变量 ==========
LEARNING_RATES=(1e-5)
EPOCHS_LIST=(10)
VQ_TYPE="ResidualVQ"
FROZEN=0  # ✅ 可调：是否冻结模型除输出层和嵌入层以外的参数（1=冻结，0=不冻结）

# 固定参数
PRED_LEN=24
SEQ_LEN=96
Token_LEN=16
D_MODEL=64
N_EMBED=256

BATCH_SIZE=4
ELECT_RATE=1
PRETRAIN_LR=1e-3
DEVICES="0,2"

# ========== 显式指定你想用的 GPU ==========
export CUDA_VISIBLE_DEVICES=$DEVICES

# ========== 启动实验 ==========
for LR in "${LEARNING_RATES[@]}"; do
    for EPOCHS in "${EPOCHS_LIST[@]}"; do

        VQVAE_PATH="./TSTokenizer/checkpoints/CzeLan_96_dm64_dr0.2_emb256_wl4_bl2_ResidualVQ_unfreeze_codebook"
                
        CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR}/pred_${PRED_LEN}_seq_${SEQ_LEN}/lr_${LR}_ep_${EPOCHS}_vq_${VQ_TYPE}_chat_mask_64_pretrain_frozen_${FROZEN}"
        LOG_FILE="$LOG_DIR/experiment_pred_${PRED_LEN}_seq${SEQ_LEN}_lr_${LR}_ep_${EPOCHS}_$(date +'%Y%m%d_%H%M%S')_pretrain_frozen_${FROZEN}.log"

        echo "🔹 Running experiment with lr=$LR, epochs=$EPOCHS, frozen=$FROZEN on GPUs $DEVICES"

        accelerate launch \
            --multi_gpu \
            --num_processes 2 \
            --main_process_port 29600 \
            run.py \
            --is_training 1 \
            --pretrain 0 \
            --shuffle 0 \
            --batch_size "$BATCH_SIZE" \
            --data CzeLan \
            --root_path "/data/tinyy/first/CrossTimeNet/dataset" \
            --data_path "CzeLan.csv" \
            --pred_len "$PRED_LEN" \
            --seq_len "$SEQ_LEN" \
            --token_len "$Token_LEN" \
            --n_embed "$N_EMBED" \
            --d_model "$D_MODEL" \
            --learning_rate "$LR" \
            --weight_decay 0 \
            --model "qwen4ts" \
            --task_name "long_term_forecast_bert_v4" \
            --vqvae_model_path "$VQVAE_PATH" \
            --dropout 0.1 \
            --chan_indep 0 \
            --enc_in 11 \
            --feat_dim 11 \
            --local_model_path "/data/tinyy/first/CrossTimeNet/2-models/Qwen2.5-0.5B" \
            --pretrained_model "checkpoints_pretrain/pred_720_seq_512/lr_1e-5_ep_10_vq_ResidualVQ_chat_mask_64_pretrain_frozen_0_0.5/long_term_forecast_bert_v4_ETTh1_qwen4ts_720_ResidualVQ/checkpoint.pth" \
            --frozen "$FROZEN" \
            --zero 1 \
            --layers 0 \
            --params 1 \
            --wave_length 4 \
            --checkpoints "$CHECKPOINT_DIR" \
            --seed 42 \
            --init_method "word" \
            --train_epochs "$EPOCHS" \
            --pretrain_lr "$PRETRAIN_LR" \
            --use_multi_gpu \
            --elect_rate "$ELECT_RATE" \
            --VQ_type "$VQ_TYPE" \
            --accumulation_steps 4 \
            --test 0 \
            > "$LOG_FILE" 2>&1
    done
done
