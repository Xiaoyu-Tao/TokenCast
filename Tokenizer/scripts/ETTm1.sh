if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Tokenizer" ]; then
    mkdir ./logs/Tokenizer
fi

export CUDA_VISIBLE_DEVICES="7"

# source /data/tingyue/anaconda3/bin/activate DD-Time

root_path_name=/data/tinyy/vqvae/dataset
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
wave_length=4
token_len=16
seq_len=96
pred_len=720
vq_model='ResidualVQ' # VanillaVQ, SimVQ

# Array of n_embed values to test
n_embeds=(64 128 256 512)

for n_embed in "${n_embeds[@]}"
do
    save_path="dependent/ETTm1_${n_embed}_entropy"
    log_name=$model_id_name'_tok'$token_len'_emb'$n_embed'_wl'$wave_length'_dm'$d_model'_bl'$block_num'_'$vq_model'_'$pred_len'_seq_len'$seq_len'_dependent_entropy.log'

    python -u main.py \
        --is_training 1 \
        --vq_model $vq_model \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --data $data_name \
        --wave_length $wave_length \
        --features M \
        --token_len $token_len \
        --pred_len $pred_len \
        --seq_len $seq_len \
        --n_embed $n_embed \
        --enc_in 7 \
        --d_model 64 \
        --block_num 3 \
        --dropout 0.2 \
        --num_epoch 50 \
        --eval_per_epoch \
        --lr 0.0003 \
        --chan_indep 1 \
        --save_path $save_path \
        > logs/Tokenizer/$log_name
done