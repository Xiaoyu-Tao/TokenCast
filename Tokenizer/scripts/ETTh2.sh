if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Tokenizer" ]; then
    mkdir ./logs/Tokenizer
fi

export CUDA_VISIBLE_DEVICES="1"
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
source ~/.bashrc
source /data/tinyy/miniconda3/bin/activate LLM

root_path_name=/data/tinyy/vqvae/dataset
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

# Array of n_embed values to test
n_embeds=(512 256)
wave_length=4
token_len=96
seq_len=96
pred_len=720
vq_model='ResidualVQ' # VanillaVQ, SimVQ
block_num=3

# Loop through different n_embed values
for n_embed in "${n_embeds[@]}"; do
    save_path="dependent/ETTh2_${n_embed}_entropy_dist"  # Fixed string interpolation
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
        --chan_indep 1 \
        --enc_in 7 \
        --d_model 64 \
        --block_num $block_num \
        --dropout 0.2 \
        --num_epoch 50 \
        --eval_per_epoch \
        --save_path "$save_path" \
        --lr 0.0003 > "logs/Tokenizer/${model_id_name}_${token_len}_emb${n_embed}_wl${wave_length}_bl${block_num}_${vq_model}_depent_entropy_dist.log"
done