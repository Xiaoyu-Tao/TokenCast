if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Tokenizer" ]; then
    mkdir ./logs/Tokenizer
fi

export CUDA_VISIBLE_DEVICES="1"

# source /data/tingyue/anaconda3/bin/activate DD-Time

root_path_name=/data/tinyy/vqvae/dataset
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom
token_len=16
seq_len=96
pred_len=144
vq_model='ResidualVQ' # VanillaVQ, SimVQ

# Define arrays for n_embed and wave_length values to test
n_embeds=(128)
wave_lengths=(4)

for n_embed in "${n_embeds[@]}"; do
    for wave_length in "${wave_lengths[@]}"; do
        save_path="independent/traffic_${wave_length}_${n_embed}"
        log_name="${model_id_name}_tok${token_len}_emb${n_embed}_wl${wave_length}_dm64_bl6_${vq_model}_${pred_len}_seq_len${seq_len}_independent_${n_embed}.log"
        
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
            --enc_in 862 \
            --d_model 256 \
            --block_num 6 \
            --dropout 0.2 \
            --num_epoch 50 \
            --lr 5e-4 \
            --chan_indep 0 \
            --train_batch_size 8 \
            --test_batch_size 8 \
            --save_path $save_path \
            > logs/Tokenizer/$log_name
    done
done