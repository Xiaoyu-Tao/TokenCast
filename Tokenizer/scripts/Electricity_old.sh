if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Tokenizer" ]; then
    mkdir ./logs/Tokenizer
fi

export CUDA_VISIBLE_DEVICES="2"

root_path_name=/data/tinyy/vqvae/dataset
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

token_len=12
seq_len=96
pred_len=144
vq_model='ResidualVQ' # VanillaVQ, SimVQ

# 你可以根据需要修改下面的参数列表
wave_lengths=(4 8 2)
n_embeds=(128 256 512 1024)
d_models=(64 32)

for wave_length in "${wave_lengths[@]}"; do
  for n_embed in "${n_embeds[@]}"; do
    for d_model in "${d_models[@]}"; do
    save_path="independent/Electricity_${wave_length}_${n_embed}_${d_model}"
      log_name=${model_id_name}_tok${token_len}_emb${n_embed}_wl${wave_length}_dm${d_model}_bl2_${vq_model}_${pred_len}_seq_len${seq_len}_independent.log
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
        --enc_in 321 \
        --d_model $d_model \
        --block_num 2 \
        --dropout 0.2 \
        --num_epoch 50 \
        --lr 0.0005 \
        --chan_indep 0 \
        --save_path $save_path \
        > logs/Tokenizer/$log_name
    done
  done
done
