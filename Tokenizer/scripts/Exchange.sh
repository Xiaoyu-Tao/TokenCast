if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Tokenizer" ]; then
    mkdir ./logs/Tokenizer
fi

export CUDA_VISIBLE_DEVICES="3"

source /data/tingyue/anaconda3/bin/activate DD-Time

root_path_name=../datas/dataset/exchange_rate
data_path_name=exchange_rate.csv
model_id_name=exchange_rate
data_name=custom

n_embed=256
wave_length=8
token_len=16
vq_model='SimVQ' # VanillaVQ, SimVQ

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
    --chan_indep 1\
    --enc_in 8 \
    --d_model 64 \
    --block_num 2 \
    --dropout 0.2 \
    --num_epoch 1 \
    --eval_per_epoch \
    --lr 0.001 >logs/Tokenizer/$model_id_name'_'$token_len'_emb'$n_embed'_wl'$wave_length'_'$vq_model''.log