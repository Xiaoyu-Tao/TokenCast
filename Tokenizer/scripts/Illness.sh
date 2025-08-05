if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Tokenizer" ]; then
    mkdir ./logs/Tokenizer
fi

export CUDA_VISIBLE_DEVICES="2"

source /data/tingyue/anaconda3/bin/activate DD-Time

root_path_name=../dataset/illness/
data_path_name=national_illness.csv
model_id_name=national_illness
data_name=custom

n_embed=32
wave_length=6
token_len=12
vq_model='SimVQ_CNN' # VanillaVQ, SimVQ

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
    --enc_in 7 \
    --d_model 64 \
    --block_num 2 \
    --dropout 0.2 \
    --num_epoch 1 \
    --eval_per_epoch \
    --lr 0.0005 >logs/Tokenizer/$model_id_name'_'$token_len'_emb'$n_embed'_wl'$wave_length'_'$vq_model''.log