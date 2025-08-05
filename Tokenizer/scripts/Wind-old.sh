if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Tokenizer" ]; then
    mkdir ./logs/Tokenizer
fi

export CUDA_VISIBLE_DEVICES="3"

# source /data/tingyue/anaconda3/bin/activate DD-Time

root_path_name=/data/tinyy/first/CrossTimeNet/dataset
data_path_name=Wind.csv
model_id_name=Wind
data_name=Wind

wave_length=4
seq_len=96
token_len=96
pred_len=144
n_embeds=(256 512)
vq_model='ResidualVQ' # VanillaVQ, SimVQ

block_num=2
for n_embed in "${n_embeds[@]}"; do
save_path="dependent/${data_name}_${n_embed}_entropy_dist"
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
        --pred_len $pred_len \
        --d_model 64 \
        --block_num $block_num \
        --dropout 0.2 \
        --num_epoch 30 \
        --eval_per_epoch \
        --train_batch_size 256 \
        --test_batch_size 256 \
        --lr 0.001 >logs/Tokenizer/$model_id_name'_'$token_len'_emb'$n_embed'_wl'$wave_length'_bl'$block_num'_'$vq_model''.log
done