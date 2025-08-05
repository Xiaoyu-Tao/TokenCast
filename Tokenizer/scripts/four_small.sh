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
# source /data/tingyue/anaconda3/bin/activate DD-Time

root_path_name=/data/tinyy/first/CrossTimeNet/dataset
data_path_name=Covid-19.csv
model_id_name=Covid-19
data_name=Covid-19

wave_length=4
seq_len=36
token_len=96
pred_len=60
n_embeds=(64)
vq_model='ResidualVQ' # VanillaVQ, SimVQ

block_num=2
for n_embed in "${n_embeds[@]}"; do
save_path="dependent/${data_name}_${n_embed}_entropy_dist_0.1_entropy_temp_0.5"
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
        --enc_in 948 \
        --pred_len $pred_len \
        --d_model 64 \
        --block_num $block_num \
        --dropout 0.2 \
        --num_epoch 30 \
        --eval_per_epoch \
        --train_batch_size 4 \
        --test_batch_size 4 \
        --save_path $save_path \
        --lr 0.0001 >logs/Tokenizer/$model_id_name'_'$token_len'_emb'$n_embed'_wl'$wave_length'_bl'$block_num'_'$vq_model''.log
done


root_path_name=/data/tinyy/first/CrossTimeNet/dataset
data_path_name=FRED-MD.csv
model_id_name=FRED-MD
data_name=FRED-MD

wave_length=4
seq_len=36
token_len=96
pred_len=60
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
        --chan_indep 0 \
        --enc_in 107 \
        --pred_len $pred_len \
        --d_model 64 \
        --block_num $block_num \
        --dropout 0.2 \
        --num_epoch 30 \
        --eval_per_epoch \
        --train_batch_size 4 \
        --test_batch_size 4 \
        --lr 0.0001 >logs/Tokenizer/$model_id_name'_'$token_len'_emb'$n_embed'_wl'$wave_length'_bl'$block_num'_'$vq_model''.log
done


root_path_name=/data/tinyy/first/CrossTimeNet/dataset
data_path_name=NYSE.csv
model_id_name=NYSE
data_name=NYSE

wave_length=4
seq_len=36
token_len=96
pred_len=60
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
        --chan_indep 0 \
        --enc_in 5 \
        --pred_len $pred_len \
        --d_model 64 \
        --block_num $block_num \
        --dropout 0.2 \
        --num_epoch 30 \
        --eval_per_epoch \
        --train_batch_size 4 \
        --test_batch_size 4 \
        --lr 0.0001 >logs/Tokenizer/$model_id_name'_'$token_len'_emb'$n_embed'_wl'$wave_length'_bl'$block_num'_'$vq_model''.log
done

root_path_name=/data/tinyy/first/CrossTimeNet/dataset
data_path_name=Wike2000.csv
model_id_name=Wike2000
data_name=Wike2000

wave_length=4
seq_len=36
token_len=96
pred_len=60
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
        --chan_indep 0 \
        --enc_in 2000 \
        --pred_len $pred_len \
        --d_model 64 \
        --block_num $block_num \
        --dropout 0.2 \
        --num_epoch 30 \
        --eval_per_epoch \
        --train_batch_size 4 \
        --test_batch_size 4 \
        --lr 0.0001 >logs/Tokenizer/$model_id_name'_'$token_len'_emb'$n_embed'_wl'$wave_length'_bl'$block_num'_'$vq_model''.log
done