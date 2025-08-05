mkdir -p ./exp_PEMS08/logs
mkdir -p ./exp_PEMS08/checkpoints


export CUDA_VISIBLE_DEVICES="1"
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
source ~/.bashrc
source /data/tinyy/miniconda3/bin/activate LLM


root_path_name=/data/tinyy/first/CrossTimeNet/dataset
data_path_name=PEMS08.csv
model_id_name=PEMS08
data_name=PEMS

wave_length=4
seq_len=144
token_len=96
pred_len=144
vq_model='ResidualVQ'
block_num=2

n_embeds=(128 256)
entropy_penalties=(0.2 0.05 0.1)
entropy_temps=(1.0 0.3 0.5 0.8)
d_models=(128 256)

# 多参数组合实验
for d_model in "${d_models[@]}"; do
    for n_embed in "${n_embeds[@]}"; do
        for entropy_penalty in "${entropy_penalties[@]}"; do
            for entropy_temp in "${entropy_temps[@]}"; do

                tag="${data_name}_emb${n_embed}_d${d_model}_pen${entropy_penalty}_temp${entropy_temp}"
                save_path="./exp_PEMS08/checkpoints/${tag}"
                log_file="./exp_PEMS08/logs/${tag}.log"

                mkdir -p $save_path

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
                    --enc_in 170 \
                    --pred_len $pred_len \
                    --d_model $d_model \
                    --block_num $block_num \
                    --dropout 0.2 \
                    --num_epoch 30 \
                    --eval_per_epoch \
                    --train_batch_size 128 \
                    --test_batch_size 128 \
                    --entropy_penalty $entropy_penalty \
                    --entropy_temp $entropy_temp \
                    --save_path $save_path \
                    --lr 0.0001 \
                    > $log_file

            done
        done
    done
done