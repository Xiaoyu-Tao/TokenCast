import argparse
import os
import json

parser = argparse.ArgumentParser()
# basic config
parser.add_argument("--is_training", type=int, default=1)
parser.add_argument("--save_path", type=str, default=None)
parser.add_argument("--load_path", type=str, default=None)

# dataset
parser.add_argument('--data', type=str, required=False, default='ETTh1', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')

# dataloader
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--token_len', type=int, default=336, help='input sequence length')
parser.add_argument('--percent', type=int, default=100)
# seq_len
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='label sequence length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')

parser.add_argument('--root_path', type=str, default='./dataset/ETT-small', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--test_batch_size", type=int, default=64)

parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument('--use_fullmodel', type=int, default=0, help='use full model or just encoder')
parser.add_argument('--use_closedllm', type=int, default=0, help='use closedllm or not') 
parser.add_argument('--text_len', type=int, default=4)

# model args
parser.add_argument("--d_model", type=int, default=64)
parser.add_argument("--enc_in", type=int, default=21)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--block_num", type=int, default=2)

parser.add_argument("--n_embed", type=int, default=256)
parser.add_argument("--wave_length", type=int, default=7)
parser.add_argument("--chan_indep", type=int, default=0, help="independent channels")

parser.add_argument("--vq_model", type=str, default='SimVQ', help='options:[SimVQ, VanillaVQ, SimVQ_CNN]')

# Revin
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')


# train args
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--lr_decay_rate", type=float, default=0.99)
parser.add_argument("--lr_decay_steps", type=int, default=300)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--num_epoch", type=int, default=60)
parser.add_argument("--eval_per_steps", type=int, default=300)
parser.add_argument("--device", type=str, default="cuda:0")

parser.add_argument("--eval_per_epoch", action="store_true")
parser.add_argument('--multi_dataset', action='store_true', help='Enable multi-dataset joint training')
parser.add_argument('--entropy_penalty', type=float, default=0.1, help='Penalty weight for entropy regularization')
parser.add_argument('--entropy_temp', type=float, default=0.5, help='Temperature for soft assignment in VQ')

args = parser.parse_args()

# Train_data,Test_data = load_ETT(Path="/data/tinyy/vqvae1/dataset/ETT-small",folder=args.data)

args.dataset = args.data_path.split('.')[0]

vq_setting = "unfreeze_codebook"
if args.save_path is None:
    path_str = 'checkpoints//{}_{}_dm{}_dr{}_emb{}_wl{}_bl{}_{}_{}'.format(
            args.dataset,
            args.token_len,
            args.d_model,
            args.dropout,
            args.n_embed,
            args.wave_length,
            args.block_num,
            args.vq_model,
            vq_setting
            )
    
    args.save_path = path_str
    
if not os.path.exists(args.save_path):
    print("Creating save dir: {}".format(args.save_path))
    os.makedirs(args.save_path)

with open(args.save_path + "/args.json", "w") as f:
    tmp = args.__dict__
    json.dump(tmp, f, indent=1)
    print(args)
