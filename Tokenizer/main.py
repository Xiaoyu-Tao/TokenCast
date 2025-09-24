import json
import torch
import random
import os
import numpy as np
import warnings

warnings.filterwarnings('ignore')
from data_provider.data_factory import data_provider
from args import args
from process import Trainer
from models.VQVAE import VQVAE
from models.W_SimVQ import W_SimVQ
from models.W_SimVQ_CNN import W_SimVQ_CNN
from models.W_InstructTimeVQ import W_InstructTimeVQ
from models.ResidualVQ_tcn_enc import VQVAE as ResidualVQ
# from dataset import Dataset
import torch.utils.data as Data

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

def get_data(flag):
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader

def main():
    seed_everything(seed=2024)


    train_data, train_loader = get_data(flag='train')
    vali_data, vali_loader = get_data(flag='val')
    test_data, test_loader = get_data(flag='test')

    print('dataset initial ends')

    
    if args.vq_model == 'SimVQ':
        model = W_SimVQ(args)
    elif args.vq_model == 'VanillaVQ':
        model = W_InstructTimeVQ(args)
    elif args.vq_model == 'SimVQ_CNN':
        model = W_SimVQ_CNN(args)
    elif args.vq_model == 'ResidualVQ':
        model = ResidualVQ(args)
    else:
        raise ValueError('Invalid VQ model name')
    
    
    print('model initial ends')

    trainer = Trainer(args, model, train_loader, vali_loader, test_loader, verbose=True)
    print('trainer initial ends')

    if args.is_training:
        trainer.train()

    trainer.test()


if __name__ == '__main__':
    main()

