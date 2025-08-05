import os
import time
import torch
import pickle
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from loss import MSE
from torch.optim.lr_scheduler import LambdaLR

from utils.tools import plot_token_distribution, plot_token_distribution_with_stratify
from utils.tools import plot_and_save_reconstruction, plot_PCA, statistic_freqs

class Trainer():
    def __init__(self, args, model, train_loader, vali_loader, test_loader, verbose=False):
        self.args = args
        self.verbose = verbose
        self.device = args.device
        self.print_process(self.device)
        self.model = model.to(torch.device(self.device))

        self.train_loader = train_loader
        self.vali_loader = vali_loader
        self.test_loader = test_loader
        
        self.lr_decay = args.lr_decay_rate
        self.lr_decay_steps = args.lr_decay_steps
        self.weight_decay = args.weight_decay
        self.model_name = self.model.get_name()
        self.print_process(self.model_name)

        self.cr = MSE(self.model)

        self.num_epoch = args.num_epoch
        self.eval_per_steps = args.eval_per_steps
        self.save_path = args.save_path
        
        if args.load_path is not None:
            self.load_path = args.load_path  
        else:
            self.load_path = args.save_path
        
        if self.num_epoch:
            self.result_file = open(self.save_path + '/result.txt', 'w')
            self.result_file.close()

        self.step = 0
        self.best_metric = 1e9
        self.metric = 'reconst_mse'

    def train(self):
        self.print_process('\n######### Start Training #########')
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.weight_decay)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: self.lr_decay ** step, verbose=self.verbose)
        for epoch in range(self.num_epoch):
            loss_epoch, time_cost = self._train_one_epoch(epoch)
            self.result_file = open(self.save_path + '/result.txt', 'a+')
            self.print_process(
                'Basic Model train epoch:{0}, loss:{1:.6f}, training_time:{2:.6f}'.format(epoch + 1, loss_epoch, time_cost))
            print('Basic Model train epoch:{0}, loss:{1:.6f}, training_time:{2:.6f}'.format(epoch + 1, loss_epoch, time_cost),
                  file=self.result_file)
            self.result_file.close()
        self.print_process(self.best_metric)
        return self.best_metric
      
    def _eval(self, epoch):
        metric_dict = {}
        for key in ['train', 'valid', 'test']:
            if key == 'train': data_loader = self.train_loader
            elif key == 'valid': data_loader = self.vali_loader
            elif key == 'test': data_loader = self.test_loader
            
            _metric = self.eval_model_vqvae(data_loader)
            metric_dict[key] = _metric
            
            print(f'{key}: ', end='')
            self.print_process(_metric)
            
        print('\n')
        
        metric = metric_dict['valid']
        self.result_file = open(self.save_path + '/result.txt', 'a+')
        
        if not self.args.eval_per_epoch:
            print('step{0}'.format(self.step), file=self.result_file)
        else:
            print('epoch{0}'.format(epoch), file=self.result_file)
            
        print(metric, file=self.result_file)
        self.result_file.close()
        
        print(self.metric, metric[self.metric], self.best_metric)
        
        if metric[self.metric] < self.best_metric:
            self.model.eval()
            torch.save(self.model.state_dict(), self.save_path + '/model.pkl')
            self.result_file = open(self.save_path + '/result.txt', 'a+')
            
            if not self.args.eval_per_epoch:
                print('best model saved at step{0}'.format(self.step))
            else:
                print('best model saved at epoch{0}'.format(epoch))
            
            self.result_file.close()
            self.best_metric = metric[self.metric]
            
        self.model.train()
        
    def _get_all_ids(self, data_loader):
        mse = nn.MSELoss()
        total_recon_loss = 0.0
        total_batches = 0

        # get test token distribution and calculate mse
        ids = []
        with torch.no_grad():
            for idx, (batch_x, batch_y, _,_) in enumerate(data_loader):
                batch_y = batch_y[:, -self.args.pred_len:, :].float().to(self.args.device)
                seqs_x = batch_x.float().to(self.args.device)
                out_x, _, id_x = self.model(seqs_x,batch_y)
                
                ids.append(id_x.flatten())
                seqs_x = torch.cat([seqs_x,batch_y],dim=1)
                recon_loss = mse(out_x, seqs_x)

                total_recon_loss += recon_loss.item()
                total_batches += 1
        
        ids = torch.cat(ids).cpu().numpy()
        return ids

    def _train_one_epoch(self, epoch):
        t0 = time.perf_counter()
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader) if self.verbose else self.train_loader
        loss_sum = 0
        for idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm_dataloader):
            self.optimizer.zero_grad()
            batch_y = batch_y[:, -self.args.pred_len:, :].float().to(self.args.device)
            loss = self.cr.compute(batch_x.float().to(self.args.device),batch_y)
            loss_sum += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            self.step += 1
            if self.step % self.lr_decay_steps == 0:
                self.scheduler.step()
                
                
            if (self.step % self.eval_per_steps == 0) and (self.args.eval_per_epoch == False):
                self._eval(epoch)
                
        if self.args.eval_per_epoch:
            self._eval(epoch)
            
            # plot the distribution of train and test tokens
            train_ids = self._get_all_ids(self.train_loader)
            test_ids = self._get_all_ids(self.test_loader)
            
            plot_path = os.path.join(self.load_path, 'token_distribution_epoch{}'.format(epoch))
            
            plot_token_distribution_with_stratify(train_ids, test_ids, \
                save_dir=plot_path, max_token_num=self.args.n_embed, freq=True)

        return loss_sum / len(self.train_loader), time.perf_counter() - t0

    def eval_model_vqvae(self, data_loader):
        self.model.eval()
        tqdm_data_loader = tqdm(data_loader) if self.verbose else data_loader
        metrics = {'reconst_mse': 0, 'latent_mse': 0}

        with torch.no_grad():
            for idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm_data_loader):
                batch_y = batch_y[:, -self.args.pred_len:, :].float().to(self.args.device)
                loss_dict = self.cr.compute(batch_x.float().to(self.args.device), batch_y,details=True)
                metrics['reconst_mse'] += loss_dict['recon_loss']
                metrics['latent_mse'] += loss_dict['latent_loss']
                
        metrics['reconst_mse'] /= len(data_loader)
        metrics['latent_mse'] /= len(data_loader)
        
        return metrics
    
    def print_process(self, *x):
        if self.verbose:
            print(*x)

    def test(self):
        self.print_process('\n######### Start Testing #########')
        
        state_dict = torch.load(os.path.join(self.load_path, 'model.pkl'), map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # plot the low-dimensional representation of the code book
        
        # get reconst mse
        mse = nn.MSELoss()
        total_recon_loss = 0.0
        total_batches = 0
        mae = nn.L1Loss()

        # get train token distribution
        train_ids = []
        with torch.no_grad():
            for idx, (batch_x, batch_y, _, _) in enumerate(self.train_loader):
                batch_y = batch_y[:, -self.args.pred_len:, :].float().to(self.args.device)
                seqs_x = batch_x.float().to(self.args.device)
                _, _, id_x = self.model(seqs_x,batch_y)
                
                train_ids.append(id_x.flatten())
                
        if False:
            train_ids = torch.cat(train_ids).cpu().numpy()
            train_tokens = train_ids.flatten()
            train_uni_elements,  train_cnts_elements = np.unique(train_tokens, return_counts=True)
            
            statistic_freqs(train_tokens)
            
            total_nums = len(train_tokens)
            statis = 0
            board = total_nums * (statis / 100.)
            
            elect_index = np.where(train_cnts_elements >= board)
            elect_ids = train_uni_elements[elect_index]
            self.model.elect_codebook(elect_ids, statis)
            
        # get test token distribution and calculate mse
        test_ids = []
        total_mse_input = 0.0
        total_mae_input = 0.0
        total_mse_output = 0.0
        total_mae_output = 0.0

        total_batches = 0

        with torch.no_grad():
            for idx, (batch_x, batch_y, _, _) in enumerate(self.test_loader):
                batch_y = batch_y[:, -self.args.pred_len:, :].float().to(self.args.device)
                seqs_x = batch_x.float().to(self.args.device)
                out_x, _, id_x = self.model(seqs_x,batch_y)
                
                test_ids.append(id_x.flatten())
                # 拼接 ground truth
                full_target = torch.cat([seqs_x, batch_y], dim=1)
                # 对应输出长度
                input_len = seqs_x.size(1)
                total_len = full_target.size(1)

                # 分别计算 MSE/MAE
                mse_input = mse(out_x[:, :input_len, :], full_target[:, :input_len, :])
                mae_input = mae(out_x[:, :input_len, :], full_target[:, :input_len, :])

                mse_output = mse(out_x[:, input_len:, :], full_target[:, input_len:, :])
                mae_output = mae(out_x[:, input_len:, :], full_target[:, input_len:, :])

                # 累加
                total_mse_input += mse_input.item()
                total_mae_input += mae_input.item()
                total_mse_output += mse_output.item()
                total_mae_output += mae_output.item()

                total_batches += 1

        # 平均值
        avg_mse_input = total_mse_input / total_batches
        avg_mae_input = total_mae_input / total_batches
        avg_mse_output = total_mse_output / total_batches
        avg_mae_output = total_mae_output / total_batches

        print('[Input Part]  MSE: {:.6f}, MAE: {:.6f}'.format(avg_mse_input, avg_mae_input))
        print('[Output Part] MSE: {:.6f}, MAE: {:.6f}'.format(avg_mse_output, avg_mae_output))
                        
        # plot the distribution of train and test tokens

        
        plot_path = os.path.join(self.load_path, 'token_distribution')
        
        test_ids = torch.cat(test_ids).cpu().numpy()
        
        # print the statistics of the token distribution
        # 
        
        codebook_plot_path = os.path.join(self.load_path, 'codebook_with_used_freqs.png')
        # codebook = self.model.get_codebook_weight()
        # plot_PCA(train_ids, codebook, codebook_plot_path, max_token_num=self.args.n_embed)

        # exit(0)
        train_ids = self._get_all_ids(self.train_loader)
        
        statistic_freqs(train_ids.flatten())
        # test_ids = self._get_all_ids(self.test_loader)
        plot_token_distribution_with_stratify(train_ids, test_ids, \
            save_dir=plot_path, max_token_num=self.args.n_embed)
        
        # count the frequence of train tokens
        freq = np.bincount(train_ids, minlength=self.args.n_embed)
        fixed_freq = np.where(freq > 0, freq, 1e-7)
        
        print(len(freq))
        
        n_classes = len(set(train_ids))
        weight = len(train_ids) / (n_classes * fixed_freq)
        scale = 5.0  # 控制最终平均权重为多少（可调，建议 2~5）
        # weight = weight / np.mean(weight)  # 归一化为均值为 1
        weight = weight * scale   
        
        mask = freq > 0
        train_tokens = train_ids.flatten()
        train_uni_elements,  train_cnts_elements = \
            np.unique(train_tokens, return_counts=True)
            
        weight_dict = {
            'weight': weight,
            'mask': mask,
            'train_uni_elements': train_uni_elements,
            'train_cnts_elements': train_cnts_elements,
            'total_nums': len(train_ids)
        }
        
        print("Successfully save weight.pkl")
        
        save_w_path = os.path.join(self.load_path, 'weight.pkl')
        pickle.dump(weight_dict, open(save_w_path, 'wb'))
        plot_path = os.path.join(self.load_path, 'reconstruction')
        os.makedirs(plot_path, exist_ok=True)
        
        plot_and_save_reconstruction(self.model, self.test_loader, plot_path)
        print("Images have been saved.")
        
        exit(0)
        
        # Just calculate the minimun weight from existing tokens
        
        # print((freq > 0).shape)
        
        # real_min_weight = np.min(weight, where=(freq > 0), initial=np.inf)
        # max_weight = real_min_weight * 20
        # weight = np.clip(weight, a_min=None, a_max=max_weight)
        
        # print("#### Weight Statistics: ####")
        # print(weight.shape, max(weight), min(weight)) # min:0.11 max: 647
        

        
        print("#### Token Distribution Analysis ####")
        print("Training Set: Used token is {}, Total token is {}".format(len(set(train_ids)), self.args.n_embed))
        print("Test Set: Used token is {}, Total token is {}".format(len(set(test_ids)), self.args.n_embed))


        avg_recon_loss = total_recon_loss / total_batches
        
        print('reconstruct loss(mse) on test dataset: {:.6f}\n'.format(avg_recon_loss))

        
            