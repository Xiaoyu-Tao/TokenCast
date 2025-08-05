import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from collections.abc import Iterable
from sklearn.decomposition import PCA

plt.switch_backend('agg')

def plot_token_distribution(train_tokens: torch.Tensor, test_tokens: torch.Tensor, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    
    _train_tokens = train_tokens.flatten()
    _test_tokens = test_tokens.flatten()
    
    # 使用 np.unique 获取数组中每个元素的出现次数
    train_uni_elements,  train_cnts_elements = np.unique(_train_tokens, return_counts=True)
    test_uni_elements,  test_cnts_elements = np.unique(_test_tokens, return_counts=True)

    plt.clf()

    # 绘制 Groundtruth 的 Token 分布
    plt.bar(train_uni_elements, train_cnts_elements, label='Train')
    plt.xlabel('Token ID')
    plt.ylabel('Token Count')
    plt.title('Token Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'train_token_distribution.png'))
    
    plt.clf()
    
    # 绘制 Prediction 的 Token 分布
    plt.bar(test_uni_elements, test_cnts_elements, label='Test')
    plt.xlabel('Token ID')
    plt.ylabel('Token Count')
    plt.title('Token Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'test_token_distribution.png'))
    
    plt.clf()
    
    # 绘制 Groundtruth 和 Prediction 的 Token 分布
    plt.bar(train_uni_elements, train_cnts_elements, label='Train')
    plt.bar(test_uni_elements, test_cnts_elements, label='Test')
    plt.xlabel('Token ID')
    plt.ylabel('Token Count')
    plt.title('Token Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'train_test_token_distribution.png'))
    
    plt.clf()
    
def plot_PCA(train_ids, X, save_path, max_token_num):
    # calculate the frequency of each token
    train_tokens = train_ids.flatten()
    train_uni_elements,  train_cnts_elements = np.unique(train_tokens, return_counts=True)
    train_cnts = np.zeros((max_token_num, ))
    train_cnts[train_uni_elements] = train_cnts_elements
    
    mask = np.where(train_cnts > 0)
    X = X[mask]
    train_cnts = train_cnts[mask]
    
    print(train_cnts)
    
    pca = PCA(n_components=2)
    X_r = pca.fit_transform(X)
    
    X, y = X_r[:, 0], X_r[:, 1]
    weights = train_cnts

    scatter = plt.scatter(X, y, c=weights, cmap='hot')
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA with weights')
    
    # 添加颜色条
    plt.colorbar(scatter)
    
    plt.savefig(save_path)
    
    exit(0)

def statistic_freqs(train_ids):
    train_tokens = train_ids.flatten()
    train_uni_elements,  train_cnts_elements = np.unique(train_tokens, return_counts=True)
    
    total_nums = len(train_tokens)
    statis_list = [10,5,2,1.5,1.2,1,0.8,0.7,0.6,0.5,0.2,0.1]
    
    for statis in statis_list:
        board = total_nums * (statis / 100.)
        print(f'Freqs large than {statis}%: {np.sum(train_cnts_elements >= board)}')
        
    return 

def plot_token_distribution_with_stratify(train_tokens: torch.Tensor, test_tokens: torch.Tensor, \
                            save_dir: str, max_token_num=255, freq=True):
    
    os.makedirs(save_dir, exist_ok=True)
    
    _train_tokens = train_tokens.flatten()
    _test_tokens = test_tokens.flatten()
    
    # 使用 np.unique 获取数组中每个元素的出现次数
    train_uni_elements,  train_cnts_elements = np.unique(_train_tokens, return_counts=True)
    test_uni_elements,  test_cnts_elements = np.unique(_test_tokens, return_counts=True)
    
    if freq:
        train_cnts_elements = train_cnts_elements / len(_train_tokens)
        test_cnts_elements = test_cnts_elements / len(_test_tokens)

    plt.clf()

    # 绘制 Groundtruth 的 Token 分布
    plt.bar(train_uni_elements, train_cnts_elements, label='Train')
    plt.xlabel('Token ID')
    plt.ylabel('Token Count')
    plt.title('Token Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'train_token_distribution.png'))
    
    plt.clf()
    
    # 绘制 Prediction 的 Token 分布
    plt.bar(test_uni_elements, test_cnts_elements, label='Test')
    plt.xlabel('Token ID')
    plt.ylabel('Token Count')
    plt.title('Token Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'test_token_distribution.png'))
    
    plt.clf()
    
    # 绘制 Groundtruth 和 Prediction 的 Token 分布
    train_cnts = np.zeros((max_token_num, ))
    train_cnts[train_uni_elements] = train_cnts_elements
    
    test_cnts = np.zeros((max_token_num, ))
    test_cnts[test_uni_elements] = test_cnts_elements
    
    data1, data2 = train_cnts, test_cnts
    
    data_low = [min(d1, d2) for d1, d2 in zip(data1, data2)]
    data_high = [max(d1, d2) for d1, d2 in zip(data1, data2)]

    colors_low = ['blue' if d1 < d2 else 'orange' for d1, d2 in zip(data1, data2)]
    colors_high = ['orange' if d1 < d2 else 'blue' for d1, d2 in zip(data1, data2)]

    # 设置横坐标
    x = np.arange(len(data1))

    # print(x, data_low, data_high)

    # 绘制柱状图
    plt.bar(x, data_low, color=colors_low, label='Test')
    plt.bar(x, data_high, bottom=data_low, color=colors_high, label='Train') 

    plt.xlabel('Token ID')
    plt.ylabel('Token Count')
    plt.title('Token Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'train_test_token_distribution.png'))
    
    plt.clf()


def clever_format(nums, format="%.2f"):
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []

    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + "T")
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + "G")
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + "M")
        elif num > 1e3:
            clever_nums.append(format % (num / 1e3) + "K")
        else:
            clever_nums.append(format % num + "B")

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums,)

    return clever_nums


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint_96.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def plot_and_save_reconstruction(model, test_loader, save_path, dims_to_plot=None):
    model.eval()
    test_data_iter = iter(test_loader)
    batch_x, batch_y, _, _ = next(test_data_iter)

    sample_index = 0
    sample_data_x = batch_x[sample_index].unsqueeze(0).float().to(next(model.parameters()).device)
    sample_data_y = batch_y[sample_index].unsqueeze(0).float().to(next(model.parameters()).device)

    with torch.no_grad():
        reconstructed, _, _ = model(sample_data_x, sample_data_y)

    original_data = sample_data_y.squeeze(0).cpu().numpy()
    reconstructed_data = reconstructed.squeeze(0)[-sample_data_y.size(1):].cpu().numpy()

    # 自动判断是否为单变量时间序列
    if original_data.ndim == 1 or (original_data.ndim == 2 and original_data.shape[1] == 1):
        original_data = original_data.squeeze()
        reconstructed_data = reconstructed_data.squeeze()

        plt.figure(figsize=(12, 4))
        plt.plot(original_data, label='Original')
        plt.plot(reconstructed_data, label='Reconstructed')
        plt.title("Single-variable Reconstruction Comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'single_variable_reconstruction.pdf'))
        plt.close()
    else:
        # 多变量情况：默认绘制前几个维度
        num_dims = original_data.shape[1]
        if dims_to_plot is None:
            dims_to_plot = list(range(min(4, num_dims)))

        fig, axes = plt.subplots(len(dims_to_plot), 1, figsize=(12, len(dims_to_plot) * 2))
        for idx, dim in enumerate(dims_to_plot):
            axes[idx].plot(original_data[:, dim], label=f'Original Dim {dim}')
            axes[idx].plot(reconstructed_data[:, dim], label=f'Recon Dim {dim}')
            axes[idx].set_title(f"Data Comparison - Dim {dim}")
            axes[idx].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'multi_variable_dim_comparison.pdf'))
        plt.close()

        # 总体重建图
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        for dim in dims_to_plot:
            ax[0].plot(original_data[:, dim], label=f'Dim {dim}')
            ax[1].plot(reconstructed_data[:, dim], label=f'Dim {dim}')
        ax[0].set_title("Original Data - Selected Dims")
        ax[1].set_title("Reconstructed Data - Selected Dims")
        ax[0].legend()
        ax[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'multi_variable_total_reconstruction.pdf'))
        plt.close()