import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from collections.abc import Iterable
from torch.optim.lr_scheduler import LambdaLR
import math
import traceback
from peft import PeftModel, get_peft_model
plt.switch_backend('agg')

def is_peft_model(model):
    return isinstance(model, PeftModel)

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


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print('Updating learning rate to {}'.format(lr))
            else:
                print('Updating learning rate to {}'.format(lr))


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))  # 线性 warmup
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1. + math.cos(math.pi * progress))  # cosine decay
    return LambdaLR(optimizer, lr_lambda)

class EarlyStopping:
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True,test_fn=None):
        """
        Args:
            accelerator: HuggingFace Accelerator 实例（可选）
            patience (int): 当验证集 loss 多久没有提升就早停
            verbose (bool): 是否打印日志
            delta (float): 最小改进值，避免浮动误判
            save_mode (bool): 是否保存最优模型
        """
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_mode = save_mode
        self.test_fn = test_fn  # ✅ 显式初始化

    def __call__(self, val_loss, model, path):
        """
        每个 epoch 验证之后调用
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)

        elif score < self.best_score + self.delta:
            self.counter += 1  # ✅ 修复漏加
            if self.accelerator:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """
        保存当前验证集 loss 最小的模型
        """
        if self.verbose:
            msg = f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...'
            if self.accelerator:
                self.accelerator.print(msg)
            else:
                print(msg)

        # ✅ 拼接保存路径（加后缀）
        save_path = os.path.join(path, 'checkpoint.pth')

        # ✅ unwrap model（如果是多卡包裹的模型）
        model_to_save = self.accelerator.unwrap_model(model) if self.accelerator else model

        # ✅ 判断是否是 PEFT 模型（LoRA）
        if is_peft_model(model_to_save):
            # 保存的是 LoRA adapter 权重（非全模型）
            save_path = os.path.join(path, "lora_adapter")
            model_to_save.save_pretrained(save_path)
            if self.accelerator:
                self.accelerator.print(f"[LoRA] Adapter saved to: {save_path}")
            else:
                print(f"[LoRA] Adapter saved to: {save_path}")
        else:
            # 保存的是普通全模型
            save_path = os.path.join(path, "checkpoint.pth")
            torch.save(model_to_save.state_dict(), save_path)
            if self.accelerator:
                self.accelerator.print(f"[Full] Model state_dict saved to: {save_path}")
            else:
                print(f"[Full] Model state_dict saved to: {save_path}")

        self.val_loss_min = val_loss
        # ✅ 测试 callback
        if self.test_fn is not None:
            try:
                self.accelerator.print("[🚀] Testing saved model after val improvement...") if self.accelerator else print("[🚀] Testing...")
                self.test_fn()
            except Exception as e:
                if self.accelerator:
                    self.accelerator.print(f"[❌] Test failed after saving best model: {e}")
                    self.accelerator.print(traceback.format_exc())  # ⬅ 打印堆栈
                else:
                    print(f"[❌] Test failed after saving best model: {e}")
                    print(traceback.format_exc())  # ⬅ 打印堆栈


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
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    
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


def plot_token_distribution_with_stratify(gt_tokens: torch.Tensor, pred_tokens: torch.Tensor, \
                save_dir: str, max_token_num=255, dataset='test', freq=False):
    
    os.makedirs(save_dir, exist_ok=True)
    
    _gt_tokens = gt_tokens.flatten().detach().cpu().numpy()
    _pred_tokens = pred_tokens.flatten().detach().cpu().numpy()
    
    # 使用 np.unique 获取数组中每个元素的出现次数
    gt_uni_elements, gt_cnts_elements = np.unique(_gt_tokens, return_counts=True)
    pred_uni_elements, pred_cnts_elements = np.unique(_pred_tokens, return_counts=True)
    
    if freq:
        gt_cnts_elements = gt_cnts_elements / gt_cnts_elements.sum()
        pred_cnts_elements = pred_cnts_elements / pred_cnts_elements.sum()

    plt.clf()

    # 绘制 Groundtruth 的 Token 分布
    plt.bar(gt_uni_elements, gt_cnts_elements, label='GroundTruth')
    plt.xlabel('Token ID')
    plt.ylabel('Token Count')
    plt.title('Token Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'gt_token_distribution_on_{dataset}.png'))
    
    plt.clf()
    
    # 绘制 Prediction 的 Token 分布
    plt.bar(pred_uni_elements, pred_cnts_elements, label='Prediction')
    plt.xlabel('Token ID')
    plt.ylabel('Token Count')
    plt.title('Token Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'pred_token_distribution_on_{dataset}.png'))
    
    plt.clf()
    
    # 绘制 Groundtruth 和 Prediction 的 Token 分布
    gt_cnts = np.zeros((max_token_num, ))
    gt_cnts[gt_uni_elements] = gt_cnts_elements
    
    pred_cnts = np.zeros((max_token_num, ))
    pred_cnts[pred_uni_elements] = pred_cnts_elements
    
    data1, data2 = gt_cnts, pred_cnts
    
    print('data: ', data1.shape, data2.shape)
    
    data_low = [min(d1, d2) for d1, d2 in zip(data1, data2)]
    data_high = [max(d1, d2) for d1, d2 in zip(data1, data2)]

    colors_low = ['blue' if d1 < d2 else 'orange' for d1, d2 in zip(data1, data2)]
    colors_high = ['orange' if d1 < d2 else 'blue' for d1, d2 in zip(data1, data2)]

    # 设置横坐标
    x = np.arange(len(data1))

    # print(x, data_low, data_high)

    # 绘制柱状图
    data_high = (np.array(data_high) - np.array(data_low)).tolist()
    plt.bar(x, data_low, color=colors_low, label='Prediction')
    plt.bar(x, data_high, bottom=data_low, color=colors_high, label='GroundTruth') 

    plt.xlabel('Token ID')
    plt.ylabel('Token Count')
    plt.title('Token Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'gt_pred_token_distribution_on_{dataset}.png'))
    
    plt.clf()