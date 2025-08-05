import os
import time
import torch
import torch.nn as nn
import warnings
import numpy as np
from tqdm import tqdm
from torch import optim
import sys
import pickle
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, clever_format, plot_token_distribution_with_stratify, get_cosine_schedule_with_warmup
from utils.metrics import metric, token_metric
from models.Model4F import Model
from ecg_tokenizer.model_v1 import W_SimVQ
from ecg_tokenizer.Sim_VQ_CNN import W_SimVQ_CNN
from ecg_tokenizer.W_SimVQ_CNN_double import W_SimVQ_CNN_double
from ecg_tokenizer.W_SimVQ_CNN_double_token import W_SimVQ_CNN_double_token
from ecg_tokenizer.ResidualVQ_tcn_enc import VQVAE as ResidualVQ
from ecg_tokenizer.TimeSeriesPromptGenerator import TimeSeriesPromptGenerator
from torch.nn.utils.rnn import pad_sequence
from peft import PeftModel
warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast_Bert_v4(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)


    def _build_model(self):
        if self.args.VQ_type == 'SimVQ':
            vq_model = W_SimVQ(self.args)
        elif self.args.VQ_type == 'SimVQ_CNN':
            vq_model = W_SimVQ_CNN(self.args)
        elif self.args.VQ_type == 'W_SimVQ_CNN_double':
            vq_model = W_SimVQ_CNN_double(self.args)
        elif self.args.VQ_type == 'SimVQ_CNN_double_token':
            vq_model = W_SimVQ_CNN_double_token(self.args)
        elif self.args.VQ_type == 'ResidualVQ':
            vq_model = ResidualVQ(self.args)
        else:
            raise ValueError(f"VQ type@ {self.args.VQ_type} not supported!")

        vqvae_state_dict = torch.load(os.path.join(self.args.vqvae_model_path, 'model.pkl'), map_location="cpu")
        vq_model.load_state_dict(vqvae_state_dict, strict=False)
        weight_dict = pickle.load(open(os.path.join(self.args.vqvae_model_path, 'weight.pkl'), 'rb'))
        self.args.elected_n_embed = self.args.n_embed

        model = self.model_dict[self.args.model].Model(self.args).float()
        if not self.args.zero and self.args.pretrained_model:
            if self.accelerator.is_local_main_process:
                print(f"Loading pretrained model from {self.args.pretrained_model}")
            state_dict = torch.load(self.args.pretrained_model, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            if self.accelerator.is_local_main_process:
                print('Model loaded successfully.')

        weight, mask = weight_dict['weight'], weight_dict['mask']
        real_min_weight = np.min(weight, where=(mask == True), initial=np.inf)
        max_weight = real_min_weight * self.args.max_mpls
        classification_weight = None
        if self.args.max_mpls > 0:
            weight = np.clip(weight, a_min=None, a_max=max_weight)
            classification_weight = torch.tensor(weight, dtype=torch.float)
            if self.accelerator.is_local_main_process:
                print(f'Classification weight loaded: shape {classification_weight.shape}, min {real_min_weight}, max {max_weight}')

        vq_model = vq_model.to(self.device)
        for p in vq_model.parameters():
            p.requires_grad = False
        self.vq_model = vq_model

        # model = self.accelerator.prepare(model)
        self.model = model

        return model, vq_model, classification_weight
    
    
    import torch

    def build_input_and_label(self, batch_x, batch_y, start_date, end_date, is_train=True):
        """
        构建符合详细格式的输入和标签，动态接收起止日期。

        Args:
            self: 类实例
            batch_x (torch.Tensor): 输入的时间序列数据
            batch_y (torch.Tensor): 目标的时间序列数据
            start_date (str): 输入窗口的开始日期 (e.g., "2021-01-01")
            end_date (str): 输入窗口的结束日期 (e.g., "2024-12-01")
            is_train (bool): 是否为训练模式

        Returns:
            ...
        """
        # 元数据部分仍然是硬编码的
        series_metadata = {
            "source": "FRED-MD (Federal Reserve Economic Data - Monthly)",
            "name": "Industrial Production Index",
            "id": "INDPRO",
            "category": "Output and Income",
            "transformation": "Logarithmic first difference",
            "semantic_meaning": "The provided tokens represent the monthly percentage growth rate of industrial production."
        }

        # ... (从 VQ-VAE 获取 tokens 到 定义 tokenizer 和特殊标记的部分，与之前完全相同)
        # 1. 从 VQ-VAE 获取离散 tokens
        tokens = self.vq_model.get_code(batch_x, batch_y)
        
        # 自动获取 token 数量
        input_token_count = self.args.seq_len // self.args.wave_length
        output_token_count = self.args.pred_len // self.args.wave_length
        
        output_tokens = tokens[:, -output_token_count:]
        input_tokens = tokens[:, :-output_token_count]

        # 添加词表偏移量
        input_tokens = input_tokens + self.model.original_len
        output_tokens = output_tokens + self.model.original_len

        # 准备 tokenizer 和特殊标记
        tokenizer = self.model.text_tokenizer
        device = batch_x.device
        batch_size = input_tokens.size(0)

        ts_start_id = tokenizer.convert_tokens_to_ids("<TS_START>")
        ts_end_id = tokenizer.convert_tokens_to_ids("<TS_END>")
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        
        ts_start = torch.full((batch_size, 1), ts_start_id, dtype=torch.long, device=device)
        ts_end = torch.full((batch_size, 1), ts_end_id, dtype=torch.long, device=device)
        im_end = torch.full((batch_size, 1), im_end_id, dtype=torch.long, device=device)

        input_tokens_with_markers = torch.cat([ts_start, input_tokens, ts_end], dim=1)
        if is_train:
            output_tokens_with_markers = torch.cat([ts_start, output_tokens, ts_end], dim=1)

        def encode_and_repeat(text):
            ids = tokenizer(text, return_tensors="pt", padding=False, add_special_tokens=False)["input_ids"].to(device)
            return ids.repeat(batch_size, 1) if ids.shape[0] == 1 else ids

        # 2. 构建 System Prompt
        system_prompt_text = (
            "<|im_start|>system\n"
            "You are an expert econometrician and time series forecaster. Your task is to analyze the provided "
            "macroeconomic data and context to produce the most likely forecast. Pay close attention to all "
            "metadata, especially the transformation and statistical properties.\n<|im_end|>\n"
        )
        system_ids = encode_and_repeat(system_prompt_text)

        # 3. 构建 User Prompt, 现在包含动态日期
        input_mean = batch_x.mean().item()
        input_std = batch_x.std().item()

        user_prompt_prefix_text = f"""<|im_start|>user
    Your primary task is to perform time series forecasting. You must return only the predicted time series tokens, enclosed strictly between <TS_START> and <TS_END> markers.

    ### Time Series Metadata ###
    - **Source**: {series_metadata['source']}
    - **Series Name**: {series_metadata['name']}
    - **Series ID**: {series_metadata['id']}
    - **Category**: {series_metadata['category']}
    - **Transformation Applied**: {series_metadata['transformation']}
    - **Semantic Meaning**: {series_metadata['semantic_meaning']}

    ### Statistical Properties of the Input Data ###
    - **Input Window Start Date**: {start_date}
    - **Input Window End Date**: {end_date}
    - **Input Window Mean (of transformed data)**: {input_mean:.4f}
    - **Input Window Std. Dev. (of transformed data)**: {input_std:.4f}

    Based on the metadata, statistical properties, economic context, and the historical tokens provided below, predict the next {output_token_count} tokens.
    """
        user_prompt_prefix_ids = encode_and_repeat(user_prompt_prefix_text)

        user_prompt_suffix_text = f"\nThe tokens capture historical trends in the {series_metadata['id']} growth rate.\n<|im_end|>\n"
        user_prompt_suffix_ids = encode_and_repeat(user_prompt_suffix_text)

        # 4. 构建 Assistant Prompt 的起始部分
        assistant_start_ids = encode_and_repeat("<|im_start|>assistant\n")

        # 5. 组合最终的输入和标签 (这部分逻辑不变)
        if is_train:
            # ... (与上一版完全相同)
            input_ids = torch.cat([
                system_ids, user_prompt_prefix_ids, input_tokens_with_markers,
                user_prompt_suffix_ids, assistant_start_ids, output_tokens_with_markers, im_end
            ], dim=1)
            labels = torch.full_like(input_ids, -100)
            start_of_label = (
                system_ids.shape[1] + user_prompt_prefix_ids.shape[1] +
                input_tokens_with_markers.shape[1] + user_prompt_suffix_ids.shape[1] +
                assistant_start_ids.shape[1]
            )
            end_of_label = start_of_label + output_tokens_with_markers.shape[1] + im_end.shape[1]
            labels[:, start_of_label:end_of_label] = input_ids[:, start_of_label:end_of_label]
            return input_ids, labels
        else:
            # ... (与上一版完全相同)
            input_ids = torch.cat([
                system_ids, user_prompt_prefix_ids, input_tokens_with_markers,
                user_prompt_suffix_ids, assistant_start_ids
            ], dim=1)
            output_tokens_original = output_tokens - self.model.original_len
            input_tokens_original = input_tokens - self.model.original_len
            return input_ids, output_tokens_original, input_tokens_original, 0, output_tokens_original.shape[1]

    def _print_trainable_parameters(self, model):
        """Print statistics about model parameters, including trainable vs frozen counts, memory, and device usage."""
        freeze_params = 0
        trainable_params = 0
        trainable_param_list = []
        total_size_bytes = 0
        device_counter = {}

        # Collect parameter statistics
        for name, param in model.named_parameters():
            param_device = str(param.device)
            param_dtype = param.dtype
            param_size = param.nelement() * param.element_size()

            if param.requires_grad:
                trainable_params += param.nelement()
                total_size_bytes += param_size
                trainable_param_list.append((name, param_device, param_dtype))
                device_counter[param_device] = device_counter.get(param_device, 0) + 1
            else:
                freeze_params += param.nelement()

        total_params = trainable_params + freeze_params

        def format_size(num_bytes):
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if num_bytes < 1024.0:
                    return f"{num_bytes:.2f}{unit}"
                num_bytes /= 1024.0
            return f"{num_bytes:.2f}PB"

        if self.accelerator.is_local_main_process:
            print('=' * 60)
            print('Model Parameter Statistics:')
            print(f'Trainable parameters: {trainable_params:,}')
            print(f'Frozen parameters:    {freeze_params:,}')
            print(f'Total parameters:     {total_params:,}')
            print(f'Trainable ratio:      {(trainable_params / total_params) * 100:.2f}%')
            print(f'Estimated trainable parameter size: {format_size(total_size_bytes)}')

            print('\nDevice distribution of trainable parameters:')
            for device, count in device_counter.items():
                print(f'- {device}: {count} tensors')

            print('\nTrainable parameter names (first 10):')
            for name, device, dtype in trainable_param_list[:10]:
                print(f'- {name} (device: {device}, dtype: {str(dtype)})')

            if len(trainable_param_list) > 10:
                print(f'... and {len(trainable_param_list) - 10} more parameters')
            print('=' * 60)


    def _get_data(self, flag, data=None):
        """Get data loader for training, validation, or testing."""
        data_set, data_loader = data_provider(self.args, flag, data)
        return data_set, data_loader

    def _select_optimizer(self, lr):
        """Create an optimizer for the model."""
        return optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.args.weight_decay)

    def _select_criterion(self):
        """Create a loss function."""
        return nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
    
    def pretrain(self, setting):
        """Train the model with distributed multi-GPU support using HuggingFace Accelerate."""
        loaders = {k: self._get_data(flag=k) for k in ['train', 'val', 'test']}
        train_data, train_loader = loaders['train']
        vali_data, vali_loader = loaders['val']
        test_data, test_loader = loaders['test']
        criterion = self._select_criterion()
        self._print_trainable_parameters(self.model)
        path = os.path.join(self.args.checkpoints, setting)
        if self.accelerator.is_local_main_process:
            os.makedirs(path, exist_ok=True)
        model_optim = self._select_optimizer(lr=self.args.learning_rate)
        train_steps = len(train_loader)
        time_now = time.time()
        scheduler = get_cosine_schedule_with_warmup(
            model_optim,
            warmup_epochs=getattr(self.args, 'warmup_epochs', 2) * train_steps,
            total_epochs=self.args.train_epochs * train_steps
        )
        def test_callback():
            mse, mae = self.test(setting, test=1, save_root=self.args.checkpoints)
            self.accelerator.print(f"[✅] Test after saving best model | MSE: {mse:.3f}, MAE: {mae:.3f}")

        early_stopping = EarlyStopping(accelerator=self.accelerator, patience=self.args.patience,test_fn=test_callback)
        best_model_path = os.path.join(path, 'checkpoint.pth')
        
        self.accelerator.init_trackers(setting)
        train_loader, vali_loader, test_loader = self.accelerator.prepare(train_loader, vali_loader, test_loader)
        self.model, model_optim, scheduler = self.accelerator.prepare(self.model, model_optim, scheduler)
        accumulation_steps = getattr(self.args, 'accumulation_steps', 1)
        iter_verbose = 100
        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss, iter_count = [], 0
            epoch_time = time.time()
            all_ts_correct = []
            all_ts_total = []
            all_text_correct = []
            all_text_total = []
            all_correct = []
            all_total = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, start_dates, end_dates) in enumerate(train_loader):
                iter_count += 1
                batch_x, batch_y = batch_x.float(), batch_y.float()
                batch_y = batch_y[:, -self.args.pred_len:, :]
                ts_ids, labels = self.build_input_and_label(
                    batch_x,
                    batch_y,
                    start_date=start_dates[0], # 使用批次中第一个样本的开始日期
                    end_date=end_dates[0],     # 使用批次中第一个样本的结束日期
                    is_train=True
                )
                ts_ids = ts_ids.to(self.device)
                labels = labels.to(self.device)
                inputs = {
                    'text_ids': None,
                    'ts_ids': ts_ids,
                    'labels': ts_ids
                }
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = outputs.loss
                else:
                    outputs = self.model(inputs)
                    loss = outputs.loss
                loss = loss.mean() / accumulation_steps
                train_loss.append(loss.item() * accumulation_steps)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1)
                    # === Shifted Accuracy Calculation ===
                    shifted_preds = preds[:, :-1]
                    shifted_labels = labels[:, 1:]
                    valid_mask = shifted_labels != -100
                    ts_mask = valid_mask & (shifted_labels >= self.model.original_len)
                    text_mask = valid_mask & (shifted_labels < self.model.original_len)
                    ts_correct = ((shifted_preds == shifted_labels) & ts_mask).int()
                    text_correct = ((shifted_preds == shifted_labels) & text_mask).int()
                    total_correct = ((shifted_preds == shifted_labels) & valid_mask).int()
                    ts_total = ts_mask.int()
                    text_total = text_mask.int()
                    total_token = valid_mask.int()
                    ts_correct = self.accelerator.gather_for_metrics(ts_correct).sum()
                    ts_total = self.accelerator.gather_for_metrics(ts_total).sum()
                    text_correct = self.accelerator.gather_for_metrics(text_correct).sum()
                    text_total = self.accelerator.gather_for_metrics(text_total).sum()
                    total_correct = self.accelerator.gather_for_metrics(total_correct).sum()
                    total_token = self.accelerator.gather_for_metrics(total_token).sum()
                    all_ts_correct.append(ts_correct)
                    all_ts_total.append(ts_total)
                    all_text_correct.append(text_correct)
                    all_text_total.append(text_total)
                    all_correct.append(total_correct)
                    all_total.append(total_token)
                
                # if i ==2:
                #     break
                self.accelerator.backward(loss)
                if (i + 1) % accumulation_steps == 0:
                    model_optim.step()
                    scheduler.step()
                    model_optim.zero_grad()
                if (i + 1) % iter_verbose == 0:
                    ts_acc = torch.stack(all_ts_correct).sum().item() / max(torch.stack(all_ts_total).sum().item(), 1)
                    text_acc = torch.stack(all_text_correct).sum().item() / max(torch.stack(all_text_total).sum().item(), 1)
                    total_acc = torch.stack(all_correct).sum().item() / max(torch.stack(all_total).sum().item(), 1)
                    self.accelerator.print(
                        f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item() * accumulation_steps:.7f} | "
                        f"text_acc: {text_acc * 100:.2f}% | ts_acc: {ts_acc * 100:.2f}% | total_acc: {total_acc * 100:.2f}%")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    self.accelerator.print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()
                    step = epoch * train_steps + i
                    self.accelerator.log(
                        {"train_loss": loss.item() * accumulation_steps,
                        "learning_rate": scheduler.get_last_lr()[0]},
                        step=step
                    )
            train_loss = np.average(train_loss)
            ts_acc = torch.stack(all_ts_correct).sum().item() / max(torch.stack(all_ts_total).sum().item(), 1)
            text_acc = torch.stack(all_text_correct).sum().item() / max(torch.stack(all_text_total).sum().item(), 1)
            total_acc = torch.stack(all_correct).sum().item() / max(torch.stack(all_total).sum().item(), 1)
            self.accelerator.print(f"Epoch {epoch + 1} completed in {time.time() - epoch_time:.2f}s")
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            self.accelerator.print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} "
                f"Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f} "
                f"Text Acc: {text_acc * 100:.2f}% TS Acc: {ts_acc * 100:.2f}% Total Acc: {total_acc * 100:.2f}%"
            )
            self.accelerator.log(
                {"epoch": epoch,
                "train_loss_avg": train_loss,
                "val_loss": vali_loss,
                "test_loss": test_loss,
                "train_ts_acc": ts_acc,
                "train_text_acc": text_acc,
                "train_total_acc": total_acc},
                step=epoch
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                self.accelerator.print("Early stopping")
                break
        if self.accelerator.is_local_main_process and os.path.exists(best_model_path):
            model_state = torch.load(best_model_path, map_location=self.device)
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_state_dict(model_state)
        self.accelerator.end_training()
        return self.model
    def vali(self, vali_data, vali_loader, criterion):
        """Run validation and return average loss (print total/text/ts token accuracy)."""
        self.model.eval()
        total_loss = []

        all_ts_correct, all_ts_total = [], []
        all_text_correct, all_text_total = [], []
        all_correct, all_total = [], []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, start_dates, end_dates) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:, :]

                ts_ids, labels = self.build_input_and_label(
                    batch_x,
                    batch_y,
                    start_date=start_dates[0], # 使用批次中第一个样本的开始日期
                    end_date=end_dates[0],     # 使用批次中第一个样本的结束日期
                    is_train=True
                )
                ts_ids = ts_ids.to(self.device)
                labels = labels.to(self.device)

                inputs = {
                    'text_ids': None,
                    'ts_ids': ts_ids,
                    'labels': labels
                }

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                loss = outputs.loss.mean()
                total_loss.append(loss.detach().cpu().item())
                # if i==2:
                #     break

                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1)

                    # Shift predictions and labels
                    shifted_preds = preds[:, :-1].contiguous()
                    shifted_labels = labels[:, 1:].contiguous()

                    valid_mask = shifted_labels != -100
                    ts_mask = valid_mask & (shifted_labels >= self.model.original_len)
                    text_mask = valid_mask & (shifted_labels < self.model.original_len)
                    total_mask = valid_mask

                    ts_correct = ((shifted_preds == shifted_labels) & ts_mask).int()
                    text_correct = ((shifted_preds == shifted_labels) & text_mask).int()
                    total_correct = ((shifted_preds == shifted_labels) & total_mask).int()

                    ts_total = ts_mask.int()
                    text_total = text_mask.int()
                    total_token = total_mask.int()

                    ts_correct = self.accelerator.gather_for_metrics(ts_correct).sum()
                    ts_total = self.accelerator.gather_for_metrics(ts_total).sum()
                    text_correct = self.accelerator.gather_for_metrics(text_correct).sum()
                    text_total = self.accelerator.gather_for_metrics(text_total).sum()
                    total_correct = self.accelerator.gather_for_metrics(total_correct).sum()
                    total_token = self.accelerator.gather_for_metrics(total_token).sum()

                    all_ts_correct.append(ts_correct)
                    all_ts_total.append(ts_total)
                    all_text_correct.append(text_correct)
                    all_text_total.append(text_total)
                    all_correct.append(total_correct)
                    all_total.append(total_token)

        self.model.train()

        ts_acc = torch.stack(all_ts_correct).sum().item() / max(torch.stack(all_ts_total).sum().item(), 1)
        text_acc = torch.stack(all_text_correct).sum().item() / max(torch.stack(all_text_total).sum().item(), 1)
        total_acc = torch.stack(all_correct).sum().item() / max(torch.stack(all_total).sum().item(), 1)

        self.accelerator.print(
            f"[Validation] Loss: {np.mean(total_loss):.6f} | Total Acc: {total_acc * 100:.2f}% "
            f"(Text: {text_acc * 100:.2f}% | TS: {ts_acc * 100:.2f}%)"
        )

        return np.mean(total_loss)

    def train(self, setting, test=0):
        """Train the model with distributed multi-GPU support using HuggingFace Accelerate."""

        loaders = {k: self._get_data(flag=k) for k in ['train', 'val', 'test']}
        train_data, train_loader = loaders['train']
        vali_data, vali_loader = loaders['val']
        test_data, test_loader = loaders['test']

        criterion = self._select_criterion()
        self._print_trainable_parameters(self.model)

        path = os.path.join(self.args.checkpoints, setting)
        if self.accelerator.is_local_main_process:
            os.makedirs(path, exist_ok=True)

        model_optim = self._select_optimizer(lr=self.args.learning_rate)
        train_steps = len(train_loader)
        time_now = time.time()

        scheduler = get_cosine_schedule_with_warmup(
            model_optim,
            warmup_epochs=getattr(self.args, 'warmup_epochs', 2) * train_steps,
            total_epochs=self.args.train_epochs * train_steps
        )

        def test_callback():
            mse, mae = self.test(setting, test=1, save_root=self.args.checkpoints)
            self.accelerator.print(f"[✅] Test after saving best model | MSE: {mse:.3f}, MAE: {mae:.3f}")

        early_stopping = EarlyStopping(accelerator=self.accelerator, patience=self.args.patience,test_fn=test_callback)
        best_model_path = os.path.join(path, 'checkpoint.pth')
        # if os.path.exists(best_model_path):
        #     self.accelerator.print("Resuming from last checkpoint.")
        #     self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        self.accelerator.init_trackers(setting)
        train_loader, vali_loader, test_loader = self.accelerator.prepare(train_loader, vali_loader, test_loader)
        self.model, model_optim, scheduler = self.accelerator.prepare(self.model, model_optim, scheduler)

        accumulation_steps = getattr(self.args, 'accumulation_steps', 1)
        iter_verbose = 100

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss, iter_count = [], 0
            epoch_time = time.time()

            all_ts_correct = []
            all_ts_total = []
            all_text_correct = []
            all_text_total = []
            all_correct = []
            all_total = []

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, start_dates, end_dates) in enumerate(train_loader):
                iter_count += 1

                batch_x, batch_y = batch_x.float(), batch_y.float()
                batch_y = batch_y[:, -self.args.pred_len:, :]

                ts_ids, labels = self.build_input_and_label(
                    batch_x,
                    batch_y,
                    start_date=start_dates[0], # 使用批次中第一个样本的开始日期
                    end_date=end_dates[0],     # 使用批次中第一个样本的结束日期
                    is_train=True
                )
                ts_ids = ts_ids.to(self.device)
                labels = labels.to(self.device)

                inputs = {
                    'text_ids': None,
                    'ts_ids': ts_ids,
                    'labels': labels
                }

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = outputs.loss
                else:
                    outputs = self.model(inputs)
                    loss = outputs.loss

                loss = loss.mean() / accumulation_steps
                train_loss.append(loss.item() * accumulation_steps)

                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1)

                    # === Shifted Accuracy Calculation ===
                    shifted_preds = preds[:, :-1]
                    shifted_labels = labels[:, 1:]

                    valid_mask = shifted_labels != -100
                    ts_mask = valid_mask & (shifted_labels >= self.model.original_len)
                    text_mask = valid_mask & (shifted_labels < self.model.original_len)

                    ts_correct = ((shifted_preds == shifted_labels) & ts_mask).int()
                    text_correct = ((shifted_preds == shifted_labels) & text_mask).int()
                    total_correct = ((shifted_preds == shifted_labels) & valid_mask).int()

                    ts_total = ts_mask.int()
                    text_total = text_mask.int()
                    total_token = valid_mask.int()

                    ts_correct = self.accelerator.gather_for_metrics(ts_correct).sum()
                    ts_total = self.accelerator.gather_for_metrics(ts_total).sum()
                    text_correct = self.accelerator.gather_for_metrics(text_correct).sum()
                    text_total = self.accelerator.gather_for_metrics(text_total).sum()
                    total_correct = self.accelerator.gather_for_metrics(total_correct).sum()
                    total_token = self.accelerator.gather_for_metrics(total_token).sum()

                    all_ts_correct.append(ts_correct)
                    all_ts_total.append(ts_total)
                    all_text_correct.append(text_correct)
                    all_text_total.append(text_total)
                    all_correct.append(total_correct)
                    all_total.append(total_token)
                
                # if i ==2:
                #     break

                self.accelerator.backward(loss)
                if (i + 1) % accumulation_steps == 0:
                    model_optim.step()
                    scheduler.step()
                    model_optim.zero_grad()

                if (i + 1) % iter_verbose == 0:
                    ts_acc = torch.stack(all_ts_correct).sum().item() / max(torch.stack(all_ts_total).sum().item(), 1)
                    text_acc = torch.stack(all_text_correct).sum().item() / max(torch.stack(all_text_total).sum().item(), 1)
                    total_acc = torch.stack(all_correct).sum().item() / max(torch.stack(all_total).sum().item(), 1)
                    self.accelerator.print(
                        f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item() * accumulation_steps:.7f} | "
                        f"text_acc: {text_acc * 100:.2f}% | ts_acc: {ts_acc * 100:.2f}% | total_acc: {total_acc * 100:.2f}%")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    self.accelerator.print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

                    step = epoch * train_steps + i
                    self.accelerator.log(
                        {"train_loss": loss.item() * accumulation_steps,
                        "learning_rate": scheduler.get_last_lr()[0]},
                        step=step
                    )

            train_loss = np.average(train_loss)
            ts_acc = torch.stack(all_ts_correct).sum().item() / max(torch.stack(all_ts_total).sum().item(), 1)
            text_acc = torch.stack(all_text_correct).sum().item() / max(torch.stack(all_text_total).sum().item(), 1)
            total_acc = torch.stack(all_correct).sum().item() / max(torch.stack(all_total).sum().item(), 1)

            self.accelerator.print(f"Epoch {epoch + 1} completed in {time.time() - epoch_time:.2f}s")

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            self.accelerator.print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} "
                f"Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f} "
                f"Text Acc: {text_acc * 100:.2f}% TS Acc: {ts_acc * 100:.2f}% Total Acc: {total_acc * 100:.2f}%"
            )

            self.accelerator.log(
                {"epoch": epoch,
                "train_loss_avg": train_loss,
                "val_loss": vali_loss,
                "test_loss": test_loss,
                "train_ts_acc": ts_acc,
                "train_text_acc": text_acc,
                "train_total_acc": total_acc},
                step=epoch
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                self.accelerator.print("Early stopping")
                break

        if self.accelerator.is_local_main_process and os.path.exists(best_model_path):
            self.accelerator.print(f"Loading best model from {best_model_path}")
            model_state = torch.load(best_model_path, map_location=self.device)
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_state_dict(model_state)

        self.accelerator.end_training()
        return self.model

    @torch.no_grad()
    def decode_ts(self,input_ids, output_ids, B):
        """Decode time series tokens back to values."""
        # num_t = 2
        B_C, n_nt = output_ids.shape
        # output_ids = torch.reshape(output_ids, (-1, num_t))
        device = input_ids.device
        input_tokens = torch.cat([input_ids, output_ids.to(device)], dim=1)
        # Decode tokens
        input_tokens = input_tokens.to(self.vq_model.device if hasattr(self.vq_model, 'device') else self.device)

        decode_ts = self.vq_model.decode_ids(input_tokens).squeeze()
        if decode_ts.ndim == 2:
            decode_ts = decode_ts.unsqueeze(0)
        if self.args.chan_indep:
            decode_ts = torch.reshape(decode_ts, (B_C, -1))
            decode_ts = torch.reshape(decode_ts, (B, -1, decode_ts.shape[-1]))
            decode_ts = decode_ts.permute(0, 2, 1)  
        
        # Apply revin if used
        B, L, C = decode_ts.shape
        if self.vq_model.revin == True:
            decode_ts = self.vq_model.revin_layer(decode_ts, 'denorm')
        
        return decode_ts[:, -self.args.pred_len:, :]

    def process_output_tokens(self, output_tokens):
        """Process output tokens to separate text and time series tokens based on special tokens."""
        batch_size = output_tokens.shape[0]
        
        # Get special token IDs
        ts_start_id = self.model.text_tokenizer.convert_tokens_to_ids('<TS_START>')
        ts_end_id = self.model.text_tokenizer.convert_tokens_to_ids('<TS_END>')
        
        # Initialize lists to store tokens
        text_tokens_list = []
        ts_tokens_list = []
        
        # Process each sequence in the batch
        for i in range(batch_size):
            seq = output_tokens[i]
            
            # Find special token positions
            ts_start_pos = torch.where(seq == ts_start_id)[0]
            ts_end_pos = torch.where(seq == ts_end_id)[0]
            
            if len(ts_start_pos) == 0 or len(ts_end_pos) == 0:
                self.accelerator.print(f"Warning: Missing special tokens in sequence {i}")
                continue
                
            # Extract text tokens (before first TS_START)
            text_tokens = seq[:ts_start_pos[0]]
            
            # Extract time series tokens (between TS_START and TS_END)
            ts_tokens = seq[ts_start_pos[0]+1:ts_end_pos[0]]
            
            text_tokens_list.append(text_tokens)
            ts_tokens_list.append(ts_tokens)
        
        # Stack the tokens back into batch
        ts_tokens = torch.stack(ts_tokens_list)
        
        ts_tokens = ts_tokens - self.model.original_len
        
        return text_tokens_list, ts_tokens
    
    @torch.no_grad()
    def test_func(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            self.accelerator.print("Loading model...")
            self.model.load_state_dict(torch.load(self.args.pretrained_model, map_location=self.device))

        self.model.eval()
        preds, trues, inputx = [], [], []
        output_tokens_list, gt_tokens_list = [], []

        folder_path = os.path.join('./test_results2', setting)
        if self.accelerator.is_local_main_process:
            os.makedirs(folder_path, exist_ok=True)

        test_loader = self.accelerator.prepare(test_loader)
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, start_dates, end_dates) in enumerate(tqdm(test_loader, desc="Testing", disable=not self.accelerator.is_local_main_process, file=sys.stderr)):
            # if i < 167:
            #     continue  # ✅ 只执行第 137 个 batch
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_y = batch_y[:, -self.args.pred_len:, :]

            B = batch_x.shape[0]
            ts_ids, gt_tokens, input_tokens, text_tokens_len,ts_token_len = self.build_input_and_label(
                batch_x,
                batch_y,
                start_date=start_dates[0], # 使用批次中第一个样本的开始日期
                end_date=end_dates[0],     # 使用批次中第一个样本的结束日期
                is_train=False
            )

            inputs = {
                'text_ids': None,
                'ts_ids': ts_ids.to(self.device),
                'labels': gt_tokens.to(self.device),
            }

            output_tokens = self.model.gen_ts(inputs, text_tokens_len,ts_token_len)
            gt_tokens_list.append(gt_tokens)

            text_tokens, ts_tokens = self.process_output_tokens(output_tokens)
            
            output_tokens_list.append(ts_tokens)
            
            outputs = self.decode_ts(input_tokens.to(self.device),ts_tokens, B=B)

            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            preds.append(pred)
            trues.append(true)
            inputx.append(batch_x.detach().cpu().numpy())
            

            if i % 5 == 0 and self.accelerator.is_local_main_process:
                input_np = batch_x.detach().cpu().numpy()
                gt = np.concatenate((input_np[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((input_np[0, :, -1], pred[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, f"{i}.pdf"))
            
            # if i == 2:
            #     break

        output_tokens_list = torch.cat(output_tokens_list, dim=0)
        gt_tokens_list = torch.cat(gt_tokens_list, dim=0)
        return preds, trues, inputx, output_tokens_list, gt_tokens_list


    def test(self, setting, test=0, save_root='checkpoints'):
        _, _ = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')

        if test:
            self.accelerator.print("Loading model...")

            model_path = os.path.join(save_root, setting)
            lora_adapter_path = os.path.join(model_path, "lora_adapter")
            state_dict_path = os.path.join(model_path, "checkpoint.pth")
            print(state_dict_path)

            unwrapped_model = self.accelerator.unwrap_model(self.model)

            if os.path.exists(lora_adapter_path):
                # ✅ LoRA adapter exists，加载 adapter
                self.accelerator.print(f"Loading LoRA adapter from {lora_adapter_path}")
                self.model = PeftModel.from_pretrained(unwrapped_model, lora_adapter_path)
            elif os.path.exists(state_dict_path):
                # ✅ 加载普通 state_dict
                self.accelerator.print(f"Loading full model state_dict from {state_dict_path}")
                model_state = torch.load(state_dict_path, map_location=self.device)
                unwrapped_model.load_state_dict(model_state)
            else:
                raise FileNotFoundError(f"No checkpoint found in {model_path}")
        self.model.eval()
        preds, trues, inputx, output_tokens, gt_tokens = self.test_func(setting=setting)

        # Plot token distribution
        plot_token_distribution_with_stratify(
            gt_tokens, output_tokens,
            save_dir=os.path.join(save_root, setting),
            max_token_num=self.args.elected_n_embed,
            dataset='test'
        )

        token_metric_dict = token_metric(output_tokens, gt_tokens)
        self.accelerator.print("Token Metric:")
        self.accelerator.print(token_metric_dict)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        self.accelerator.print(f"mse: {mse}, mae: {mae}")

        if self.accelerator.is_local_main_process:
            with open("result.txt", 'a') as f:
                f.write(f"{setting}\n")
                f.write(f"mse: {mse}, mae: {mae}\n\n")

        return mse, mae



    def test_single_sample_overfit(self, setting, num_epochs=150):
        print('\n################# Single Sample Overfit Test #################')
        self.device = self.args.gpu
        self.model.to(self.device)
        self.vq_model.to(self.device)

        # 取单个样本
        train_data, train_loader = self._get_data(flag='train')
        single_batch = next(iter(train_loader))
        batch_x, batch_y, _, _ = single_batch
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y[:, -self.args.pred_len:, :].float().to(self.device)

        # 构造输入与标签
        ts_ids, labels = self.build_input_and_label(batch_x, batch_y, is_train=True)
        inputs = {
            'text_ids': None,
            'ts_ids': ts_ids.to(self.device),
            'labels': labels.to(self.device)
        }

        model_optim = self._select_optimizer(lr=0.001)
        criterion = self._select_criterion()

        print(f"Training on single sample for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            self.model.train()
            model_optim.zero_grad()
            outputs = self.model(inputs)
            loss = outputs.loss.mean()
            loss.backward()
            model_optim.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

                # Token-level Accuracy with SHIFT
                with torch.no_grad():
                    logits = outputs.logits  # (B, T, V)
                    preds = torch.argmax(logits, dim=-1)  # (B, T)

                    shifted_preds = preds[:, :-1]
                    shifted_labels = labels[:, 1:]

                    valid_mask = shifted_labels != -100
                    ts_mask = valid_mask & (shifted_labels >= self.model.original_len)
                    text_mask = valid_mask & (shifted_labels < self.model.original_len)

                    ts_acc = ((shifted_preds == shifted_labels) & ts_mask).sum().item() / max(ts_mask.sum().item(), 1)
                    text_acc = ((shifted_preds == shifted_labels) & text_mask).sum().item() / max(text_mask.sum().item(), 1)
                    total_acc = ((shifted_preds == shifted_labels) & valid_mask).sum().item() / max(valid_mask.sum().item(), 1)

                    print(f"Token Acc | Text: {text_acc*100:.2f}%, TS: {ts_acc*100:.2f}%, Total: {total_acc*100:.2f}%")

        # 推理评估
        test_ts_ids, test_labels, out_token_shape = self.build_input_and_label(batch_x, batch_y, is_train=False)
        inputs = {
            'text_ids': None,
            'ts_ids': test_ts_ids.to(self.device),
            'labels': test_labels.to(self.device),
        }

        self.model.eval()
        with torch.no_grad():
            output_tokens = self.model.gen_ts(inputs, out_token_shape)
            text_tokens, ts_tokens = self.process_output_tokens(output_tokens)

            B = batch_x.shape[0]
            decoded_outputs = self.decode_ts(ts_tokens, B=B)
            decoded_text = self.model.text_tokenizer.batch_decode(
                text_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            print(decoded_text)

            decoded_outputs = decoded_outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()
            
            print("ts_tokens shape:", ts_tokens.shape)
            print("gt_tokens shape:", test_labels.shape)
            token_metric_dict = token_metric(ts_tokens, test_labels)
            print("\nToken Metrics:")
            print(token_metric_dict)

            mae, mse, rmse, mape, mspe = metric(decoded_outputs, batch_y)
            print("\nReconstruction Metrics:")
            print(f"MSE: {mse:.6f}")
            print(f"MAE: {mae:.6f}")
            print(f"RMSE: {rmse:.6f}")

            folder_path = './test_results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            input_data = batch_x.detach().cpu().numpy()
            gt = np.concatenate((input_data[0, :, -1], batch_y[0, :, -1]), axis=0)
            pd = np.concatenate((input_data[0, :, -1], decoded_outputs[0, :, -1]), axis=0)
            visual(gt, pd, os.path.join(folder_path, 'single_sample_overfit.pdf'))

            plot_token_distribution_with_stratify(
                test_labels, ts_tokens,
                save_dir=folder_path,
                max_token_num=self.args.elected_n_embed,
                dataset='single_sample_overfit'
            )

        return mse, mae

