import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    LogitsProcessorList, LogitsProcessor
)
import random
from peft import get_peft_model, LoraConfig, TaskType

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets, position_weights=None):
        """
        inputs: (B, L, C) or (B, C) logits
        targets: (B, L) or (B,) with class indices, may include -100 for ignore
        position_weights: (B, L) or (B,) or None
        """
        if inputs.dim() == 3:
            B, L, C = inputs.shape
            inputs = inputs.reshape(B * L, C)
            targets = targets.reshape(B * L)
            if position_weights is not None:
                position_weights = position_weights.reshape(B * L)
        else:
            B, C = inputs.shape
            targets = targets.reshape(B)
            if position_weights is not None:
                position_weights = position_weights.reshape(B)

        valid_mask = (targets != self.ignore_index).float()
        # Compute cross entropy (no reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)  # (N,)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # (N,)

        # Apply optional weights
        if position_weights is not None:
            focal_loss = focal_loss * position_weights

        # Reduction
        if self.reduction == 'mean':
            denom = (valid_mask * (position_weights if position_weights is not None else 1.0)).sum()
            return focal_loss.sum() / (denom + 1e-8)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            if inputs.dim() == 2:
                return focal_loss.view(B)
            else:
                return focal_loss.view(B, L)

class TsTokenFormatController(LogitsProcessor):
    def __init__(self, ts_token_range, ts_start_token_id, ts_end_token_id, ts_start_pos, ts_len):
        self.ts_token_start, self.ts_token_end = ts_token_range
        self.ts_start_token_id = ts_start_token_id
        self.ts_end_token_id = ts_end_token_id
        self.ts_start_pos = ts_start_pos  
        self.ts_end_pos = ts_start_pos + 1 + ts_len  

    def __call__(self, input_ids, scores):
        cur_len = input_ids.shape[1]
       
        mask = torch.full_like(scores, float("-inf"))

        if cur_len == self.ts_start_pos:
            mask[:, self.ts_start_token_id] = scores[:, self.ts_start_token_id]
            return mask

        elif self.ts_start_pos < cur_len < self.ts_end_pos:

            mask[:, self.ts_token_start:self.ts_token_end] = scores[:, self.ts_token_start:self.ts_token_end]

            
            topk = torch.topk(mask[:, self.ts_token_start:self.ts_token_end], k=5, dim=-1)

            return mask

        elif cur_len == self.ts_end_pos:
            
            mask[:, self.ts_end_token_id] = scores[:, self.ts_end_token_id]
            return mask
        else:
            return scores

class Model(nn.Module):
    def __init__(self,configs):
        super(Model, self).__init__()
        self.configs = configs
        config = AutoConfig.from_pretrained(self.configs.local_model_path)
        self.d_model = config.hidden_size
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.configs.local_model_path)
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        
        # 添加时序特殊token
        special_tokens_dict = {
        'additional_special_tokens': ['<TS_START>', '<TS_END>']
        }

        self.text_tokenizer.add_special_tokens(special_tokens_dict)
        # self.text_tokenizer.apply_chat_template
        
        self.n_embed = self.configs.elected_n_embed
        # 初始化Qwen模型
        self.model = self._initialize_model(config)
        # 初始化嵌入层
        self._initialize_embedding_layer()

        self._initialize_output_layer(config)

        if self.configs.layers:
            num_layers = len(self.model.model.layers)  # 获取 Transformer 总层数
            print(f"Qwen2.5 共有 {num_layers} 层 Transformer")  
            for param in self.model.model.parameters():
                param.requires_grad = False
            n_unfreeze = self.configs.n_layers  # 只解冻最后 3 层
            print(n_unfreeze)

            for i in range(num_layers - n_unfreeze, num_layers):
                for param in self.model.model.layers[i].parameters():
                    param.requires_grad = True # 解冻最后 3 层
            
            # ✅ Step 3: 解冻 embedding 层
            for param in self.model.model.embed_tokens.parameters():
                param.requires_grad = True

            # ✅ Step 4: 解冻输出层（lm_head）
            for param in self.model.lm_head.parameters():
                param.requires_grad = True

        if self.configs.frozen:
            # 全部冻结
            for param in self.parameters():
                param.requires_grad = False

            # 解冻嵌入层
            for param in self.model.model.embed_tokens.parameters():
                param.requires_grad = True

            # 解冻输出层
            for param in self.model.lm_head.parameters():
                param.requires_grad = True

        
        if self.configs.use_lora:
            print("🔧 Applying LoRA to model...")
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM,  # 因为是语言模型
                target_modules=["q_proj", "v_proj"]  # 根据你实际使用的模型修改模块名（可print确认）
            )
            self.model = get_peft_model(self.model, lora_config)
            

    def _initialize_model(self, config):
        if self.configs.params:
            return AutoModelForCausalLM.from_pretrained(
                self.configs.local_model_path, 
                output_attentions=True, 
                output_hidden_states=True,
                trust_remote_code=True
            )
        else:
            return AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    def _initialize_embedding_layer(self, use_normal_dist=True):
        original_weight = self.model.model.embed_tokens.weight
        self.original_len = len(original_weight)

        # 🔸 获取 special token 数量
        special_tokens_len = len(self.text_tokenizer.additional_special_tokens)

        if use_normal_dist:
            mu = torch.mean(original_weight, dim=0)
            n = original_weight.size()[0]
            sigma = ((original_weight - mu).T @ (original_weight - mu)) / n
            dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5*sigma)

            ts_weight = torch.stack([dist.sample() for _ in range(self.n_embed)], dim=0)
            special_tokens_weight = torch.stack([dist.sample() for _ in range(special_tokens_len)], dim=0)
        else:
            random.seed(self.configs.seed)
            sample_indices = random.sample(range(len(original_weight)), self.n_embed)
            ts_weight = original_weight[sample_indices]

            special_indices = random.sample(range(len(original_weight)), special_tokens_len)
            special_tokens_weight = original_weight[special_indices]

        # 🔸 扩展词表
        total_vocab_size = self.original_len + self.n_embed + special_tokens_len
        self.model.resize_token_embeddings(total_vocab_size)

        # 🔸 赋值新嵌入
        start_idx = self.original_len
        end_idx = start_idx + self.n_embed
        self.model.model.embed_tokens.weight.data[start_idx:end_idx] = ts_weight

        start_idx = end_idx
        end_idx = start_idx + special_tokens_len
        self.model.model.embed_tokens.weight.data[start_idx:end_idx] = special_tokens_weight

        # 🔸 保存 embedding 权重以供输出层使用
        self.embedding_weight = self.model.model.embed_tokens.weight


    def _initialize_output_layer(self, config):
        # 创建输出层，与embedding layer共享权重
        output_layer = nn.Linear(config.hidden_size, self.embedding_weight.size(0), bias=False)
        # 使用embedding layer的权重初始化输出层
        output_layer.weight.data = self.embedding_weight.data
        
        # 替换Qwen模型的输出层
        self.model.set_output_embeddings(output_layer)
        self.model.lm_head.weight = self.model.model.embed_tokens.weight
        
    def forward(self, inputs):    
        text_ids, input_ids, labels = inputs['text_ids'], inputs['ts_ids'], inputs['labels']
        device = input_ids.device

        # 构造 attention mask
        attention_mask = torch.ones(input_ids.shape[0], input_ids.shape[1], dtype=torch.float32, device=device)

        
        new_token_weight = self.configs.new_token_weight if hasattr(self.configs, 'new_token_weight') else 1
        orig_token_weight = 1
        position_weights = torch.where(labels >= self.original_len, new_token_weight, orig_token_weight)
        ts_start_id = self.text_tokenizer.convert_tokens_to_ids("<TS_START>")
        ts_end_id = self.text_tokenizer.convert_tokens_to_ids("<TS_END>")

        
        # 🚀 正确地传 input_ids，别用 inputs_embeds！
        outputs = self.model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        loss_fn = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean')
        loss = loss_fn(outputs.logits[..., :-1, :], labels[..., 1:], position_weights[..., 1:])
        outputs.loss = loss
        
        return outputs


    def gen_ts(self, inputs, text_token_len=112,ts_token_len=12):
        

        tokenizer = self.text_tokenizer
        device = next(self.model.parameters()).device
        original_len = self.original_len  # 原始词表长度
        n_ts_token = self.n_embed
        ts_token_range = (original_len, original_len + n_ts_token)

        input_ids = inputs['ts_ids']
        
        device = next(self.model.parameters()).device
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
        input_ids = input_ids.to(device)

        ts_end_token_id = tokenizer.convert_tokens_to_ids("<TS_END>")
        ts_start_token_id = tokenizer.convert_tokens_to_ids("<TS_START>")
        max_len = text_token_len+ts_token_len+2

        logits_processor = LogitsProcessorList([
            TsTokenFormatController(
                ts_token_range=ts_token_range,     # 假设时序token id是这个范围
                ts_start_token_id=ts_start_token_id,          # <TS_START>
                ts_end_token_id=ts_end_token_id ,            # <TS_END>
                ts_start_pos=text_token_len+input_ids.shape[1],                  # 文本 token 为前 32 个
                ts_len=ts_token_len                     # 时序 token 输出长度
            )
        ])

        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_len,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            logits_processor=logits_processor
        )

        return generated.sequences[:, input_ids.shape[1]:]


    @staticmethod
    def init_weights_kaiming(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            m.bias.data.fill_(0.01)

