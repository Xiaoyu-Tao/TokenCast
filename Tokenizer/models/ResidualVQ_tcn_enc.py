import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, constant_
# from layers.Transformer_EncDec import Encoder, EncoderLayer
# from layers.SelfAttention_Family import FullAttention, AttentionLayer
# from vector_quantize_pytorch import ResidualVQ
from models.RevIN import RevIN
from einops import rearrange
import torch
from torch import nn
from torch.nn.utils import weight_norm
from models.CasualTRM import CasualTRM

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

class Quantize(nn.Module):
    def __init__(self, dim, n_embed,configs, beta=0.25, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.beta = beta
        self.entropy_penalty = configs.entropy_penalty
        self.entropy_temp = configs.entropy_temp
        self.eps = eps

        self.embedding = nn.Embedding(n_embed, dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=dim ** -0.5)
        self.embedding_proj = nn.Linear(dim, dim)

    def forward(self, input):
        B, T, C = input.shape
        flatten = input.reshape(-1, C)

        # if self.training:
        #     flatten = flatten + 0.01 * torch.randn_like(flatten)

        # codebook projection
        codebook = self.embedding_proj(self.embedding.weight)  # [n_embed, dim]

        # compute distance
        d = torch.sum(flatten ** 2, dim=1, keepdim=True) + \
            torch.sum(codebook ** 2, dim=1) - 2 * torch.matmul(flatten, codebook.t())  # [B*T, n_embed]

        # soft assignment (for encoder entropy loss)
        logits = -d / self.entropy_temp
        probs = F.softmax(logits, dim=-1)
        soft_entropy = -torch.sum(probs * torch.log(probs + self.eps), dim=-1).mean()
        max_entropy = np.log(self.n_embed)
        norm_soft_entropy = soft_entropy / max_entropy
        soft_entropy_loss = self.entropy_penalty * (1.0 - norm_soft_entropy)

        # hard assignment
        indices = torch.argmax(probs, dim=-1)  # [B*T]
        z_q = F.embedding(indices, codebook).view(B, T, C)

        # commitment and embedding loss
        diff_loss = F.mse_loss(z_q.detach(), input)
        commit_loss = F.mse_loss(z_q, input.detach())
        vq_loss = diff_loss + self.beta * commit_loss

        # additional hard token usage entropy (non-differentiable, optional)
        with torch.no_grad():
            one_hot = F.one_hot(indices, num_classes=self.n_embed).float()  # [B*T, n_embed]
            avg_probs = one_hot.mean(dim=0) + self.eps
            token_usage_entropy = -torch.sum(avg_probs * torch.log(avg_probs))
            token_usage_max = torch.log(torch.tensor(self.n_embed, dtype=token_usage_entropy.dtype, device=token_usage_entropy.device))
            norm_token_usage_entropy = token_usage_entropy / token_usage_max
            token_entropy_loss = self.entropy_penalty * (1.0 - norm_token_usage_entropy)

        # total loss (only soft_entropy_loss is differentiable w.r.t. encoder)
        total_loss = vq_loss + soft_entropy_loss + token_entropy_loss

        # straight-through estimator
        z_q = input + (z_q - input).detach()

        return z_q, total_loss, indices


    def embed_code(self, embed_id):
        embedding = F.embedding(embed_id, self.embed.transpose(0, 1))
        return self.embedding_proj(embedding)


# TCN from tsai
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, channel_in, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 1 ** i
            in_channels = channel_in if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



class Encoder(nn.Module):
    def __init__(self, chan_indep,channel_in, hidden_dim, block_num=3, kernel_size=3, dropout=0.2):
        super().__init__()
        self.chan_indep = chan_indep

        self.TCN = TemporalConvNet(channel_in, [hidden_dim]*block_num, kernel_size=kernel_size, dropout=dropout)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        if self.chan_indep:   
            x = x.reshape(-1, x.shape[-1]).unsqueeze(1)
        x = self.TCN(x)
        x = x.permute(0, 2, 1)
        return x
    
class Decoder(nn.Module):
    def __init__(self, patch_len, enc_in, hidden_dim, n_heads=4, block_num=3, dropout=0.2):
        super().__init__()
        self.decoder = CasualTRM(dim=hidden_dim, d_ff=hidden_dim*4,
                                 n_heads=n_heads, n_layers=block_num, dropout=dropout)

        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, patch_len * enc_in)
        )

        self.patch_len = patch_len
        self.enc_in = enc_in

    def forward(self, x):
        """
        x: [B, T, D]  → D = hidden_dim
        output: [B, patch_len * enc_in, 1] 
        """
        B, T, D = x.shape
        x, _ = self.decoder(x)  
        x = self.linear(x)             # x: [B, T, D]
        x = x.view(B, T * self.patch_len, self.enc_in) # [B, pred_len, enc_in]
       

        return x

class VQVAE(nn.Module):
    def __init__(
            self,
            configs
    ):
        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        total_len = self.seq_len + self.pred_len
        hidden_dim = configs.d_model
        n_embed = configs.n_embed
        block_num = configs.block_num
        self.patch_len = configs.wave_length
        
        self.revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        # Channel independent
        self.chan_indep = configs.chan_indep

        enc_in = configs.enc_in if configs.chan_indep == 0 else 1
        
        data_shape = (total_len, enc_in)
        
        self.enc = Encoder(self.chan_indep,enc_in, hidden_dim, block_num)
        wave_patch = (self.patch_len, hidden_dim)
        self.quantize_input = nn.Conv2d(1, hidden_dim, kernel_size=wave_patch, stride=wave_patch)
        self.quantize = Quantize(hidden_dim, n_embed,configs)
        self.dec = Decoder(self.patch_len,enc_in, hidden_dim)
        
        if self.revin:
            self.revin_layer = RevIN(enc_in, affine=affine, subtract_last=subtract_last)

    def forward(self, x,y):
        if self.revin:
            x_look_back = self.revin_layer(x, 'norm')
            x_pred = self.revin_layer._normalize(y)
            x = torch.cat([x_look_back, x_pred], dim=1)


        
        n_var = x.shape[-1]
        B = x.shape[0]
        enc = self.enc(x)
        enc = enc.unsqueeze(1)
        quant = self.quantize_input(enc).squeeze(-1).transpose(1, 2)

        quant, diff,ids  = self.quantize(quant)

        dec = self.dec(quant)
        if self.chan_indep:
            dec = dec.permute(0,2,1)
            dec = dec.reshape(-1, n_var, dec.shape[-1])
            dec = dec.permute(0,2,1)
        if self.revin:
            dec = self.revin_layer(dec, 'denorm')


        return dec, diff, ids
    def get_name(self):
        return 'r_vqvae'

    def get_embedding(self, x):
        enc = self.enc(x)
        enc = enc.unsqueeze(1)
        quant = self.quantize_input(enc).squeeze(-1).transpose(1, 2)
        quant, ids, diff = self.quantize(quant)
        return quant

    def get_ids(self, x):
        if self.revin:
            x = self.revin_layer(x, 'norm')
        enc = self.enc(x)
        enc = enc.unsqueeze(1)
        quant = self.quantize_input(enc).squeeze(-1).transpose(1, 2)
        quant, ids, diff = self.quantize(quant)
        return ids

    def decode_from_ids(self, look_back, ids):
        """ 
        lookback: [bs x seq_len x 1]
        ids: [bs x pred_num x codebook_num]
        """
        if self.revin:
            look_back = self.revin_layer(look_back, 'norm')
        enc = self.enc(look_back)
        enc = enc.unsqueeze(1)
        
        quant = self.quantize_input(enc).squeeze(-1).transpose(1, 2)
        quant, _, _ = self.quantize(quant)

        B, N, Q = ids.shape
        ids_t = ids.permute(2, 0, 1).reshape(Q, -1)  # (Q, B*N)
        quant_ids = torch.arange(Q, device=ids.device).unsqueeze(1).expand(Q, B*N)


        embeddings = self.quantize.codebooks[quant_ids, ids_t]  # (Q, B*N, D)
        embeddings = embeddings.permute(1, 0, 2).reshape(B, N, Q, -1)
        embedding = embeddings.sum(dim=-2)

        quant = torch.cat([quant, embedding], dim=1)
        dec = self.dec(quant)

        if self.revin:
            dec = self.revin_layer(dec, 'denorm')
        return dec[:, look_back.shape[1]:, :]
        
    def decode(self, quant):
        dec = self.dec(quant)
        return dec

