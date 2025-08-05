import torch
from torch import nn
# from layers.Transformer_EncDec import Encoder, EncoderLayer
import torch.nn.functional as F

HAS_FLASH = False


# Rotary Positional Embedding
def build_rope_cache(max_seq_len, head_dim, device):
    # head_dim must be even for RoPE
    half_dim = head_dim // 2
    freq_seq = torch.arange(half_dim, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000 ** (freq_seq / half_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)             # [max_seq_len, half_dim]
    emb = torch.cat((freqs, freqs), dim=-1)       # [max_seq_len, head_dim]
    cos = emb.cos()[None, :, None, :]             # [1, max_seq_len, 1, head_dim]
    sin = emb.sin()[None, :, None, :]
    return cos, sin


def apply_rope(x, cos, sin):
    # x: [B, seq_len, H, head_dim]
    # Split even/odd dims
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    cos_even = cos[..., ::2]
    sin_even = sin[..., ::2]
    # Rotate
    x_even_rot = x_even * cos_even - x_odd * sin_even
    x_odd_rot = x_even * sin_even + x_odd * cos_even
    # Interleave back
    # Stack and reshape
    x_rot = torch.stack([x_even_rot, x_odd_rot], dim=-1)  # [..., head_dim/2, 2]
    x_rot = x_rot.flatten(-2)  # [..., head_dim]
    return x_rot


class AttentionLayer(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1,
                 use_flash=True, use_rope=True, max_seq_len=2048):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"Embedding dim {dim} must be divisible by n_heads {n_heads}")
        head_dim = dim // n_heads
        if use_rope and (head_dim % 2 != 0):
            raise ValueError(f"head_dim ({head_dim}) must be even for RoPE")
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.use_flash = use_flash and HAS_FLASH
        self.use_rope = use_rope
        self.max_seq_len = max_seq_len

        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.out_dropout = nn.Dropout(dropout)
        self.dropout = dropout

        # 注册 RoPE 缓存
        self.register_buffer('cos_cache', None, persistent=False)
        self.register_buffer('sin_cache', None, persistent=False)

    def _update_rope_cache(self, seq_len, device):
        if self.cos_cache is None or self.cos_cache.shape[1] < seq_len:
            cos, sin = build_rope_cache(self.max_seq_len,
                                        self.head_dim,
                                        device)
            self.cos_cache = cos
            self.sin_cache = sin

    def forward(self, x, past_kv=None, attention_mask=None, use_cache=False):
        """
        x: [B, T, dim]
        past_kv: Tuple(k, v) or None, k/v shape = [B, P, H, head_dim]
        attention_mask: BoolTensor [B, T] or None (only for non-cache)
        use_cache: whether to return new (k, v)
        """
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.head_dim
        device = x.device

        # QKV projection, shape [B, T, 3, H, Dh]
        qkv = self.qkv_proj(x).view(B, T, 3, H, Dh)
        # split into q, k, v: each [B, T, H, Dh]
        q, k, v = qkv.unbind(2)

        # RoPE if enabled
        if self.use_rope:
            past_len = past_kv[0].shape[1] if past_kv is not None else 0
            total_len = past_len + T
            self._update_rope_cache(total_len, device)
            cos_all = self.cos_cache[:, :total_len, :, :]
            sin_all = self.sin_cache[:, :total_len, :, :]
            # apply to k and q
            k = apply_rope(k, cos_all[:, :k.shape[1]], sin_all[:, :k.shape[1]])
            q = apply_rope(q,
                           cos_all[:, past_len:past_len+T],
                           sin_all[:, past_len:past_len+T])

        # concatenate past K/V
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)  # [B, P+T, H, Dh]
            v = torch.cat([past_v, v], dim=1)

        # attention
        if self.use_flash:
            out = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True
            )
        else:
            q_ = q.permute(0, 2, 1, 3)
            k_ = k.permute(0, 2, 1, 3)
            v_ = v.permute(0, 2, 1, 3)
            # qkv shape [B, H, T, Dh]

            scores = (q_ @ k_.transpose(-2, -1)) * self.scale  # [B, H, T, P+T]
            # causal mask only in non-cache
            if not use_cache:
                key_len = k.shape[1]
                causal_mask = torch.arange(key_len, device=device)[None, :] <= (
                    torch.arange(T, device=device)[:, None]
                )
                # expand to [B, H, T, key_len]
                scores = scores.masked_fill(~causal_mask[None, None, :, :], float('-inf'))
            # padding mask
            if attention_mask is not None and not use_cache:
                mask2 = attention_mask[:, None, None, :].expand(B, 1, T, k.shape[1])
                scores = scores.masked_fill(~mask2, float('-inf'))
            probs = F.softmax(scores, dim=-1)
            out = probs @ v_  # [B, H, T, Dh]
            out = out.permute(0, 2, 1, 3)  # [B, T, H, Dh]

        # 5) output projection + dropout
        out = out.reshape(B, T, H * Dh)
        out = self.out_proj(out)
        out = self.out_dropout(out)

        return out, (k, v) if use_cache else None


class CasualLayer(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1,
                 ffn_hidden=None, use_flash=True,
                 use_rope=True, max_seq_len=2048):
        super().__init__()
        ffn_hidden = ffn_hidden or dim * 4
        self.ln1 = nn.LayerNorm(dim)
        self.attn = AttentionLayer(dim, n_heads,
                                   dropout, use_flash,
                                   use_rope, max_seq_len)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, past_kv=None, attention_mask=None, use_cache=False):
        # self-attn
        norm_x = self.ln1(x)
        attn_out, new_kv = self.attn(norm_x,
                                     past_kv=past_kv,
                                     attention_mask=attention_mask,
                                     use_cache=use_cache)
        x = x + attn_out
        # ffn
        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out
        return x, new_kv


class CasualTRM(nn.Module):
    def __init__(self, dim=768, d_ff=3072, n_heads=12, n_layers=12,
                 max_seq_len=2048, dropout=0.1,
                 use_flash=True, use_rope=True):
        super(CasualTRM, self).__init__()
        if dim % n_heads != 0:
            raise ValueError(f"Embedding dim {dim} must be divisible by n_heads {n_heads}")
        self.use_rope = use_rope
        # optional learned positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, dim)) if not use_rope else None
        self.layers = nn.ModuleList([
            CasualLayer(dim, n_heads, dropout,
                         d_ff, use_flash,
                         use_rope, max_seq_len)
            for _ in range(n_layers)
        ])
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, inputs_embeds=None,
                attention_mask=None, past_kv_list=None,
                use_cache=False):
        
        if use_cache and past_kv_list is not None:
            inputs_embeds = inputs_embeds[:, -1:, :].contiguous()
        x = inputs_embeds

        B, T, _ = x.shape
        # positional embeddings if not using RoPE
        if not self.use_rope:
            if use_cache and past_kv_list is not None:
                past_len = past_kv_list[0][0].size(1)
                pos_embed = self.pos_embed[:, past_len:past_len + T, :]
            else:
                pos_embed = self.pos_embed[:, :T, :]
            x = x + pos_embed
        new_past = []
        for i, layer in enumerate(self.layers):
            past_kv = past_kv_list[i] if past_kv_list else None
            x, new_kv = layer(x,
                              past_kv=past_kv,
                              attention_mask=attention_mask,
                              use_cache=use_cache)
            if use_cache:
                new_past.append(new_kv)
        return x, (new_past if use_cache else None)
