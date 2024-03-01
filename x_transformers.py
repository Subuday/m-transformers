import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from attend import Attend
from helpers import default, exists

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)


def apply_rotary_pos_emb(t, freqs, scale = 1):
    return (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.g


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        self.block = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x):
        return self.block(x)
    

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        num_heads = 8,
        flash = False,
        gate_per_head = False,
        num_kv_heads = None
    ):
        super().__init__()
        self.num_heads = num_heads
        num_kv_heads = default(num_kv_heads, num_heads)
        self.num_kv_heads = num_kv_heads

        q_dim = dim_head * num_heads
        k_dim = dim_head * num_kv_heads
        v_dim = dim_head * num_kv_heads
        out_dim = dim_head * num_heads

        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, q_dim, bias = False)
        self.to_k = nn.Linear(dim, k_dim, bias = False)
        self.to_v = nn.Linear(dim, v_dim, bias = False) 
        
        self.to_qkv = nn.Linear(dim, dim_head * 3, bias = False)
        self.attend = Attend(
            scale = self.scale,
            flash = flash
        )

        # per head gating, from https://arxiv.org/abs/2306.12929
        self.gate_per_head = None
        if gate_per_head:
            self.gate_per_head = nn.Linear(dim, num_heads)
            nn.init.constant_(self.gate_per_head.weight, 0)
            nn.init.constant_(self.gate_per_head.bias, 10)

        self.to_out = nn.Linear(out_dim, dim, bias = False)


    def forward(self, x, rotary_pos_emb = None):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.num_heads)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_kv_heads), (k, v))

        if exists(rotary_pos_emb):
            q = apply_rotary_pos_emb(q, rotary_pos_emb)
            k = apply_rotary_pos_emb(k, rotary_pos_emb)

        out = self.attend(q, k, v)

        # per head gating, from https://arxiv.org/abs/2306.12929
        if exists(self.gate_per_head):
            gated_heads = self.gate_per_head(x)
            gated_heads = rearrange(gated_heads, 'b n h -> b h n 1').sigmoid()
            out = out * gated_heads

        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.to_out(out)
        return out


class AttentionLayers(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        num_heads = 8,
        attn_flash = False,
        attn_gate_per_head = False,
        attn_num_kv_heads = None,
        ff_mult = 4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    RMSNorm(dim = dim),
                    Attention(
                        dim = dim,
                        dim_head = dim_head,
                        num_heads = num_heads,
                        flash = attn_flash,
                        gate_per_head = attn_gate_per_head,
                        num_kv_heads = attn_num_kv_heads
                    ),
                    RMSNorm(dim = dim),
                    FeedForward(dim = dim, mult = ff_mult)
                ])
            )
        self.out_norm = RMSNorm(dim)

    def forward(self, x, rotary_pos_emb = None):
        for attn_norm, attn, ff_norm, ff in self.layers:
            attn_input = attn_norm(x)
            x = attn(attn_input, rotary_pos_emb = rotary_pos_emb) + x
            
            ff_input = ff_norm(x)
            x = ff(ff_input) + x
        
        out = self.out_norm(x)
        return out


class Decoder(AttentionLayers):
    def __init__(self, **kwargs):
        super().__init__(causal = True, **kwargs)

