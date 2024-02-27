import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from attend import Attend
from helpers import exists

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
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5

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

        self.to_out = nn.Linear(dim_head, dim, bias = False)


    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), (q, k, v))

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
        ff_mult = 4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    RMSNorm(dim = dim),
                    Attention(dim = dim, dim_head = dim_head, num_heads = num_heads, flash = attn_flash, gate_per_head = attn_gate_per_head),
                    RMSNorm(dim = dim),
                    FeedForward(dim = dim, mult = ff_mult)
                ])
            )
        self.out_norm = RMSNorm(dim)

    def forward(self, x):
        for attn_norm, attn, ff_norm, ff in self.layers:
            attn_input = attn_norm(x)
            x = attn(attn_input) + x
            
            ff_input = ff_norm(x)
            x = ff(ff_input) + x
        
        out = self.out_norm(x)
        return out


class Decoder(AttentionLayers):
    def __init__(self, **kwargs):
        super().__init__(causal = True, **kwargs)

