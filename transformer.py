from einops import repeat
import torch
import torch.nn as nn
from x_transformers import AttentionLayers

class TokenEmbedding(nn.Module):
    def __init__(self, dim, num_tokens):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, dim)

    def forward(self, x):
        token_emb = self.emb(x.long())
        return token_emb
    

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.scale = dim ** -0.5
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'
        pos = torch.arange(seq_len, device = device)
        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return pos_emb


class RotaryEmbedding(nn.Module):
    def __init__(
        self, 
        dim,
        base = 10000,
        base_rescale_factor = 1.
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        t = torch.arange(x.shape[1], device = x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq) / self.interpolation_factor
        freqs = torch.cat((freqs, freqs), dim = -1)
        freqs = repeat('n d -> n (repeat d)', repeat = 2)
        return freqs


class TransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        attn_layers: AttentionLayers,
        emb_dim = 512,
        use_rotary_pos_emb = False,
    ):
        super.__init__()
        dim = attn_layers.dim
        self.use_rotary_pos_emb = use_rotary_pos_emb
        
        self.token_emb = TokenEmbedding(emb_dim, num_tokens)
        self.pos_emb = RotaryEmbedding(attn_layers.dim_head) if use_rotary_pos_emb else AbsolutePositionalEmbedding(emb_dim, max_seq_len) 
        self.attn_layers = attn_layers
        self.to_logits = nn.Linear(dim, num_tokens, bias = False) 

    def forward(self, x):
        x = self.token_emb(x) 
        
        rotary_pos_emb = None
        if not self.use_rotary_pos_emb:
            x = self.pos_emb(x) + x
        else:
            rotary_pos_emb = self.pos_emb(x)

        x = self.attn_layers(x, rotary_pos_emb = rotary_pos_emb)
        out = self.to_logits(x)
        return out