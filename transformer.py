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


class TransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        attn_layers: AttentionLayers,
        emb_dim = 512,
    ):
        super.__init__()
        dim = attn_layers.dim
        
        self.token_emb = TokenEmbedding(emb_dim, num_tokens)
        self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len)
        self.attn_layers = attn_layers
        self.to_logits = nn.Linear(dim, num_tokens, bias = False) 

    def forward(self, x):
        x = self.token_emb(x) + self.pos_emb(x)
        x = self.attn_layers(x)
        out = self.to_logits(x)
        return out