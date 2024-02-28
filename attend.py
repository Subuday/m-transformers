from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attend(nn.Module):
    def __init__(
        self,
        scale,
        flash = False,
    ):
        super().__init__()
        self.scale = scale
        self.flash = flash

    @staticmethod
    def _create_mask(n, m, device):
        return torch.ones((n, m), device = device, dtype = torch.bool).triu(m - n + 1)

    def flash_attn(self, q, k, v):
        sdp_kwargs: dict = dict(
            enable_flash = True,
            enable_math = True,
            enable_mem_efficient = True
        )
        with torch.backends.cuda.sdp_kernel(**sdp_kwargs):
            out = F.scaled_dot_product_attention(q, k, v, scale = self.scale)
        return out

    def forward(self, q, k, v):
        num_heads, num_kv_heads = q.shape[1], k.shape[1]

        # handle grouped multi-query attention
        if num_kv_heads == 1:
            k, v = map(lambda t: rearrange(t, 'b 1 n d -> b n d'), (k, v))
        elif num_kv_heads < num_heads:
            k, v = map(lambda t: repeat(t, 'b kvh n d -> b (r kvh) n d', r = num_heads // num_kv_heads), (k, v))

        if self.flash:
            return self.flash_attn(q, k, v)

        device = q.device

        eq = 'b j d' if k.ndim == 3 else 'b h j d'
        qk = torch.einsum(f'b h i d, {eq} -> b h i j', q, k)
        qk = qk * self.scale

        mask = self._create_mask(qk.shape[-2], qk.shape[-1], device)
        # TODO: Recheck if it can create problems with other types
        mask_value = -torch.finfo(qk.dtype).max
        qk = qk.masked_fill(mask, mask_value)

        # TODO: Check if need to use float32
        qk_type = qk.dtype
        qk = F.softmax(qk, dim = -1, dtype = torch.float32)
        qk = qk.type(qk_type)

        out = torch.einsum(f'b h i j, {eq} -> b h i d', qk, v)
        return out