import torch
import torch.nn as nn
import torch.nn.functional as F

class Attend(nn.Module):
    def __init__(
        self,
        scale,
    ):
        super().__init__()
        self.scale = scale

    @staticmethod
    def _create_mask(n, m, device):
        return torch.ones((n, m), device = device, dtype = torch.bool).triu(m - n + 1)

    def forward(self, q, k, v):
        device = q.device

        qk = torch.einsum('b h i d, b h j d -> b h i j', k, v)
        qk = qk * self.scale

        mask = self._create_mask(qk.shape[-2], qk.shape[-1], device)
        # TODO: Recheck if it can create problems with other types
        mask_value = -torch.finfo(qk.dtype).max
        qk = qk.masked_fill(mask, mask_value)

        # TODO: Check if need to use float32
        qk_type = qk.dtype
        qk = F.softmax(qk, dim = -1, dtype = torch.float32)
        qk = qk.type(qk_type)

        out = torch.einsum('b h i j, b h j d -> b h i d', qk, v)
        return out