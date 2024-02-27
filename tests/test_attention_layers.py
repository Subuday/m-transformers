import unittest
import torch

from x_transformers import AttentionLayers

torch.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TestAttentionLayers(unittest.TestCase):

    @staticmethod
    def _create_input(batch_size):
        x = torch.rand((batch_size, 32, 512)).to(device)
        return x

    def _test_forward(self, batch_size, attn_flash = False, attn_gate_per_head = False):
        x = self._create_input(batch_size)
        attn_layers = AttentionLayers(
            dim = 512,
            depth = 6, 
            dim_head = 64, 
            num_heads = 8, 
            attn_flash = attn_flash,
            attn_gate_per_head = attn_gate_per_head,
            ff_mult = 4
        ).to(device)
        _ = attn_layers(x)

    def test_forward(self):
        self._test_forward(1)
        self._test_forward(3)

    def test_flash_attn(self):
        self._test_forward(3, attn_flash = True)

    def test_gate_per_heads(self):
        self._test_forward(3, attn_gate_per_head = True)