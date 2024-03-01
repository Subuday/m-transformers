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

    @staticmethod
    def _create_rotary_pos_emb(max_seq_len, dim_head):
        pos_emb = torch.rand((max_seq_len, dim_head)).to(device)
        return pos_emb

    def _test_forward(self, batch_size, attn_flash = False, attn_gate_per_head = False, attn_num_kv_heads = None, use_rotary_pos_emb = False):
        x = self._create_input(batch_size)
        attn_layers = AttentionLayers(
            dim = 512,
            depth = 6, 
            dim_head = 64, 
            num_heads = 8, 
            attn_flash = attn_flash,
            attn_gate_per_head = attn_gate_per_head,
            attn_num_kv_heads = attn_num_kv_heads,
            ff_mult = 4
        ).to(device)
        _ = attn_layers(
            x,
            rotary_pos_emb = self._create_rotary_pos_emb(x.shape[1], dim_head = 64) if use_rotary_pos_emb else None
        )

    def test_forward(self):
        self._test_forward(1)
        self._test_forward(3)

    def test_flash_attn(self):
        self._test_forward(3, attn_flash = True)

    def test_gate_per_heads(self):
        self._test_forward(3, attn_gate_per_head = True)

    def test_multi_query_attn(self):
        self._test_forward(3, attn_num_kv_heads = 1)

    def test_grouped_multi_query_attn(self):
        self._test_forward(3, attn_num_kv_heads = 4)

    def test_pos_rotary_emb(self):
        self._test_forward(3, use_rotary_pos_emb = True)