import torch
import unittest
import numpy as np

from tests.custom_base_test import CustomModuleBaseTest
from mypt.transformers.transformer_block import TransformerBlock


class TestTransformerBlock(CustomModuleBaseTest):
    def setUp(self) -> None:  # noqa: D401
        self.d_model = 32
        self.num_heads = 4
        self.key_dim = 8
        self.value_dim = 8
        self.block = TransformerBlock(self.d_model, self.num_heads, self.value_dim, self.key_dim)

    # --- helper ---
    def _get_valid_input(self, batch: int = 4, seq_len: int = 10) -> torch.Tensor:
        return torch.randn(batch, seq_len, self.d_model)

    # --- tests ---
    def test_structure_children(self):
        expected_order = [self.block.ln1, self.block.att, self.block.ln2, self.block.ffn]
        self.assertEqual(list(self.block.children()), expected_order,
                         "children() should return ln1, att, ln2, ffn in order")

        expected_named = [('ln1', self.block.ln1),
                          ('att', self.block.att),
                          ('ln2', self.block.ln2),
                          ('ffn', self.block.ffn)]
        self.assertEqual(list(self.block.named_children()), expected_named,
                         "named_children() should return correct mapping")

    def test_output_shape(self):
        x = self._get_valid_input()
        out = self.block(x)
        self.assertEqual(out.shape, x.shape, "Output shape should match input shape")

    # ---- inherited checks ----
    def test_common_base(self):
        inp = self._get_valid_input()
        self._test_module_is_nn_module(self.block)
        self._test_eval_mode(self.block)
        self._test_train_mode(self.block)
        self._test_consistent_output_without_dropout_bn(self.block, inp)
        self._test_consistent_output_in_eval_mode(self.block, inp)
        self._test_batch_size_one_in_eval_mode(self.block, self._get_valid_input(1, 5))
        self._test_named_parameters_length(self.block)


if __name__ == '__main__':
    import mypt.code_utils.pytorch_utils as pu
    pu.seed_everything(0)
    unittest.main() 