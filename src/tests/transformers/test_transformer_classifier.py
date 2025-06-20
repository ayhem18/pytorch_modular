import torch
import unittest
import numpy as np

from tests.custom_base_test import CustomModuleBaseTest
from mypt.transformers.transformer_classifier import TransformerClassifier, _POOLING_REGISTRY


class TestTransformerClassifier(CustomModuleBaseTest):
    def setUp(self):
        self.d_model = 32
        self.num_heads = 4
        self.key_dim = 8
        self.value_dim = 8
        self.num_blocks = 2
        self.num_classes = 3
        self.num_cls_layers = 2

    # helper to create model
    def _make_model(self, pooling: str) -> TransformerClassifier:
        return TransformerClassifier(
            d_model=self.d_model,
            num_transformer_blocks=self.num_blocks,
            num_classification_layers=self.num_cls_layers,
            num_heads=self.num_heads,
            value_dim=self.value_dim,
            key_dim=self.key_dim,
            num_classes=self.num_classes,
            pooling=pooling,
        )

    # helper input
    def _get_valid_input(self, batch: int = 4, seq_len: int = 10):
        return torch.randn(batch, seq_len, self.d_model)

    def _get_pad_mask(self, batch: int, seq_len: int):
        mask = (torch.rand(batch, seq_len) > 0.3).int()
        for b in range(batch):
            if mask[b].sum() == 0:
                mask[b, 0] = 1
        return mask

    # iterate over pooling strategies ---------------------------------
    def test_all_pooling_variants(self):
        for pooling in _POOLING_REGISTRY.keys():
            model = self._make_model(pooling)
            self._run_common_tests(model, pooling)

    # internal common test suite
    def _run_common_tests(self, model: TransformerClassifier, pooling: str):
        inp = self._get_valid_input()
        pad_mask = self._get_pad_mask(inp.size(0), inp.size(1))

        # verify pooling property
        self.assertEqual(model.get_pooling_type(), pooling)

        # shape check
        out = model(inp, pad_mask)
        self.assertEqual(out.shape, (inp.size(0), self.num_classes))

        # base tests
        self._test_module_is_nn_module(model)
        self._test_eval_mode(model)
        self._test_train_mode(model)
        self._test_consistent_output_in_eval_mode(model, inp, pad_mask)
        self._test_named_parameters_length(model)
        self._test_batch_size_one_in_eval_mode(model, self._get_valid_input(1, 7), self._get_pad_mask(1, 7))


if __name__ == "__main__":
    import mypt.code_utils.pytorch_utils as pu

    pu.seed_everything(0)
    unittest.main() 