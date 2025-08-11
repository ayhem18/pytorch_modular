import torch
import random
import unittest

from tests.custom_base_test import CustomModuleBaseTest
from mypt.building_blocks.attention.attention_stages.projection import MultiHeadSelfAttentionProjection

class TestMultiHeadSelfAttentionProjection(CustomModuleBaseTest):
    def setUp(self) -> None:
        self.d_model = 32
        self.num_heads = 4
        self.key_dim = 64
        self.value_dim = 64

        self.block = MultiHeadSelfAttentionProjection(
            d_model=self.d_model,
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            value_dim=self.value_dim
        )

        self.num_iterations = 100

    def _get_valid_input(self, batch_size: int = 2, seq_len: int = 10):
        return torch.randn(batch_size, seq_len, self.d_model)


    def test_forward_output_shape(self):
        for _ in range(self.num_iterations):
            input_tensor = self._get_valid_input()
            q, k, v = self.block(input_tensor)
            self.assertEqual(q.shape, (2, self.num_heads, 10, self.key_dim))
            self.assertEqual(k.shape, (2, self.num_heads, 10, self.key_dim))
            self.assertEqual(v.shape, (2, self.num_heads, 10, self.value_dim))
        
    def test_value_error_if_d_model_not_divisible_by_num_heads(self):
        for _ in range(self.num_iterations):
            num_heads = random.randint(4, 16)
            d_model = num_heads * random.randint(2, 5) + random.randint(1, num_heads - 1)
            with self.assertRaises(ValueError):
                MultiHeadSelfAttentionProjection(
                    d_model=d_model,
                    num_heads=num_heads,
                    key_dim=64,
                    value_dim=64
                )


    def test_module_is_nn_module(self):
        self._test_module_is_nn_module(self.block)

    def test_eval_mode(self):
        self._test_eval_mode(self.block)

    def test_train_mode(self):
        self._test_train_mode(self.block)

    def test_consistent_output_in_eval_mode(self):
        for _ in range(self.num_iterations):
            input_tensor = self._get_valid_input()
            self._test_consistent_output_in_eval_mode(self.block, input_tensor)

    def test_batch_size_one_in_eval_mode(self):
        for _ in range(self.num_iterations):
            input_tensor = self._get_valid_input(batch_size=1)
            self._test_batch_size_one_in_eval_mode(self.block, input_tensor)

    def test_batch_size_one_in_train_mode(self):
        for _ in range(self.num_iterations):
            input_tensor = self._get_valid_input(batch_size=1)
            self._test_batch_size_one_in_train_mode(self.block, input_tensor)

    def test_named_parameters_length(self):
        self._test_named_parameters_length(self.block)

    def test_to_device(self):
        for _ in range(self.num_iterations):
            input_tensor = self._get_valid_input()
            self._test_to_device(self.block, input_tensor)

    @unittest.skip("Skipping nan test for now")
    def test_gradcheck(self):
        for _ in range(self.num_iterations):    
            input_tensor = self._get_valid_input()
            self._test_gradcheck(self.block, input_tensor)

    @unittest.skip("Skipping large values test for now")    
    def test_gradcheck_large_values(self):
        for _ in range(self.num_iterations):
            input_tensor = self._get_valid_input()
            self._test_gradcheck_large_values(self.block, input_tensor)

    @unittest.skip("Skipping nan test for now")
    def test_grad_against_nan(self):
        for _ in range(self.num_iterations):
            input_tensor = self._get_valid_input()
            self._test_grad_against_nan(self.block, input_tensor)

    @unittest.skip("Skipping large values test for now")
    def test_grad_against_nan_large_values(self):
        for _ in range(self.num_iterations):
            input_tensor = self._get_valid_input()
            self._test_grad_against_nan_large_values(self.block, input_tensor)


if __name__ == "__main__":
    from mypt.code_utils.pytorch_utils import seed_everything
    seed_everything(42)
    unittest.main()