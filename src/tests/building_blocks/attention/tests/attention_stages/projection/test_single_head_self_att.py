
import torch
import unittest

from tests.custom_base_test import CustomModuleBaseTest
from mypt.building_blocks.attention.attention_stages.projection import SingleHeadSelfAttentionProjection

class TestSingleHeadSelfAttentionProjection(CustomModuleBaseTest):
    def setUp(self) -> None:
        self.input_dim = 32
        self.key_dim = 64
        self.value_dim = 64

        self.block = SingleHeadSelfAttentionProjection(
            input_dim=self.input_dim,
            key_dim=self.key_dim,
            value_dim=self.value_dim
        )

        self.num_iterations = 100

    def _get_valid_input(self, batch_size: int = 2, seq_len: int = 10):
        return torch.randn(batch_size, seq_len, self.input_dim)


    # SingleHeadSelfAttentionProjection-specific tests
    def test_forward_output_shape(self):
        for _ in range(self.num_iterations):
            input_tensor = self._get_valid_input()
            q, k, v = self.block(input_tensor)
            self.assertEqual(q.shape, (2, 1, 10, self.key_dim))
            self.assertEqual(k.shape, (2, 1, 10, self.key_dim))
            self.assertEqual(v.shape, (2, 1, 10, self.value_dim))

    
    # general tests
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


    # gradient related tests
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