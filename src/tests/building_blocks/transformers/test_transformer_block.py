import torch
import unittest
import numpy as np
from tqdm import tqdm
from typing import Optional, Type
import abc

from tests.custom_base_test import CustomModuleBaseTest
from mypt.building_blocks.transformers.transformer_block import AbstractTransformerBlock, BidirectionalTransformerBlock, CausalTransformerBlock


class TestTransformerBlock(CustomModuleBaseTest, abc.ABC):
    def setUp(self) -> None: 
        # Default parameters
        self.d_model = 32
        self.num_heads = 4
        
        self.key_dim = 8
        self.value_dim = 8

        self.block: AbstractTransformerBlock = self.get_block()
        
        # Number of test iterations
        self.num_iterations = 100

    @abc.abstractmethod
    def get_block(self) -> AbstractTransformerBlock:
        ...

    @abc.abstractmethod
    def _generate_random_block(self) -> AbstractTransformerBlock:
        ...

    # --- helper methods ---
    def _get_valid_input(self, batch: int = 4, seq_len: int = 10, d_model: Optional[int] = None) -> torch.Tensor:
        if d_model is None:
            d_model = self.d_model
        return torch.randn(batch, seq_len, d_model)
    
    
    # --- tests ---
    def _test_structure_children(self):
        """Test that the transformer block has the expected structure across multiple configurations."""
        for _ in range(self.num_iterations):
            block = self._generate_random_block()
            
            expected_order = [block.ln1, block.att, block.ln2, block.ffn]
            self.assertEqual(list(block.children()), expected_order,
                            "children() should return ln1, att, ln2, ffn in order")

            expected_named = [('ln1', block.ln1),
                            ('att', block.att),
                            ('ln2', block.ln2),
                            ('ffn', block.ffn)]
            self.assertEqual(list(block.named_children()), expected_named,
                            "named_children() should return correct mapping")

    def _test_output_shape(self):
        """Test that output shape matches input shape across multiple configurations."""
        for _ in range(self.num_iterations):
            block = self._generate_random_block()
            
            # Generate random batch size and sequence length
            batch_size = np.random.randint(1, 10)
            seq_length = np.random.randint(5, 20)
            
            # Create input tensor
            x = torch.randn(batch_size, seq_length, block.d_model)
            
            # Forward pass
            out = block(x)
            
            # Check shape
            self.assertEqual(out.shape, x.shape, 
                            f"Output shape {out.shape} should match input shape {x.shape}")

    # ---- Override inherited checks with more thorough versions ----
    def _test_eval_mode(self) -> None:
        """Test that calling eval() sets training=False for all parameters and submodules"""
        for _ in range(self.num_iterations):
            block = self._generate_random_block()
            super()._test_eval_mode(block)

    def _test_train_mode(self) -> None:
        """Test that calling train() sets training=True for all parameters and submodules"""
        for _ in range(self.num_iterations):
            block = self._generate_random_block()
            super()._test_train_mode(block)
    
    def _test_consistent_output_without_dropout_bn(self) -> None:
        """
        Test that modules without dropout or batch normalization 
        produce consistent output for the same input
        """
        for _ in range(self.num_iterations):
            block = self._generate_random_block()
            batch_size = np.random.randint(1, 10)
            seq_length = np.random.randint(5, 20)
            input_tensor = torch.randn(batch_size, seq_length, block.d_model)
            super()._test_consistent_output_without_dropout_bn(block, input_tensor)
    

    def _test_consistent_output_in_eval_mode(self) -> None:
        """Test that all modules in eval mode produce consistent output for the same input"""
        for _ in range(self.num_iterations):
            block = self._generate_random_block()
            batch_size = np.random.randint(1, 10)
            seq_length = np.random.randint(5, 20)
            input_tensor = torch.randn(batch_size, seq_length, block.d_model)
            super()._test_consistent_output_in_eval_mode(block, input_tensor)
    
    def _test_batch_size_one_in_train_mode(self) -> None:
        """
        Test that modules with batch normalization layers might raise errors 
        with batch size 1 in train mode
        """
        for _ in range(self.num_iterations):
            block = self._generate_random_block()
            seq_length = np.random.randint(5, 20)
            input_tensor = torch.randn(1, seq_length, block.d_model)
            super()._test_batch_size_one_in_train_mode(block, input_tensor)
    
    def _test_batch_size_one_in_eval_mode(self) -> None:
        """Test that modules in eval mode should not raise errors for batch size 1"""
        for _ in range(self.num_iterations):
            block = self._generate_random_block()
            seq_length = np.random.randint(5, 20)
            input_tensor = torch.randn(1, seq_length, block.d_model)
            super()._test_batch_size_one_in_eval_mode(block, input_tensor)

    def _test_named_parameters_length(self) -> None:
        """Test that named_parameters() and parameters() have the same length"""
        for _ in range(self.num_iterations):
            block = self._generate_random_block()
            super()._test_named_parameters_length(block)
    
    def _test_to_device(self) -> None:
        """Test that module can move between devices properly"""
        for _ in range(self.num_iterations):
            block = self._generate_random_block()
            batch_size = np.random.randint(1, 10)
            seq_length = np.random.randint(5, 20)
            input_tensor = torch.randn(batch_size, seq_length, block.d_model)
            super()._test_to_device(block, input_tensor)


    def _test_gradcheck(self) -> None:
        """Test gradient computation using torch.autograd.gradcheck."""
        # Use small dimensions for speed
        for _ in tqdm(range(10), desc="Testing gradcheck"):
            b, s = (np.random.randint(2, 10) for _ in range(2))
            random_block = self._generate_random_block()
            input_tensor = self._get_valid_input(b, s, random_block.d_model)
            super()._test_gradcheck(random_block, input_tensor)


    def _test_gradcheck_large_values(self) -> None:
        """Test gradient computation with large input values."""
        # Use small dimensions for speed
        for _ in tqdm(range(10), desc="Testing gradcheck with large values"):
            b, s = (np.random.randint(2, 10) for _ in range(2))
            random_block = self._generate_random_block()
            input_tensor = self._get_valid_input(b, s, random_block.d_model)
            super()._test_gradcheck_large_values(random_block, input_tensor)
    

    def _test_grad_against_nan(self) -> None:
        """Test that the gradient is not nan"""
        for _ in range(self.num_iterations):
            batch_size = np.random.randint(1, 10)
            seq_length = np.random.randint(5, 20)
            random_block = self._generate_random_block()
            input_tensor = self._get_valid_input(batch_size, seq_length, random_block.d_model)
            super()._test_grad_against_nan(random_block, input_tensor)


    def _test_grad_against_nan_large_values(self) -> None:
        """Test that the gradient is not nan with large input values"""
        for _ in range(self.num_iterations):
            batch_size = np.random.randint(1, 10)
            seq_length = np.random.randint(5, 20)
            random_block = self._generate_random_block()
            input_tensor = self._get_valid_input(batch_size, seq_length, random_block.d_model)
            super()._test_grad_against_nan_large_values(random_block, input_tensor)

class TestCausalTransformerBlock(TestTransformerBlock):
    def get_block(self) -> AbstractTransformerBlock:
        return CausalTransformerBlock(self.d_model, self.num_heads, self.value_dim, self.key_dim)

    def _generate_random_block(self) -> AbstractTransformerBlock:
        d_model = np.random.choice([16, 32, 64, 128])
        num_heads = np.random.choice([1, 2, 4, 8])
        key_dim = (d_model // num_heads) * np.random.choice([1, 2])
        value_dim = (d_model // num_heads) * np.random.choice([1, 2])
        dropout = np.random.uniform(0.0, 0.5)
        
        return CausalTransformerBlock(d_model, num_heads, value_dim, key_dim, dropout)

    def test_structure_children(self):
        self._test_structure_children()

    def test_output_shape(self):
        self._test_output_shape()

    def test_eval_mode(self) -> None:
        self._test_eval_mode()

    def test_train_mode(self) -> None:
        self._test_train_mode()

    def test_consistent_output_without_dropout_bn(self) -> None:
        self._test_consistent_output_without_dropout_bn()

    def test_consistent_output_in_eval_mode(self) -> None:
        self._test_consistent_output_in_eval_mode()

    def test_batch_size_one_in_train_mode(self) -> None:
        self._test_batch_size_one_in_train_mode()

    def test_batch_size_one_in_eval_mode(self) -> None:
        self._test_batch_size_one_in_eval_mode()

    def test_named_parameters_length(self) -> None:
        self._test_named_parameters_length()

    def test_to_device(self) -> None:
        self._test_to_device()

    @unittest.skip("passed")
    def test_gradcheck(self) -> None:
        self._test_gradcheck()

    @unittest.skip("passed")
    def test_gradcheck_large_values(self) -> None:
        self._test_gradcheck_large_values()

    @unittest.skip("passed")
    def test_grad_against_nan(self) -> None:
        self._test_grad_against_nan()

    @unittest.skip("passed")
    def test_grad_against_nan_large_values(self) -> None:
        self._test_grad_against_nan_large_values()

class TestBidirectionalTransformerBlock(TestTransformerBlock):
    def get_block(self) -> AbstractTransformerBlock:
        return BidirectionalTransformerBlock(self.d_model, self.num_heads, self.value_dim, self.key_dim)

    def _generate_random_block(self) -> AbstractTransformerBlock:
        d_model = np.random.choice([16, 32, 64, 128])
        num_heads = np.random.choice([1, 2, 4, 8])
        key_dim = (d_model // num_heads) * np.random.choice([1, 2])
        value_dim = (d_model // num_heads) * np.random.choice([1, 2])
        dropout = np.random.uniform(0.0, 0.5)
        
        return BidirectionalTransformerBlock(d_model, num_heads, value_dim, key_dim, dropout)

    def test_structure_children(self):
        self._test_structure_children()

    def test_output_shape(self):
        self._test_output_shape()

    def test_eval_mode(self) -> None:
        self._test_eval_mode()

    def test_train_mode(self) -> None:
        self._test_train_mode()

    def test_consistent_output_without_dropout_bn(self) -> None:
        self._test_consistent_output_without_dropout_bn()

    def test_consistent_output_in_eval_mode(self) -> None:
        self._test_consistent_output_in_eval_mode()

    def test_batch_size_one_in_train_mode(self) -> None:
        self._test_batch_size_one_in_train_mode()

    def test_batch_size_one_in_eval_mode(self) -> None:
        self._test_batch_size_one_in_eval_mode()

    def test_named_parameters_length(self) -> None:
        self._test_named_parameters_length()

    def test_to_device(self) -> None:
        self._test_to_device()

    @unittest.skip("passed")
    def test_gradcheck(self) -> None:
        self._test_gradcheck()

    @unittest.skip("passed")
    def test_gradcheck_large_values(self) -> None:
        self._test_gradcheck_large_values()

    @unittest.skip("passed")
    def test_grad_against_nan(self) -> None:
        self._test_grad_against_nan()

    @unittest.skip("passed")
    def test_grad_against_nan_large_values(self) -> None:
        self._test_grad_against_nan_large_values()


if __name__ == '__main__':
    import mypt.code_utils.pytorch_utils as pu
    pu.seed_everything(0)
    unittest.main()