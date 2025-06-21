import torch
import unittest
import numpy as np
from tqdm import tqdm

from tests.custom_base_test import CustomModuleBaseTest
from mypt.transformers.transformer_block import TransformerBlock


class TestTransformerBlock(CustomModuleBaseTest):
    def setUp(self) -> None:  # noqa: D401
        # Default parameters
        self.d_model = 32
        self.num_heads = 4
        self.key_dim = 8
        self.value_dim = 8
        self.block = TransformerBlock(self.d_model, self.num_heads, self.value_dim, self.key_dim)
        
        # Number of test iterations
        self.num_iterations = 100

    # --- helper methods ---
    def _get_valid_input(self, batch: int = 4, seq_len: int = 10) -> torch.Tensor:
        return torch.randn(batch, seq_len, self.d_model)
    
    def _generate_random_block(self) -> TransformerBlock:
        """Generate a random transformer block with valid parameters."""
        # Random dimensions that make sense
        d_model = np.random.choice([16, 32, 64, 128])
        num_heads = np.random.choice([1, 2, 4, 8])
        # Ensure key_dim and value_dim are divisible by num_heads
        key_dim = (d_model // num_heads) * np.random.choice([1, 2])
        value_dim = (d_model // num_heads) * np.random.choice([1, 2])
        dropout = np.random.uniform(0.0, 0.5)
        
        return TransformerBlock(d_model, num_heads, value_dim, key_dim, dropout)

    # --- tests ---
    def test_structure_children(self):
        """Test that the transformer block has the expected structure across multiple configurations."""
        for _ in tqdm(range(self.num_iterations), desc="Testing block structure"):
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

    def test_output_shape(self):
        """Test that output shape matches input shape across multiple configurations."""
        for _ in tqdm(range(self.num_iterations), desc="Testing output shape"):
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
    def test_eval_mode(self) -> None:
        """Test that calling eval() sets training=False for all parameters and submodules"""
        for _ in tqdm(range(self.num_iterations), desc="Testing eval mode"):
            block = self._generate_random_block()
            super()._test_eval_mode(block)

    def test_train_mode(self) -> None:
        """Test that calling train() sets training=True for all parameters and submodules"""
        for _ in tqdm(range(self.num_iterations), desc="Testing train mode"):
            block = self._generate_random_block()
            super()._test_train_mode(block)
    
    def test_consistent_output_without_dropout_bn(self) -> None:
        """
        Test that modules without dropout or batch normalization 
        produce consistent output for the same input
        """
        for _ in tqdm(range(self.num_iterations), desc="Testing consistent output"):
            block = self._generate_random_block()
            batch_size = np.random.randint(1, 10)
            seq_length = np.random.randint(5, 20)
            input_tensor = torch.randn(batch_size, seq_length, block.d_model)
            super()._test_consistent_output_without_dropout_bn(block, input_tensor)
    
    def test_consistent_output_in_eval_mode(self) -> None:
        """Test that all modules in eval mode produce consistent output for the same input"""
        for _ in tqdm(range(self.num_iterations), desc="Testing eval consistency"):
            block = self._generate_random_block()
            batch_size = np.random.randint(1, 10)
            seq_length = np.random.randint(5, 20)
            input_tensor = torch.randn(batch_size, seq_length, block.d_model)
            super()._test_consistent_output_in_eval_mode(block, input_tensor)
    
    def test_batch_size_one_in_train_mode(self) -> None:
        """
        Test that modules with batch normalization layers might raise errors 
        with batch size 1 in train mode
        """
        for _ in tqdm(range(self.num_iterations), desc="Testing batch size 1 (train)"):
            block = self._generate_random_block()
            seq_length = np.random.randint(5, 20)
            input_tensor = torch.randn(1, seq_length, block.d_model)
            super()._test_batch_size_one_in_train_mode(block, input_tensor)
    
    def test_batch_size_one_in_eval_mode(self) -> None:
        """Test that modules in eval mode should not raise errors for batch size 1"""
        for _ in tqdm(range(self.num_iterations), desc="Testing batch size 1 (eval)"):
            block = self._generate_random_block()
            seq_length = np.random.randint(5, 20)
            input_tensor = torch.randn(1, seq_length, block.d_model)
            super()._test_batch_size_one_in_eval_mode(block, input_tensor)

    def test_named_parameters_length(self) -> None:
        """Test that named_parameters() and parameters() have the same length"""
        for _ in tqdm(range(self.num_iterations), desc="Testing named parameters"):
            block = self._generate_random_block()
            super()._test_named_parameters_length(block)
    
    def test_to_device(self) -> None:
        """Test that module can move between devices properly"""
        for _ in tqdm(range(self.num_iterations), desc="Testing device movement"):
            block = self._generate_random_block()
            batch_size = np.random.randint(1, 10)
            seq_length = np.random.randint(5, 20)
            input_tensor = torch.randn(batch_size, seq_length, block.d_model)
            super()._test_to_device(block, input_tensor)


if __name__ == '__main__':
    import mypt.code_utils.pytorch_utils as pu
    pu.seed_everything(0)
    unittest.main()