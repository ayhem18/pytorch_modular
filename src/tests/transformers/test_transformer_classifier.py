import torch
import unittest

import numpy as np

from typing import Optional

from tests.custom_base_test import CustomModuleBaseTest
from mypt.nets.transformers.transformer_classifier import TransformerClassifier, _POOLING_REGISTRY


class TestTransformerClassifier(CustomModuleBaseTest):
    def setUp(self):
        # Default parameters
        self.d_model = 32
        self.num_heads = 4
        self.key_dim = 8
        self.value_dim = 8
        self.num_blocks = 2
        self.num_classes = 3
        self.num_cls_layers = 2
        
        # Number of test iterations
        self.num_iterations = 50

    # --- helper methods ---
    def _generate_random_model(self, pooling: Optional[str] = None) -> TransformerClassifier:
        """Generate a random transformer classifier with valid parameters."""
        d_model = np.random.choice([16, 32, 64, 128])
        num_heads = np.random.choice([1, 2, 4, 8])
        key_dim = (d_model // num_heads) * np.random.choice([1, 2])
        value_dim = (d_model // num_heads) * np.random.choice([1, 2])
        num_blocks = np.random.randint(1, 4)
        num_classes = np.random.randint(2, 10)
        num_cls_layers = np.random.randint(1, 3)
        dropout = np.random.uniform(0.0, 0.5)
        
        if pooling is None:
            pooling = np.random.choice(list(_POOLING_REGISTRY.keys()))
        
        return TransformerClassifier(
            d_model=d_model,
            num_transformer_blocks=num_blocks,
            num_classification_layers=num_cls_layers,
            num_heads=num_heads,
            value_dim=value_dim,
            key_dim=key_dim,
            num_classes=num_classes,
            pooling=pooling,
            dropout=dropout,
        )

    def _get_valid_input(self, model: TransformerClassifier, batch: int = 4, seq_len: int = 10):
        return torch.randn(batch, seq_len, model.encoder[0].d_model)

    def _get_pad_mask(self, batch: int, seq_len: int):
        mask = (torch.rand(batch, seq_len) > 0.3).int()
        for b in range(batch):
            if mask[b].sum() == 0:
                mask[b, 0] = 1
        return mask

    # --- tests ---
    def test_all_pooling_variants(self):
        """Test each pooling variant with multiple random configurations."""
        for pooling in _POOLING_REGISTRY.keys():
            for _ in range(self.num_iterations):
                model = self._generate_random_model(pooling)
                
                # Verify pooling property
                self.assertEqual(model.get_pooling_type(), pooling)
                
                # Generate random input
                batch_size = np.random.randint(2, 10)
                seq_length = np.random.randint(5, 20)
                inp = self._get_valid_input(model, batch_size, seq_length)
                pad_mask = self._get_pad_mask(batch_size, seq_length)
                
                # Shape check
                out = model(inp, pad_mask)
                self.assertEqual(out.shape, (batch_size, model.head.output))

    def test_pooling_registry_completeness(self):
        """Test that all pooling strategies in the registry work correctly."""
        for pooling_name, pooling_class in _POOLING_REGISTRY.items():
            # Create an instance of the pooling class
            pooling_instance = pooling_class()
            self.assertTrue(hasattr(pooling_instance, 'forward'), 
                           f"Pooling class {pooling_name} should have a forward method")


    def test_eval_mode(self) -> None:
        """Test that calling eval() sets training=False for all parameters and submodules"""
        for _ in range(self.num_iterations):
            model = self._generate_random_model()
            super()._test_eval_mode(model)

    def test_train_mode(self) -> None:
        """Test that calling train() sets training=True for all parameters and submodules"""
        for _ in range(self.num_iterations):
            model = self._generate_random_model()
            super()._test_train_mode(model)
    
    def test_consistent_output_in_eval_mode(self) -> None:
        """Test that all modules in eval mode produce consistent output for the same input"""
        for _ in range(self.num_iterations):
            model = self._generate_random_model()
            batch_size = np.random.randint(1, 10)
            seq_length = np.random.randint(5, 20)
            input_tensor = self._get_valid_input(model, batch_size, seq_length)
            pad_mask = self._get_pad_mask(batch_size, seq_length)
            super()._test_consistent_output_in_eval_mode(model, input_tensor, pad_mask)
    
    def test_batch_size_one_in_train_mode(self) -> None:
        """
        Test that modules with batch normalization layers might raise errors 
        with batch size 1 in train mode
        """
        for _ in range(self.num_iterations):
            model = self._generate_random_model()
            seq_length = np.random.randint(5, 20)
            input_tensor = self._get_valid_input(model, 1, seq_length)
            pad_mask = self._get_pad_mask(1, seq_length)
            super()._test_batch_size_one_in_train_mode(model, input_tensor, pad_mask)
    
    def test_batch_size_one_in_eval_mode(self) -> None:
        """Test that modules in eval mode should not raise errors for batch size 1"""
        for _ in range(self.num_iterations):
            model = self._generate_random_model()
            seq_length = np.random.randint(5, 20)
            input_tensor = self._get_valid_input(model, 1, seq_length)
            pad_mask = self._get_pad_mask(1, seq_length)
            super()._test_batch_size_one_in_eval_mode(model, input_tensor, pad_mask)

    def test_named_parameters_length(self) -> None:
        """Test that named_parameters() and parameters() have the same length"""
        for _ in range(self.num_iterations):
            model = self._generate_random_model()
            super()._test_named_parameters_length(model)
    
    def test_to_device(self) -> None:
        """Test that module can move between devices properly"""
        for _ in range(self.num_iterations):
            model = self._generate_random_model()
            batch_size = np.random.randint(1, 10)
            seq_length = np.random.randint(5, 20)
            input_tensor = self._get_valid_input(model, batch_size, seq_length)
            pad_mask = self._get_pad_mask(batch_size, seq_length)
            super()._test_to_device(model, input_tensor, pad_mask)


if __name__ == "__main__":
    import mypt.code_utils.pytorch_utils as pu
    pu.seed_everything(0)
    unittest.main()