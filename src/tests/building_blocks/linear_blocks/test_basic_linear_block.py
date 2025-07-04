import torch
import random
import unittest

from torch import nn
from random import randint as ri

import mypt.code_utils.pytorch_utils as pu

from tests.custom_base_test import CustomModuleBaseTest
from mypt.building_blocks.linear_blocks.components import BasicLinearBlock
from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser


class TestBasicLinearBlock(CustomModuleBaseTest):
    def setUp(self):
        self.dim_analyser = DimensionsAnalyser()
        self.activation_types = [type(BasicLinearBlock._ACTIVATION_MAP[t]) for t in BasicLinearBlock._ACTIVATIONS]
        self.activation_names = BasicLinearBlock._ACTIVATIONS.copy()
        self.num_iterations = 1000

    def _generate_random_linear_block(self, is_final=None, add_activation=None, activation=None, dropout=None, norm_layer=None):
        """
        Generate a random LinearBlock with configurable parameters
        """
        in_features = ri(10, 100)
        out_features = ri(10, 100)
        
        if activation is None:
            activation = random.choice(self.activation_names)
        
        if is_final is None:
            is_final = random.choice([True, False])
            
        if add_activation is None:
            add_activation = random.choice([True, False])
            
        if dropout is None:
            dropout = random.choice([None, 0.1, 0.5])

        if dropout is False:
            dropout = None

        if norm_layer is None:
            norm_layer = random.choice(["batchnorm1d", "layernorm"])

        return BasicLinearBlock(
            in_features=in_features,
            out_features=out_features,
            activation=activation,
            dropout=dropout,
            is_final=is_final,
            add_activation=add_activation,
            norm_layer=norm_layer
        )

    def _generate_random_input(self, block, batch_size=None, seq_len=None):
        """
        Generate a random input tensor for the given block.
        For blocks with BatchNorm1d, input will be 2D (batch_size, in_features)
        For blocks with LayerNorm, input can be either 2D or 3D (batch_size, seq_len, in_features)
        """
        if batch_size is None:
            batch_size = ri(1, 16)
        
        # Find if the block has BatchNorm1d
        has_batchnorm = any(isinstance(m, nn.BatchNorm1d) for m in block.modules())
        
        if has_batchnorm:
            # For BatchNorm1d, we need 2D input
            return torch.randn(batch_size, block.in_features)
        else:
            # For LayerNorm, we can use either 2D or 3D input
            use_3d = random.choice([True, False]) if seq_len is None else True
            
            if use_3d:
                if seq_len is None:
                    seq_len = ri(5, 20)
                return torch.randn(batch_size, seq_len, block.in_features)
            else:
                return torch.randn(batch_size, block.in_features)

    def test_non_final_block_structure(self):
        """Test that non-final blocks have the correct structure"""
        for _ in range(self.num_iterations):
            # Create a non-final block
            block = self._generate_random_linear_block(is_final=False, add_activation=True)
            
            # Get all children
            children = list(block.children())
            
            # For non-final blocks with activation, we expect:
            # 1. BatchNorm1d or LayerNorm
            # 2. Optional Dropout
            # 3. Linear
            # 4. Activation
            
            # First layer should be BatchNorm1d or LayerNorm
            self.assertIsInstance(children[0], (nn.BatchNorm1d, nn.LayerNorm))
            
            # Last layer should be an activation
            self.assertIsInstance(children[-1], nn.Module)
            self.assertTrue(type(children[-1]) in self.activation_types)
            
            # Second-to-last layer should be Linear
            self.assertIsInstance(children[-2], nn.Linear)
            
            # If there's a dropout, it should be after BatchNorm and before Linear
            if len(children) > 3:  # Has dropout
                self.assertIsInstance(children[1], nn.Dropout)

    def test_non_final_block_no_activation(self):
        """Test non-final blocks without activation"""
        for _ in range(self.num_iterations):
            # Create a non-final block without activation
            block = self._generate_random_linear_block(is_final=False, add_activation=False)
            
            # Get all children
            children = list(block.children())   
            
            # Last layer should be Linear
            self.assertIsInstance(children[-1], nn.Linear)
            
            # First layer should be BatchNorm1d or LayerNorm
            self.assertIsInstance(children[0], (nn.BatchNorm1d, nn.LayerNorm))

    def test_final_block_structure(self):
        """Test that final blocks have the correct structure"""
        for _ in range(self.num_iterations):
            # Create a final block
            block = self._generate_random_linear_block(is_final=True)
            
            # Get all children
            children = list(block.children())
            
            # For final blocks, we expect only a Linear layer
            self.assertEqual(len(children), 1)
            self.assertIsInstance(children[0], nn.Linear)

    def test_different_activations(self):
        """Test different activation functions"""
        for activation in self.activation_names:
            block = self._generate_random_linear_block(is_final=False, activation=activation, add_activation=True)
            
            # Get all children
            children = list(block.children())
            
            # Last layer should be activation
            last_layer = children[-1]
            
            if activation == BasicLinearBlock._RELU:
                self.assertIsInstance(last_layer, nn.ReLU)
            elif activation == BasicLinearBlock._LEAKY_RELU:
                self.assertIsInstance(last_layer, nn.LeakyReLU)
            elif activation == BasicLinearBlock._TANH:
                self.assertIsInstance(last_layer, nn.Tanh)

    def test_with_dropout(self):
        """Test blocks with dropout"""
        for _ in range(self.num_iterations):
            dropout_prob = random.uniform(0.1, 0.5)
            block = self._generate_random_linear_block(is_final=False, dropout=dropout_prob, add_activation=True)
            
            # Get all children
            children = list(block.children())
            
            # For non-final blocks with dropout:
            # 1. BatchNorm1d or LayerNorm
            # 2. Dropout
            # 3. Linear
            # 4. Activation
            
            self.assertGreaterEqual(len(children), 4)
            
            # First layer should be BatchNorm1d or LayerNorm
            self.assertIsInstance(children[0], (nn.BatchNorm1d, nn.LayerNorm))
            self.assertIsInstance(children[1], nn.Dropout)
            self.assertIsInstance(children[2], nn.Linear)
            self.assertTrue(type(children[3]) in self.activation_types)

    def test_without_dropout(self):
        """Test blocks without dropout"""
        for _ in range(self.num_iterations):
            block = self._generate_random_linear_block(is_final=False, dropout=False)
            
            # Get all children
            children = list(block.children())
            
            # For non-final blocks without dropout:
            # 1. BatchNorm1d or LayerNorm
            # 2. Linear
            # 3. Activation
            
            # Check no dropout layer exists
            has_dropout = any(isinstance(child, nn.Dropout) for child in children)
            self.assertFalse(has_dropout)

    def test_named_children(self):
        """Test that named_children method returns the correct names"""
        # Test different configurations
        configs = [
            {"is_final": True, "dropout": None, "add_activation": True},
            {"is_final": False, "dropout": None, "add_activation": True},
            {"is_final": False, "dropout": 0.5, "add_activation": True},
            {"is_final": False, "dropout": 0.5, "add_activation": False}
        ]
        
        for config in configs:
            block = self._generate_random_linear_block(**config)
            named_children = dict(block.block.named_children())
            
            # Check that each child has a name
            self.assertEqual(len(named_children), len(list(block.named_children())))
            
            if config["is_final"]:
                # Final blocks have only Linear
                self.assertEqual(len(named_children), 1)
                self.assertTrue(any(isinstance(module, nn.Linear) for module in named_children.values()))
            else:
                # Non-final blocks have BatchNorm1d or LayerNorm and Linear
                self.assertTrue(any(isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)) for module in named_children.values()))
                self.assertTrue(any(isinstance(module, nn.Linear) for module in named_children.values()))
                
                # Check for Dropout if specified
                if config["dropout"] is not None:
                    self.assertTrue(any(isinstance(module, nn.Dropout) for module in named_children.values()))
                
                # Check for activation if specified
                if config["add_activation"]:
                    self.assertTrue(any(type(module) in [nn.ReLU, nn.LeakyReLU, nn.Tanh, nn.GELU] for module in named_children.values()))

    def test_forward_pass_shape(self):
        """Test that forward pass produces output with expected shape"""
        for _ in range(self.num_iterations):
            # Generate a random block
            block = self._generate_random_linear_block()
            
            # Get the first Linear layer to determine input size
            linear_layer = next(module for module in block.modules() if isinstance(module, nn.Linear))
            out_features = linear_layer.out_features
            
            # Create batch of random size
            block.eval() 
            batch_size = ri(1, 16)
            
            # Generate appropriate input based on the normalization layer
            input_tensor = self._generate_random_input(block, batch_size)
            
            # Get actual output shape
            output = block(input_tensor)
            
            # Check output shape based on input shape
            if input_tensor.dim() == 2:
                # For 2D input (batch_size, in_features)
                self.assertEqual(output.shape, (batch_size, out_features))
            else:
                # For 3D input (batch_size, seq_len, in_features)
                self.assertEqual(output.shape, (batch_size, input_tensor.size(1), out_features))

    def test_train_and_eval_modes(self):
        """Test that train and eval modes work correctly"""
        for _ in range(self.num_iterations):
            # Create a block with batch normalization
            block = self._generate_random_linear_block(is_final=False)
            
            # Test eval mode
            block.eval()
            
            # Verify all children are in eval mode
            for child in block.block.children():
                self.assertFalse(child.training, "Child module should be in eval mode")
            
            # Create a small batch
            input_tensor = self._generate_random_input(block, batch_size=1)
            
            # This should not raise an error in eval mode
            try:
                _ = block(input_tensor)
            except Exception as e:
                self.fail(f"Block in eval mode raised an error with batch size 1: {e}")
            
            # Test train mode
            block.train()
            
            # Verify all children are in train mode
            for child in block.block.children():
                self.assertTrue(child.training, "Child module should be in train mode")

            # Find BatchNorm1d modules
            bn_modules = [m for m in block.modules() if isinstance(m, nn.BatchNorm1d)]
            
            # If there are batch norm modules (non-final blocks)
            if bn_modules:
                # In train mode, a batch of size 1 should raise a ValueError for BatchNorm1d
                if input_tensor.dim() == 2 and input_tensor.size(0) == 1:
                    with self.assertRaises(ValueError):
                        _ = block(input_tensor)
                
                # Store running stats in train mode
                input_tensor_larger = self._generate_random_input(block, batch_size=2)
                _ = block(input_tensor_larger)  # Run with batch size > 1
                train_running_means = [bn.running_mean.clone() for bn in bn_modules]
                
                # Set to eval mode
                block.eval()
                
                # Run with different data
                _ = block(input_tensor_larger)
                
                # In eval mode, running stats should not change
                for i, bn in enumerate(bn_modules):
                    self.assertTrue(torch.allclose(train_running_means[i], bn.running_mean), 
                                   "BatchNorm running mean should not change in eval mode")

    def test_to_device(self):
        """Test that to() method works correctly"""
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        for _ in range(self.num_iterations):
            # Create a random block
            block = self._generate_random_linear_block()
            
            # Move to CUDA
            block = block.to('cuda')
            
            # Check that all parameters are on CUDA
            for param in block.parameters():
                self.assertTrue(param.is_cuda, "Parameter should be on CUDA")
            
            # Move back to CPU
            block = block.to('cpu')
            
            # Check that all parameters are on CPU
            for param in block.parameters():
                self.assertFalse(param.is_cuda, "Parameter should be on CPU")

    def test_parameters_access(self):
        """Test that parameters can be accessed correctly"""
        for _ in range(self.num_iterations):
            # Create a random block
            block = self._generate_random_linear_block()
            
            # Get parameters
            params = list(block.parameters())
            
            # Check that parameters exist
            self.assertGreater(len(params), 0, "Block should have parameters")
            
            # Check that parameters are torch Parameters
            for param in params:
                self.assertIsInstance(param, torch.nn.Parameter)
            
            # Check named parameters
            named_params = dict(block.named_parameters())
            self.assertEqual(len(named_params), len(params))

            for name, param in named_params.items():
                self.assertFalse(name.startswith('_block.'), 
                               f"Parameter name {name} does not start with '_block.'")
                self.assertIsInstance(param, torch.nn.Parameter)

    def test_batch_size_one_handling(self):
        """Test handling of batch size 1 (important for BatchNorm)"""
        for is_final in [True, False]:
            # Create a block with or without batch norm
            block = self._generate_random_linear_block(is_final=is_final)
            
            # Create input with batch size 1
            input_tensor = self._generate_random_input(block, batch_size=1)
            
            # For final blocks (no batch norm), should work in both modes
            if is_final:
                block.train()
                _ = block(input_tensor)  # Should not raise error
                
                block.eval()
                _ = block(input_tensor)  # Should not raise error
            else:
                # check if the normalization layer is batch norm
                norm_layers = [m for m in block.modules() if isinstance(m, (nn.BatchNorm1d, nn.LayerNorm))]
                has_batchnorm = any(isinstance(m, nn.BatchNorm1d) for m in norm_layers)

                # For non-final blocks (with batch norm)
                # Eval mode should work with batch size 1
                block.eval()
                _ = block(input_tensor)  # Should not raise error
                
                block.train()

                # Only BatchNorm1d should have issues with batch size 1 in train mode
                if has_batchnorm and input_tensor.dim() == 2:
                    with self.assertRaises(ValueError):
                        block.forward(input_tensor)  # having an error ensures that batch normalization indeed moved to the train mode.
                else:
                    # LayerNorm should work fine with batch size 1
                    _ = block(input_tensor)

    def test_3d_input_with_layernorm(self):
        """Test that LayerNorm blocks can handle 3D inputs"""
        for _ in range(self.num_iterations):
            # Create a block with LayerNorm
            block = self._generate_random_linear_block(is_final=False, norm_layer="layernorm")
            
            # Verify it has LayerNorm
            has_layernorm = any(isinstance(m, nn.LayerNorm) for m in block.modules())
            self.assertTrue(has_layernorm, "Block should have LayerNorm")
            
            # Create 3D input
            batch_size = ri(1, 10)
            seq_len = ri(5, 20)
            input_tensor = torch.randn(batch_size, seq_len, block.in_features)
            
            # Forward pass should work
            block.eval()
            output = block(input_tensor)
            
            # Check output shape
            self.assertEqual(output.shape, (batch_size, seq_len, block.out_features))
            
            # Train mode should also work
            block.train()
            output = block(input_tensor)
            self.assertEqual(output.shape, (batch_size, seq_len, block.out_features))

    def test_eval_mode(self):
        for _ in range(self.num_iterations):
            block = self._generate_random_linear_block()
            self._test_eval_mode(block)
    
    def test_train_mode(self):
        for _ in range(self.num_iterations):
            block = self._generate_random_linear_block()
            self._test_train_mode(block)
    
    # Custom module base tests
    def test_consistent_output_without_dropout_bn(self):
        # This shouldn't pass with BatchNorm, so we'll use final blocks
        for _ in range(self.num_iterations):
            block = self._generate_random_linear_block(is_final=True, dropout=None)
            input_tensor = self._generate_random_input(block)
            super()._test_consistent_output_without_dropout_bn(block, input_tensor)


    def test_consistent_output_in_eval_mode(self):
        for _ in range(self.num_iterations):
            block = self._generate_random_linear_block()
            input_tensor = self._generate_random_input(block)
            super()._test_consistent_output_in_eval_mode(block, input_tensor)
    
    def test_batch_size_one_in_train_mode(self):
        for _ in range(self.num_iterations):
            # Use LayerNorm to avoid BatchNorm1d issues with batch size 1
            block = self._generate_random_linear_block(is_final=False, norm_layer="layernorm")
            input_tensor = self._generate_random_input(block, batch_size=1)
            super()._test_batch_size_one_in_train_mode(block, input_tensor)


    def test_batch_size_one_in_eval_mode(self):
        for _ in range(self.num_iterations):
            block = self._generate_random_linear_block()
            input_tensor = self._generate_random_input(block, batch_size=1)
            super()._test_batch_size_one_in_eval_mode(block, input_tensor)


    def test_named_parameters_length(self):
        for _ in range(self.num_iterations):
            block = self._generate_random_linear_block()
            super()._test_named_parameters_length(block)


if __name__ == '__main__':
    pu.seed_everything(42)
    unittest.main()