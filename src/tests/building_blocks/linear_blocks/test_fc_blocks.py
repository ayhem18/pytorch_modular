import torch
import random
import unittest
import numpy as np

from torch import nn
from random import randint as ri

import mypt.code_utils.pytorch_utils as pu

from tests.custom_base_test import CustomModuleBaseTest
from mypt.building_blocks.linear_blocks.components import BasicLinearBlock
from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser
from mypt.building_blocks.linear_blocks.fc_blocks import GenericFCBlock, ExponentialFCBlock


class FCBlockTestBase(CustomModuleBaseTest):
    """
    Abstract base class for testing FC block implementations.
    This class provides common test methods but should not be run directly.
    """

    def setUp(self):
        pu.seed_everything(42)
        self.dim_analyser = DimensionsAnalyser()
        self.activation_types = [type(BasicLinearBlock._ACTIVATION_MAP[t]) for t in BasicLinearBlock._ACTIVATIONS]
        self.activation_names = BasicLinearBlock._ACTIVATIONS.copy()
    
    def _generate_random_fc_block(self, 
                                num_layers: int | None = None, 
                                activation: str | None = None, 
                                dropout: float | list[float] | None = None,
                                norm_layer: str | None = None):
        """Generate a random FC block with configurable parameters"""
        pass

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

    def _test_block_structure(self):
        """Test that the FC block has the correct structure"""
        for _ in range(100):
            # Create a block with random parameters
            num_layers = ri(2, 5)
            block = self._generate_random_fc_block(num_layers=num_layers)
            
            # Get the children of the block._block (the sequential container)
            children = list(block.children())
            
            # Check that there are num_layers blocks
            self.assertEqual(len(children), num_layers)
            
            # Check that the first n-1 blocks are non-final BasicLinearBlocks
            for i in range(num_layers - 1):
                self.assertIsInstance(children[i], BasicLinearBlock)
                self.assertFalse(children[i].is_final)
                
            # Check that the last block is a final BasicLinearBlock
            self.assertIsInstance(children[-1], BasicLinearBlock)
            self.assertTrue(children[-1].is_final)

    def _test_different_activations(self):
        """Test different activation functions"""
        for activation in self.activation_names:
            block = self._generate_random_fc_block(activation=activation)
            
            # Check all non-final blocks have the correct activation
            children = list(block.children())

            for child in children[:-1]:  # All but the last
                self.assertEqual(child.activation, activation)
                
                # Get the last layer of each non-final child, should be the activation
                child_layers = list(child.children())
                if child.add_activation:  # Only if activation is added
                    activation_layer = child_layers[-1]
                    
                    if activation == BasicLinearBlock._RELU:
                        self.assertIsInstance(activation_layer, nn.ReLU)
                    elif activation == BasicLinearBlock._LEAKY_RELU:
                        self.assertIsInstance(activation_layer, nn.LeakyReLU)
                    elif activation == BasicLinearBlock._TANH:
                        self.assertIsInstance(activation_layer, nn.Tanh)

    def _test_with_scalar_dropout(self):
        """Test blocks with scalar dropout value"""
        for _ in range(100):
            dropout_prob = random.uniform(0.1, 0.5)
            block = self._generate_random_fc_block(dropout=dropout_prob)
            
            # Check all non-final blocks have dropout with the correct value
            children = list(block.children())
            
            for child in children[:-1]:  # All but the last
                self.assertEqual(child.dropout, dropout_prob)
                
                # Check dropout layer exists with correct probability
                has_dropout = False
                for layer in child.children():
                    if isinstance(layer, nn.Dropout):
                        has_dropout = True
                        self.assertAlmostEqual(layer.p, dropout_prob)
                
                self.assertTrue(has_dropout, "Dropout layer not found")

    def _test_with_list_dropout(self):
        """Test blocks with list of dropout values"""
        for _ in range(100):
            num_layers = ri(2, 5)
            dropout_probs = [random.uniform(0.1, 0.5) for _ in range(num_layers - 1)]
            block = self._generate_random_fc_block(num_layers=num_layers, dropout=dropout_probs)
            
            # Check each non-final block has the correct dropout value
            children = list(block.children())
            
            for i, child in enumerate(children[:-1]):  # All but the last
                self.assertEqual(child.dropout, dropout_probs[i])
                
                # Check dropout layer exists with correct probability
                has_dropout = False
                for layer in child.children():
                    if isinstance(layer, nn.Dropout):
                        has_dropout = True
                        self.assertAlmostEqual(layer.p, dropout_probs[i])
                
                self.assertTrue(has_dropout, "Dropout layer not found")

    def _test_without_dropout(self):
        """Test blocks without dropout"""
        block = self._generate_random_fc_block(dropout=False)
        
        # Check all non-final blocks have no dropout
        children = list(block.children())
        
        for child in children[:-1]:  # All but the last
            self.assertIsNone(child.dropout)
            
            # Check no dropout layer exists
            has_dropout = any(isinstance(layer, nn.Dropout) for layer in child.children())
            self.assertFalse(has_dropout, "Unexpected dropout layer found")

    def _test_batch_size_one_handling(self):
        """Test handling of batch size 1 (important for BatchNorm)"""
        # Create a block with BatchNorm
        block_bn = self._generate_random_fc_block(norm_layer="batchnorm1d")
        
        # Create input with batch size 1
        input_tensor_bn = self._generate_random_input(block_bn, batch_size=1)
        
        # Eval mode should work with batch size 1
        block_bn.eval()
        output_bn = block_bn(input_tensor_bn)
        if input_tensor_bn.dim() == 2:
            self.assertEqual(output_bn.shape, (1, block_bn.output))
        
        # Create a block with LayerNorm
        block_ln = self._generate_random_fc_block(norm_layer="layernorm")
        
        # Create input with batch size 1
        input_tensor_ln = self._generate_random_input(block_ln, batch_size=1)
        
        # Train mode should work with LayerNorm and batch size 1
        block_ln.train()
        output_ln = block_ln(input_tensor_ln)
        if input_tensor_ln.dim() == 2:
            self.assertEqual(output_ln.shape, (1, block_ln.output))
        else:
            self.assertEqual(output_ln.shape, (1, input_tensor_ln.size(1), block_ln.output))

    def _test_train_and_eval_modes(self):
        """Test that train and eval modes work correctly"""
        for _ in range(100):
            # Create a block
            block = self._generate_random_fc_block()
            
            # Test eval mode
            block.eval()
            
            # Verify all children are in eval mode
            for child in block.modules():
                self.assertFalse(child.training, "Module should be in eval mode")
            
            # Create a small batch (size 1)
            input_tensor = self._generate_random_input(block, batch_size=1)
            
            # This should not raise an error in eval mode
            output = block(input_tensor)
            if input_tensor.dim() == 2:
                self.assertEqual(output.shape, (1, block.output))
            else:
                self.assertEqual(output.shape, (1, input_tensor.size(1), block.output))
            
            # Test train mode
            block.train()
            
            # Verify all children are in train mode
            for child in block.modules():
                self.assertTrue(child.training, "all children should be in train mode")

            # Non-final blocks with BatchNorm may have issues with batch size 1
            # Find BatchNorm1d modules
            bn_modules = [m for m in block.modules() if isinstance(m, nn.BatchNorm1d)]
            
            if bn_modules and input_tensor.dim() == 2:
                # in train mode, a batch of size 1 should raise a ValueError
                with self.assertRaises(ValueError):
                    _ = block(input_tensor)

                # In train mode with batch size > 1 should work
                input_tensor_larger = self._generate_random_input(block, batch_size=2)
                _ = block(input_tensor_larger)  # No error

    def _test_to_device(self):
        """Test that to() method works correctly"""
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        for _ in range(100):
            # Create a random block
            block = self._generate_random_fc_block()
            
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

    def _test_parameters_access(self):
        """Test that parameters can be accessed correctly"""
        for _ in range(100):
            # Create a random block
            block = self._generate_random_fc_block()
            
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
            
            # Check that parameter names don't have '_block.' prefix
            for name in named_params.keys():
                self.assertFalse(name.startswith('_block.'), 
                               f"Parameter name {name} should not start with '_block.'")

    def _test_forward_pass_shape(self):
        """Test that forward pass produces output with expected shape"""
        for _ in range(100):
            # Generate a random block
            block = self._generate_random_fc_block()
            
            # Create batch of random size
            batch_size = ri(1, 16)
            input_tensor = self._generate_random_input(block, batch_size)
            
            # Set to eval mode to avoid BatchNorm issues with batch size 1
            block.eval()
            
            # Get output
            output = block(input_tensor)
            
            # Check output shape
            if input_tensor.dim() == 2:
                self.assertEqual(output.shape, (batch_size, block.output))
            else:
                self.assertEqual(output.shape, (batch_size, input_tensor.size(1), block.output))

    def _test_3d_input_with_layernorm(self):
        """Test that LayerNorm blocks can handle 3D inputs"""
        for _ in range(100):
            # Create a block with LayerNorm
            block = self._generate_random_fc_block(norm_layer="layernorm")
            
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
            self.assertEqual(output.shape, (batch_size, seq_len, block.output))
            
            # Train mode should also work
            block.train()
            output = block(input_tensor)
            self.assertEqual(output.shape, (batch_size, seq_len, block.output))


class TestGenericFCBlock(FCBlockTestBase):
    """Test class for GenericFCBlock"""
    
    def _generate_random_units(self, in_features: int, output: int, num_layers: int) -> list[int]:
        """Generate a list of random unit sizes for the FC block"""
        units = [in_features]
        
        # Generate random hidden layer sizes
        for _ in range(num_layers - 1):
            units.append(ri(max(in_features, output) // 2, max(in_features, output) * 2))
        
        # Add output size
        units.append(output)
        
        return units

    def _generate_random_fc_block(self, 
                                num_layers: int | None = None, 
                                activation: str | None = None, 
                                dropout: float | list[float] | None = None,
                                norm_layer: str | None = None) -> GenericFCBlock:
        """Generate a random GenericFCBlock with configurable parameters"""
        in_features = ri(10, 100)
        output = ri(1, 10)
        
        if num_layers is None:
            num_layers = ri(2, 5)
            
        if activation is None:
            activation = random.choice(self.activation_names)
            
        if dropout is None:
            # Either no dropout, scalar dropout, or list of dropout values
            dropout_type = random.choice(['none', 'scalar', 'list'])
            
            if dropout_type == 'none':
                dropout = None
            elif dropout_type == 'scalar':
                dropout = random.uniform(0.1, 0.5)
            else:  # list
                dropout = [random.uniform(0.1, 0.5) for _ in range(num_layers - 1)]
        elif dropout is False:
            dropout = None

        if norm_layer is None:
            norm_layer = random.choice(["batchnorm1d", "layernorm"])
            
        # Generate random unit sizes
        units = self._generate_random_units(in_features, output, num_layers)
        
        return GenericFCBlock(
            output=output,
            in_features=in_features,
            num_layers=num_layers,
            units=units,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer
        )

    def test_units_validation(self):
        """Test that units validation works correctly"""
        # Test with correct number of units
        in_features = ri(10, 100)
        output = ri(1, 10)
        num_layers = ri(2, 5)
        
        # Correct number of units
        units = self._generate_random_units(in_features, output, num_layers)
        block = GenericFCBlock(
            output=output,
            in_features=in_features,
            num_layers=num_layers,
            units=units,
            activation='relu'
        )
        
        # Check that units were set correctly
        self.assertEqual(block.units, units)
        
        # Test with incorrect number of units (too few)
        units_too_few = units[:-1]  # Remove last element
        with self.assertRaises(ValueError):
            GenericFCBlock(
                output=output,
                in_features=in_features,
                num_layers=num_layers,
                units=units_too_few,
                activation='relu'
            )
        
        # Test with incorrect number of units (too many)
        units_too_many = units + [ri(1, 10)]  # Add an extra element
        with self.assertRaises(ValueError):
            GenericFCBlock(
                output=output,
                in_features=in_features,
                num_layers=num_layers,
                units=units_too_many,
                activation='relu'
            )

    def test_consecutive_layers_connect_properly(self):
        """Test that consecutive layers connect properly"""
        for _ in range(100):
            # Create a block with random parameters
            block = self._generate_random_fc_block()
            
            # Get the children of the block
            children = list(block.children())
            
            # Check that consecutive layers connect properly
            for i in range(len(children) - 1):
                self.assertEqual(children[i].out_features, children[i+1].in_features,
                                f"Layer {i} output ({children[i].out_features}) doesn't match layer {i+1} input ({children[i+1].in_features})")

    def test_forward_pass_shape(self):
        super()._test_forward_pass_shape()  

    def test_block_structure(self):
        super()._test_block_structure()

    def test_different_activations(self):
        super()._test_different_activations()

    def test_with_scalar_dropout(self):
        super()._test_with_scalar_dropout()

    def test_with_list_dropout(self):
        super()._test_with_list_dropout()

    def test_without_dropout(self):
        super()._test_without_dropout()

    def test_batch_size_one_handling(self):
        super()._test_batch_size_one_handling()

    def test_train_and_eval_modes(self):
        super()._test_train_and_eval_modes()


    def test_to_device(self):
        """Test that to() method works correctly"""
        self._test_to_device()

    def test_parameters_access(self):
        """Test that parameters can be accessed correctly"""
        self._test_parameters_access()
        
    def test_3d_input_with_layernorm(self):
        """Test that LayerNorm blocks can handle 3D inputs"""
        self._test_3d_input_with_layernorm()

    def test_eval_mode(self):
        for _ in range(100):
            block = self._generate_random_fc_block()
            super()._test_eval_mode(block)
    
    def test_train_mode(self):
        for _ in range(100):
            block = self._generate_random_fc_block()
            super()._test_train_mode(block)
    
    def test_consistent_output_in_eval_mode(self):
        for _ in range(100):
            block = self._generate_random_fc_block()
            input_tensor = self._generate_random_input(block)
            super()._test_consistent_output_in_eval_mode(block, input_tensor)
    
    def test_batch_size_one_in_train_mode(self):
        for _ in range(100):
            # Use LayerNorm to avoid BatchNorm1d issues with batch size 1
            block = self._generate_random_fc_block(norm_layer="layernorm")
            input_tensor = self._generate_random_input(block, batch_size=1)
            super()._test_batch_size_one_in_train_mode(block, input_tensor)
    
    def test_batch_size_one_in_eval_mode(self):
        for _ in range(100):
            block = self._generate_random_fc_block()
            input_tensor = self._generate_random_input(block, batch_size=1)
            super()._test_batch_size_one_in_eval_mode(block, input_tensor)
    
    def test_named_parameters_length(self):
        for _ in range(100):
            block = self._generate_random_fc_block()
            super()._test_named_parameters_length(block)


class TestExponentialFCBlock(FCBlockTestBase):
    def setUp(self):
        super().setUp()
        # Add any ExponentialFCBlock-specific setup here
        pass

    def _generate_random_fc_block(self, 
                                num_layers: int | None = None, 
                                activation: str | None = None, 
                                dropout: float | list[float] | None = None,
                                norm_layer: str | None = None) -> GenericFCBlock:
        """Generate a random ExponentialFCBlock with configurable parameters"""
        in_features = ri(10, 100)
        output = ri(1, 10)
        
        if num_layers is None:
            num_layers = ri(2, 5)
            
        if activation is None:
            activation = random.choice(self.activation_names)
        
        
        in_features = ri(10, 1000)
        output = ri(2, 20)

        while int(np.log2(in_features)) == int(np.log2(output)):
            in_features = ri(10, 1000)
            output = ri(2, 20)

            
        in_features = ri(10, 1000)
        output = ri(2, 20)

        while int(np.log2(in_features)) == int(np.log2(output)):
            in_features = ri(10, 1000)
            output = ri(2, 20)

        if dropout is None:
            # Either no dropout, scalar dropout, or list of dropout values
            dropout_type = random.choice(['none', 'scalar', 'list'])
            
            if dropout_type == 'none':
                dropout = None
            elif dropout_type == 'scalar':
                dropout = random.uniform(0.1, 0.5)
            else:  # list
                dropout = [random.uniform(0.1, 0.5) for _ in range(num_layers - 1)]
        elif dropout is False:
            dropout = None
            
        if norm_layer is None:
            norm_layer = random.choice(["batchnorm1d", "layernorm"])
            
        return ExponentialFCBlock(
            output=output,
            in_features=in_features,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer
        )

    # Tests for ExponentialFCBlock
    def test_forward_pass_shape(self):
        """Test that forward pass produces output with expected shape"""
        self._test_forward_pass_shape()

    def test_block_structure(self):
        """Test that the FC block has the correct structure"""
        self._test_block_structure()

    def test_different_activations(self):
        """Test different activation functions"""
        self._test_different_activations()

    def test_with_scalar_dropout(self):
        """Test blocks with scalar dropout value"""
        self._test_with_scalar_dropout()

    def test_with_list_dropout(self):
        """Test blocks with list of dropout values"""
        self._test_with_list_dropout()

    def test_without_dropout(self):
        """Test blocks without dropout"""
        self._test_without_dropout()
        
    def test_3d_input_with_layernorm(self):
        """Test that LayerNorm blocks can handle 3D inputs"""
        self._test_3d_input_with_layernorm()

    def test_batch_size_one_handling(self):
        """Test handling of batch size 1 (important for BatchNorm)"""
        self._test_batch_size_one_handling()

    def test_train_and_eval_modes(self):
        """Test that train and eval modes work correctly"""
        self._test_train_and_eval_modes()

    def test_to_device(self):
        """Test that to() method works correctly"""
        self._test_to_device()

    def test_parameters_access(self):
        """Test that parameters can be accessed correctly"""
        self._test_parameters_access()

    def test_eval_mode(self):
        for _ in range(100):
            block = self._generate_random_fc_block()
            super()._test_eval_mode(block)
    
    def test_train_mode(self):
        for _ in range(100):
            block = self._generate_random_fc_block()
            super()._test_train_mode(block)
    
    def test_consistent_output_in_eval_mode(self):
        for _ in range(100):
            block = self._generate_random_fc_block()
            input_tensor = self._generate_random_input(block)
            super()._test_consistent_output_in_eval_mode(block, input_tensor)
    
    def test_batch_size_one_in_train_mode(self):
        for _ in range(100):
            # Use LayerNorm to avoid BatchNorm1d issues with batch size 1
            block = self._generate_random_fc_block(norm_layer="layernorm")
            input_tensor = self._generate_random_input(block, batch_size=1)
            super()._test_batch_size_one_in_train_mode(block, input_tensor)
    
    def test_batch_size_one_in_eval_mode(self):
        for _ in range(100):
            block = self._generate_random_fc_block()
            input_tensor = self._generate_random_input(block, batch_size=1)
            super()._test_batch_size_one_in_eval_mode(block, input_tensor)
    
    def test_named_parameters_length(self):
        for _ in range(100):
            block = self._generate_random_fc_block()
            super()._test_named_parameters_length(block)


if __name__ == '__main__':
    pu.seed_everything(42)
    unittest.main()