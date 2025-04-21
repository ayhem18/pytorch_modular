import unittest
import torch
import random
import numpy as np
from torch import nn
from random import randint as ri

import mypt.code_utils.pytorch_utils as pu
from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser
from mypt.building_blocks.linear_blocks.components import BasicLinearBlock, ResidualFullyConnectedBlock
from mypt.building_blocks.linear_blocks.residual_fc_blocks import (
    GenericResidualFCBlock, 
    ExponentialResidualFCBlock
)


class ResidualFCBlockTestBase(unittest.TestCase):
    """
    Base class for testing residual FC block implementations.
    """
    def setUp(self):
        pu.seed_everything(42)
        self.dim_analyser = DimensionsAnalyser()
        self.activation_types = [type(BasicLinearBlock._ACTIVATION_MAP[t]) for t in BasicLinearBlock._ACTIVATIONS]
        self.activation_names = BasicLinearBlock._ACTIVATIONS.copy()
    
    def _generate_random_residual_fc_block(self, 
                                           num_layers=None, 
                                           activation=None,
                                           dropout=None, 
                                           force_residual=None,
                                           matching_dimensions=None) -> ResidualFullyConnectedBlock:
        """Generate a random residual FC block with configurable parameters"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def _test_block_structure(self):
        """Test that the FC block has the correct structure"""
        for _ in range(20):
            # Create a block with random parameters
            num_layers = ri(2, 5)
            block = self._generate_random_residual_fc_block(num_layers=num_layers)
            

            # Should have BasicLinearBlocks in the main stream
            main_stream_children = list(block._block.children())
            self.assertEqual(len(main_stream_children), num_layers)
            
            # Check that the first n-1 blocks are non-final BasicLinearBlocks
            for i in range(num_layers - 1):
                self.assertIsInstance(main_stream_children[i], BasicLinearBlock)
                self.assertFalse(main_stream_children[i].is_final)
                
            # Check that the last block is a final BasicLinearBlock
            self.assertIsInstance(main_stream_children[-1], BasicLinearBlock)
            self.assertTrue(main_stream_children[-1].is_final)
    
    def _test_adaptive_layer_creation(self):
        """Test that adaptive layer is created when necessary"""
        # Test with matching dimensions and force_residual=False
        block = self._generate_random_residual_fc_block(matching_dimensions=True, force_residual=False)
        self.assertIsNone(block._adaptive_layer, 
                         "Adaptive layer should not be created when dimensions match and force_residual=False")
        
        # Test with non-matching dimensions
        block = self._generate_random_residual_fc_block(matching_dimensions=False, force_residual=False)
        self.assertTrue(hasattr(block, '_adaptive_layer'), 
                        "Adaptive layer should be created when dimensions don't match")
        self.assertIsInstance(block._adaptive_layer, nn.Linear)
        self.assertEqual(block._adaptive_layer.in_features, block.in_features)
        self.assertEqual(block._adaptive_layer.out_features, block.output)
        
        # Test with force_residual=True
        block = self._generate_random_residual_fc_block(matching_dimensions=True, force_residual=True)
        self.assertTrue(hasattr(block, '_adaptive_layer'), 
                        "Adaptive layer should be created when force_residual=True")
    
    def _test_residual_forward_pass(self):
        """Test that the residual forward pass works correctly"""
        # Test with matching dimensions
        for _ in range(100):
            block = self._generate_random_residual_fc_block(matching_dimensions=True, force_residual=False)
            block.eval() # make sure to set the block to eval mode for consistent output
            x = torch.randn(2, block.in_features)
            
            # Forward through main block manually to compare
            main_output = block._block(x)
            residual_output = x  # Identity for matching dimensions
            
            # The output should be main_output + residual_output
            expected_output = main_output + residual_output
            actual_output = block(x)
            
            self.assertTrue(torch.allclose(actual_output, expected_output, rtol=1e-5, atol=1e-5),
                            "Residual forward pass should add main output and input when dimensions match")
            
            # Test with non-matching dimensions or force_residual=True
            block = self._generate_random_residual_fc_block(matching_dimensions=False, force_residual=False)
            block.eval() # make sure to set the block to eval mode for consistent output
            x = torch.randn(2, block.in_features)
            
            # Forward through components manually to compare
            main_output = block._block(x)
            residual_output = block._adaptive_layer(x)
            
            # The output should be main_output + residual_output
            expected_output = main_output + residual_output
            actual_output = block(x)
            
            self.assertTrue(torch.allclose(actual_output, expected_output, rtol=1e-5, atol=1e-5),
                            "Residual forward pass should add main output and adaptive layer output")
    
    def _test_different_activations(self):
        """Test different activation functions"""
        for activation in self.activation_names:
            block = self._generate_random_residual_fc_block(activation=activation)
            
            # Check all non-final blocks have the correct activation
            main_stream_children = list(block._block.children())
            
            for child in main_stream_children[:-1]:  # All but the last
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
        for _ in range(20):
            dropout_prob = random.uniform(0.1, 0.5)
            block = self._generate_random_residual_fc_block(dropout=dropout_prob)
            
            # Check all non-final blocks have dropout with the correct value
            main_stream_children = list(block._block.children())
            
            for child in main_stream_children[:-1]:  # All but the last
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
        for _ in range(20):
            num_layers = ri(2, 5)
            dropout_probs = [random.uniform(0.1, 0.5) for _ in range(num_layers - 1)]
            block = self._generate_random_residual_fc_block(num_layers=num_layers, dropout=dropout_probs)
            
            # Check each non-final block has the correct dropout value
            main_stream_children = list(block._block.children())
            
            for i, child in enumerate(main_stream_children[:-1]):  # All but the last
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
        block = self._generate_random_residual_fc_block(dropout=False)
        
        # Check all non-final blocks have no dropout
        main_stream_children = list(block._block.children())
        
        for child in main_stream_children[:-1]:  # All but the last
            self.assertIsNone(child.dropout)
            
            # Check no dropout layer exists
            has_dropout = any(isinstance(layer, nn.Dropout) for layer in child.children())
            self.assertFalse(has_dropout, "Unexpected dropout layer found")
    
    def _test_forward_pass_shape(self):
        """Test that the forward pass shape is correct"""
        for _ in range(20):
            # Create a random block
            block = self._generate_random_residual_fc_block()
            
            # Create a random batch size
            batch_size = ri(2, 16)
            
            # Create input tensor
            input_tensor = torch.randn(batch_size, block.in_features)
            
            # Get actual output shape
            output = block(input_tensor)
            actual_shape = tuple(output.shape)
            
            # Expected shape is (batch_size, out_features)
            expected_shape = (batch_size, block.output)
            
            # Assert shapes match
            self.assertEqual(actual_shape, expected_shape, 
                            f"Output shape mismatch: got {actual_shape}, expected {expected_shape}")
    
    def _test_train_and_eval_modes(self):
        """Test that train and eval modes work correctly"""
        for _ in range(20):
            # Create a block
            block = self._generate_random_residual_fc_block()
            
            # Test eval mode
            block.eval()
            
            # Verify all children are in eval mode
            for child in block.modules():
                self.assertFalse(child.training, "Child module should be in eval mode")
            
            # Find BatchNorm1d modules
            bn_modules = [m for m in block.modules() if isinstance(m, nn.BatchNorm1d)]
            self.assertGreater(len(bn_modules), 0, "No BatchNorm1d modules found")
            
            # Create a small batch (size 1)
            input_tensor = torch.randn(1, block.in_features)
            
            # This should not raise an error in eval mode
            try:
                _ = block(input_tensor)
            except Exception as e:
                self.fail(f"Block in eval mode raised an error with batch size 1: {e}")
            
            # Test train mode
            block.train()
            
            # Verify all children are in train mode
            for child in block.modules():
                self.assertTrue(child.training, "Child module should be in train mode")
            
            # Store running stats in train mode
            train_input = torch.randn(2, block.in_features)
            _ = block(train_input)  # Run with batch size > 1
            train_running_means = [bn.running_mean.clone() for bn in bn_modules]
            
            # Set to eval mode
            block.eval()
            
            # Run with different data
            different_data = torch.randn(2, block.in_features)
            _ = block(different_data)
            
            # In eval mode, running stats should not change
            for i, bn in enumerate(bn_modules):
                self.assertTrue(torch.allclose(train_running_means[i], bn.running_mean), 
                            "BatchNorm running mean should not change in eval mode")
    
    def _test_to_device(self):
        """Test that to() method works correctly"""
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        for _ in range(5):
            # Create a random block
            block = self._generate_random_residual_fc_block()
            
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
        for _ in range(10):
            # Create a random block
            block = self._generate_random_residual_fc_block()
            
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
            
            # Check that we have parameters for both streams when appropriate
            if hasattr(block, '_adaptive_layer'):
                adaptive_params = any('_adaptive_layer' in name for name in named_params.keys())
                self.assertTrue(adaptive_params, "No parameters found for adaptive layer")
    
    def _test_batch_size_one_handling(self):
        """Test handling of batch size 1 (important for BatchNorm)"""
        # Create a block
        block = self._generate_random_residual_fc_block()
        
        # Create input with batch size 1
        input_tensor = torch.randn(1, block.in_features)
        
        # Eval mode should work with batch size 1
        block.eval()
        output = block(input_tensor)
        self.assertEqual(output.shape, (1, block.output))
        
        # Train mode should only work with batch size > 1
        block.train()
        with self.assertRaises(ValueError):
            _ = block(input_tensor)
        
        # Train mode with batch size > 1 should work
        input_tensor = torch.randn(2, block.in_features)
        output = block(input_tensor)
        self.assertEqual(output.shape, (2, block.output))
    
    def _test_force_residual(self):
        """Test that force_residual works correctly"""
        for _ in range(100):
            # Test with force_residual=True
            try:
                block = self._generate_random_residual_fc_block(matching_dimensions=True, force_residual=True)
            except ValueError:
                # it is possible to have a ValueError when the dimensions are the same (mainly for the ExponentialResidualFCBlock)
                break 
            
            # Should have an adaptive layer
            self.assertTrue(hasattr(block, '_adaptive_layer'), 
                            "Block should have adaptive layer when force_residual=True")

            block.eval() # make sure to set the block to eval mode for consistent output

            # Test forward pass
            x = torch.randn(2, block.in_features, requires_grad=False)

            ms, rs = block._get_main_stream(), block._get_residual_stream()

            self.assertTrue(ms is block._block, "Main stream should be the block")
            self.assertTrue(rs is block._adaptive_layer, "Residual stream should be the adaptive layer")

            block_output, adaptive_layer_output, output = block.forward(x, debug=True)
            # Manually compute the expected output
            main_output = block._block.forward(x)
            residual_output = block._adaptive_layer.forward(x)
            expected_output = main_output + residual_output

            self.assertTrue(torch.allclose(block_output, main_output),
                            "Block output should be main output")

            self.assertTrue(torch.allclose(adaptive_layer_output, residual_output),
                            "Adaptive layer output should be residual output")  

            self.assertTrue(torch.allclose(output, expected_output),
                            "Output should be main + adaptive when force_residual=True")

    def _test_same_dimensions(self):
        """Test that the block has the same dimensions"""
        for _ in range(100):
            try:
                block = self._generate_random_residual_fc_block(matching_dimensions=True, force_residual=False)
            except ValueError:
                # it is possible to have a ValueError when the dimensions are the same (mainly for the ExponentialResidualFCBlock)
                break 

            self.assertEqual(block.in_features, block.output)

            block.eval() # make sure to set the block to eval mode for consistent output
            # Test forward pass
            x = torch.randn(2, block.in_features)
            output = block(x)
            
            # Manually compute the expected output
            main_output = block._block(x)
            expected_output = main_output + x
            
            self.assertTrue(torch.allclose(output, expected_output),
                            "Output should be main + input when dimensions match")

    def _test_different_dimensions(self):
        """Test that the block has different dimensions"""
        # if the output and input dimensions are different, there should be a residual stream, regardless of the force_residual argument 
        for _ in range(100):
            for fr in [True, False]:
                block = self._generate_random_residual_fc_block(matching_dimensions=False, force_residual=fr)
                self.assertNotEqual(block.in_features, block.output)
                self.assertIsNotNone(block._adaptive_layer)

                block.eval() # make sure to set the block to eval mode for consistent output

                # Test forward pass
                x = torch.randn(2, block.in_features, requires_grad=False)
                output = block(x)
                
                # Manually compute the expected output
                main_output = block._block(x)
                residual_output = block._adaptive_layer(x)
                expected_output = main_output + residual_output
                
                self.assertTrue(torch.allclose(output, expected_output),
                                "Output should be main + adaptive when dimensions don't match")



class TestGenericResidualFCBlock(ResidualFCBlockTestBase):
    """
    Tests for GenericResidualFCBlock
    """
    def _generate_random_units(self, in_features, output, num_layers):
        """Generate a random list of units for hidden layers"""
        units = [in_features]
        
        min_dim, max_dim = sorted([output, in_features])

        # Generate hidden layer units
        for _ in range(num_layers - 1):
            units.append(ri(max(min_dim, 10), max(max_dim, 64)))
            
        units.append(output)
        return units
    
    def _generate_random_residual_fc_block(self, 
                                         num_layers=None, 
                                         activation=None, 
                                         dropout=None, 
                                         force_residual=None,
                                         matching_dimensions=None):
        """Generate a GenericResidualFCBlock for testing"""
        if num_layers is None:
            num_layers = ri(2, 5)
            
        # Set default in_features and output
        in_features = ri(10, 100)
        output = ri(10, 100)
        
        # Handle matching_dimensions parameter
        if matching_dimensions is not None:
            if matching_dimensions:
                output = in_features
            else:
                # Make sure they don't match
                while output == in_features:
                    output = ri(10, 100)
        
        if activation is None:
            activation = random.choice(self.activation_names)
            
        units = self._generate_random_units(in_features, output, num_layers)
        
        if dropout is None:
            if random.choice([True, False]):
                # Use a single dropout value
                dropout_value = random.uniform(0.1, 0.5)
            else:
                # Use a list of dropout values
                dropout_value = [random.uniform(0.1, 0.5) for _ in range(num_layers - 1)]
        elif dropout is False:
            dropout_value = None
        else:
            dropout_value = dropout
            
        if force_residual is None:
            force_residual = random.choice([True, False])
            
        return GenericResidualFCBlock(
            output=output,
            in_features=in_features,
            num_layers=num_layers,
            units=units,
            activation=activation,
            dropout=dropout_value,
            force_residual=force_residual
        )
    
    def test_units_validation(self):
        """Test that the block validates the units list length"""
        in_features = ri(10, 100)
        output = ri(2, 20)
        num_layers = ri(2, 5)
        
        # Create valid units list
        valid_units = self._generate_random_units(in_features, output, num_layers)
        
        # This should work
        GenericResidualFCBlock(
            output=output,
            in_features=in_features,
            num_layers=num_layers,
            units=valid_units,
            activation='relu'
        )
        
        # Test with too few units
        invalid_units = valid_units[:-1]  # Remove last element
        
        # This should raise a ValueError
        with self.assertRaises(ValueError):
            GenericResidualFCBlock(
                output=output,
                in_features=in_features,
                num_layers=num_layers,
                units=invalid_units,
                activation='relu'
            )
    
    def test_consecutive_layers_connect_properly(self):
        """Test that consecutive layers connect properly"""
        for _ in range(10):
            # Create a random block
            block = self._generate_random_residual_fc_block()
            
            # Get all BasicLinearBlock instances
            children = list(block._block.children())
            
            # Check that consecutive layers connect properly
            for i in range(len(children) - 1):
                self.assertEqual(children[i].out_features, children[i+1].in_features,
                               f"Layer {i} out_features ({children[i].out_features}) does not match " +
                               f"Layer {i+1} in_features ({children[i+1].in_features})")
            
            # Check that first layer's in_features matches block's in_features
            self.assertEqual(children[0].in_features, block.in_features)
            
            # Check that last layer's out_features matches block's output
            self.assertEqual(children[-1].out_features, block.output)
    
    # Run all the tests from the base class
    def test_block_structure(self):
        self._test_block_structure()
    
    def test_adaptive_layer_creation(self):
        self._test_adaptive_layer_creation()
    
    def test_residual_forward_pass(self):
        self._test_residual_forward_pass()
    
    def test_different_activations(self):
        self._test_different_activations()
    
    def test_with_scalar_dropout(self):
        self._test_with_scalar_dropout()
    
    def test_with_list_dropout(self):
        self._test_with_list_dropout()
    
    def test_without_dropout(self):
        self._test_without_dropout()
    
    def test_forward_pass_shape(self):
        self._test_forward_pass_shape()
    
    def test_train_and_eval_modes(self):
        self._test_train_and_eval_modes()
    
    def test_to_device(self):
        self._test_to_device()
    
    def test_parameters_access(self):
        self._test_parameters_access()
    
    def test_batch_size_one_handling(self):
        self._test_batch_size_one_handling()

    def test_force_residual(self):
        self._test_force_residual()

    def test_same_dimensions(self):
        self._test_same_dimensions()

    def test_different_dimensions(self):
        self._test_different_dimensions()


class TestExponentialResidualFCBlock(ResidualFCBlockTestBase):
    """
    Tests for ExponentialResidualFCBlock
    """
    def _generate_random_residual_fc_block(self, 
                                         num_layers=None, 
                                         activation=None, 
                                         dropout=None, 
                                         force_residual=None,
                                         matching_dimensions=None):
        """Generate an ExponentialResidualFCBlock for testing"""
        if num_layers is None:
            num_layers = ri(2, 5)
            
        # Set default in_features and output
        in_features = ri(32, 1024)  # Use powers of 2 to ensure good exponential scaling
        output = ri(2, 16)
        
        
        if activation is None:
            activation = random.choice(self.activation_names)
            
        if dropout is None:
            if random.choice([True, False]):
                # Use a single dropout value
                dropout_value = random.uniform(0.1, 0.5)
            else:
                # Use a list of dropout values
                dropout_value = [random.uniform(0.1, 0.5) for _ in range(num_layers - 1)]
        elif dropout is False:
            dropout_value = None
        else:
            dropout_value = dropout
            
        if force_residual is None:
            force_residual = random.choice([True, False])
            
        return ExponentialResidualFCBlock(
            output=output,
            in_features=in_features,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout_value,
            force_residual=force_residual
        )
    
    def test_exponential_scaling(self):
        """Test that the block uses exponential scaling for units"""
        for _ in range(10):
            block = self._generate_random_residual_fc_block()
            
            # Check that exponential scaling is applied
            # Units should decrease roughly exponentially from in_features to output
            units = block.units
            
            # First unit should be in_features
            self.assertEqual(units[0], block.in_features)
            
            # Last unit should be output
            self.assertEqual(units[-1], block.output)
            
            # Check that units are monotonically decreasing
            if units[0] > units[-1]:
                for i in range(len(units) - 1):
                    if units[i] < units[i+1]:
                        self.fail("The values of the units should be monotonincally decreasing or increasing depending on the input and output dimensions")
            else:
                for i in range(len(units) - 1):
                    if units[i] > units[i+1]:
                        self.fail("The values of the units should be monotonincally decreasing or increasing depending on the input and output dimensions")


    # Run all the tests from the base class
    def test_block_structure(self):
        self._test_block_structure()
    
    def test_different_activations(self):
        self._test_different_activations()
    
    def test_with_scalar_dropout(self):
        self._test_with_scalar_dropout()
    
    def test_with_list_dropout(self):
        self._test_with_list_dropout()
    
    def test_without_dropout(self):
        self._test_without_dropout()
    
    def test_forward_pass_shape(self):
        self._test_forward_pass_shape()
    
    def test_train_and_eval_modes(self):
        self._test_train_and_eval_modes()
    
    def test_to_device(self):
        self._test_to_device()
    
    def test_parameters_access(self):
        self._test_parameters_access()
    
    def test_batch_size_one_handling(self):
        self._test_batch_size_one_handling()
    

    # since the input and output dimensions are always different for the ExponentialResidualFCBlock, 
    # there is no need to test the force_residual or the case where the dimensions match

    def test_different_dimensions(self):
        self._test_different_dimensions()


if __name__ == '__main__':
    unittest.main() 