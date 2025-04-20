import numpy as np
import torch
import random
import unittest

from abc import ABC, abstractmethod
from torch import nn
from random import randint as ri

import mypt.code_utils.pytorch_utils as pu

from mypt.building_blocks.linear_blocks.components import BasicLinearBlock
from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser
from mypt.building_blocks.linear_blocks.fc_blocks import GenericFCBlock, ExponentialFCBlock


class FCBlockTestBase(unittest.TestCase):
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
                                dropout: float | list[float] | None = None):
        """Generate a random FC block with configurable parameters"""
        pass

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
        # Create a block
        block = self._generate_random_fc_block()
        
        # Create input with batch size 1
        input_tensor = torch.randn(1, block.in_features)
        
        # Eval mode should work with batch size 1
        block.eval()
        output = block(input_tensor)
        self.assertEqual(output.shape, (1, block.output))
        
        # Train mode might work with batch size > 1
        block.train()
        input_tensor = torch.randn(2, block.in_features)
        output = block(input_tensor)
        self.assertEqual(output.shape, (2, block.output))

    def _test_train_and_eval_modes(self):
        """Test that train and eval modes work correctly"""
        for _ in range(100):
            # Create a block
            block = self._generate_random_fc_block()
            
            # Test eval mode
            block.eval()
            
            # Verify all children are in eval mode
            for child in block.modules():
                self.assertFalse(child.training, "BatchNorm should be in eval mode")
            
            # Create a small batch (size 1)
            input_tensor = torch.randn(1, block.in_features)
            
            # This should not raise an error in eval mode
            output = block(input_tensor)
            self.assertEqual(output.shape, (1, block.output))
            
            # Test train mode
            block.train()
            
            # Verify all children are in train mode
            for child in block.modules():
                self.assertTrue(child.training, "all children should be in train mode")

            # Non-final blocks with BatchNorm may have issues with batch size 1
            # Find BatchNorm1d modules
            bn_modules = [m for m in block.modules() if isinstance(m, nn.BatchNorm1d)]
            
            if bn_modules:
                # In train mode with batch size > 1 should work
                input_tensor = torch.randn(2, block.in_features)
                _ = block(input_tensor)  # No error
                
                # Store running stats
                train_running_means = [bn.running_mean.clone() for bn in bn_modules]
                
                # Set to eval mode
                block.eval()
                
                # Run with different data
                input_tensor = torch.randn(2, block.in_features)
                _ = block(input_tensor)
                
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

            # # Check named parameters 
            named_params = dict(block.named_parameters())
            self.assertEqual(len(named_params), len(params))
            
            num_layers = block.num_layers
            all_param_prefixes = set()
            
            # Check that parameter names follow the expected pattern
            for name, param in named_params.items():
                self.assertTrue(name.startswith('fc_'), 
                               f"Parameter name {name} does not start with 'fc_'") 
                self.assertIsInstance(param, torch.nn.Parameter)
                self.assertTrue('.' in name)
                all_param_prefixes.add(name.split('.')[0])

            for i in range(1, num_layers + 1):
                self.assertTrue(f'fc_{i}' in all_param_prefixes, 
                                f"the fc_{i} prefix is missing from the parameter names")

    def _test_forward_pass_shape(self):
        """Test that forward pass produces output with expected shape"""
        for _ in range(100):
            # Generate a random block
            block = self._generate_random_fc_block()
            
            # Create batch of random size
            block.eval()  # Set to eval mode to handle batch norm with batch size 1
            batch_size = ri(1, 16)
            input_tensor = torch.randn(batch_size, block.in_features)
            
            # Get actual output shape
            output = block(input_tensor)
            actual_shape = tuple(output.shape)
            
            # Expected shape is (batch_size, output)
            expected_shape = (batch_size, block.output)
            
            # Assert shapes match
            self.assertEqual(actual_shape, expected_shape, 
                            f"Output shape mismatch: got {actual_shape}, expected {expected_shape}")
            
            # Test using the dimension analyzer as well
            analyzed_shape = self.dim_analyser.analyse_dimensions(
                input_shape=(batch_size, block.in_features),
                net=block
            )
            self.assertEqual(analyzed_shape, expected_shape,
                            f"Dimension analyzer shape mismatch: got {analyzed_shape}, expected {expected_shape}")



class TestGenericFCBlock(FCBlockTestBase):
    """
    Concrete test class for testing GenericFCBlock.
    This class will be picked up and executed by the test runner.
    """
    
    def _generate_random_units(self, in_features: int, output: int, num_layers: int) -> list[int]:
        """Generate a random list of units for hidden layers"""
        units = [in_features]
        
        # Generate hidden layer units
        for _ in range(num_layers - 1):
            units.append(ri(max(output, 10), max(in_features, 64)))
            
        units.append(output)
        return units

    def _generate_random_fc_block(self, 
                                num_layers: int | None = None, 
                                activation: str | None = None, 
                                dropout: float | list[float] | None = None) -> GenericFCBlock:
        """Concrete implementation that generates a GenericFCBlock"""
        if num_layers is None:
            num_layers = ri(2, 5)
            
        in_features = ri(10, 100)
        output = ri(2, 20)  # Typically fewer output classes
        
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
            
        return GenericFCBlock(
            output=output,
            in_features=in_features,
            num_layers=num_layers,
            units=units,
            activation=activation,
            dropout=dropout_value
        )

    def test_units_validation(self):
        """Test that the block validates the units list length"""
        for _ in range(100):
            in_features = ri(10, 100)
            output = ri(2, 20)
            num_layers = ri(2, 5)
            
            # Create valid units list
            valid_units = self._generate_random_units(in_features, output, num_layers)
            
            # This should work
            GenericFCBlock(
                output=output,
                in_features=in_features,
                num_layers=num_layers,
                units=valid_units,
                activation='relu'
            )
            
            # Create invalid units list (too short)
            random_subset_size = ri(1, len(valid_units) - 1)
            invalid_units = random.sample(valid_units, random_subset_size)

            # This should raise a ValueError
            with self.assertRaises(ValueError):
                GenericFCBlock(
                    output=output,
                    in_features=in_features,
                    num_layers=num_layers,
                    units=invalid_units,
                    activation='relu'
                )
                
            # Create invalid units list (add a random list to the units)
            invalid_units = valid_units + ([ri(10, 100)] * ri(1, 10))
            
            # This should raise a ValueError
            with self.assertRaises(ValueError):
                GenericFCBlock(
                    output=output,
                    in_features=in_features,
                    num_layers=num_layers,
                    units=invalid_units,
                    activation='relu'
                )

    def test_consecutive_layers_connect_properly(self):
        """Test that consecutive layers have matching dimensions"""
        for _ in range(20):
            # Create a random block
            block = self._generate_random_fc_block()
            
            # Get all BasicLinearBlock instances
            children = list(block.children())
            
            # Check that consecutive layers connect properly
            for i in range(len(children) - 1):
                self.assertEqual(children[i].out_features, children[i+1].in_features,
                               f"Layer {i} out_features ({children[i].out_features}) does not match " +
                               f"Layer {i+1} in_features ({children[i+1].in_features})")
            
            # Check that first layer's in_features matches block's in_features
            self.assertEqual(children[0].in_features, block.in_features)
            
            # Check that last layer's out_features matches block's output
            self.assertEqual(children[-1].out_features, block.output)

    def test_forward_pass_shape(self):
        super()._test_forward_pass_shape()  

    # inherited tests from the base class
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
        super()._test_to_device()

    def test_parameters_access(self):
        super()._test_parameters_access()        

    def test_train_and_eval_modes(self):
        super()._test_train_and_eval_modes()

    def test_to_device(self):
        super()._test_to_device()

    def test_parameters_access(self):
        super()._test_parameters_access()


class TestExponentialLinearBlock(FCBlockTestBase):
    def setUp(self):
        self.dim_analyser = DimensionsAnalyser()
        self.activation_types = [type(BasicLinearBlock._ACTIVATION_MAP[t]) for t in BasicLinearBlock._ACTIVATIONS]
        self.activation_names = BasicLinearBlock._ACTIVATIONS.copy()
        
    def _generate_random_fc_block(self, 
                                num_layers: int | None = None, 
                                activation: str | None = None, 
                                dropout: float | list[float] | None = None) -> GenericFCBlock:
        """Concrete implementation that generates a GenericFCBlock"""
        if num_layers is None:
            num_layers = ri(2, 5)
            
        in_features = ri(10, 100)
        output = ri(2, 20)  # Typically fewer output classes
        
        if activation is None:
            activation = random.choice(self.activation_names)
        
        in_features = ri(10, 1000)
        output = ri(2, 20)

        while int(np.log2(in_features)) == int(np.log2(output)):
            in_features = ri(10, 1000)
            output = ri(2, 20)

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
            
        return ExponentialFCBlock(
            output=output,
            in_features=in_features,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout_value
        )


    def test_forward_pass_shape(self):
        super()._test_forward_pass_shape()  

    # inherited tests from the base class
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
        super()._test_to_device()

    def test_parameters_access(self):
        super()._test_parameters_access()        

    def test_train_and_eval_modes(self):
        super()._test_train_and_eval_modes()

    def test_to_device(self):
        super()._test_to_device()

    def test_parameters_access(self):
        super()._test_parameters_access()





if __name__ == '__main__':
    unittest.main()