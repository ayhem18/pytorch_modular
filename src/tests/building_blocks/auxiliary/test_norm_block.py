import torch
import random
import unittest

from torch import nn
from typing import Tuple, Dict, List, Optional

from tests.custom_base_test import CustomModuleBaseTest
from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser
from mypt.building_blocks.auxiliary.norm_act import NormActBlock


class TestNormBlock(CustomModuleBaseTest):
    """
    Test class for NormActBlock and ConditionedNormActBlock
    """

    def setUp(self):
        """
        Initialize the test case parameters
        """
        self.dim_analyser = DimensionsAnalyser()    
        
    def _generate_random_norm_act_block(self) -> Tuple[NormActBlock, Dict]:
        """
        Generate a random NormActBlock with random parameters
        """
        # Random normalization type - using the updated normalization names
        # TODO: UNDERSTAND LAYERNORM AND ADD IT TO THE TESTS
        norm_type = random.choice(['batchnorm1d', 'batchnorm2d', 'groupnorm'])
                
        # Random activation function
        activation_type = random.choice(['relu', 'leaky_relu', 'gelu', 'tanh', 'sigmoid', 'silu'])

        # Random number of features for normalization
        num_features = random.randint(1, 128)


        # Set up normalization parameters
        if norm_type == 'batchnorm1d':
            norm_params = {'num_features': num_features}
        
        elif norm_type == 'batchnorm2d':
            norm_params = {'num_features': num_features}

        # elif norm_type == 'layernorm':
        #     norm_params = {'normalized_shape': num_features}

        elif norm_type == 'instancenorm1d':
            norm_params = {'num_features': num_features}

        elif norm_type == 'instancenorm2d':
            norm_params = {'num_features': num_features}

        elif norm_type == 'groupnorm':
            # GroupNorm needs num_groups and num_channels
            num_groups = num_features
            num_channels = num_features * random.randint(2, 100)
            num_features = num_channels
            norm_params = {'num_groups': num_groups, 'num_channels': num_channels}

        # Set up activation parameters
        if activation_type == 'leaky_relu':
            activation_params = {'negative_slope': 0.01}
        else:
            activation_params = {}
        
        # Create the block
        block = NormActBlock(
            normalization=norm_type,
            normalization_params=norm_params,
            activation=activation_type,
            activation_params=activation_params
        )
        
        # Return the block and its parameters
        params = {
            'norm_type': norm_type,
            'norm_params': norm_params,
            'activation_type': activation_type,
            'activation_params': activation_params,
            'num_features': num_features
        }
        
        return block, params


    def _get_valid_input(self, params: Dict, batch_size: int = 2) -> torch.Tensor:
        """
        Generate a random input tensor with the correct shape.
        """
        if params['norm_type'] in ['batchnorm1d', 'instancenorm1d']:
            return torch.randn(batch_size, params['num_features'])
        else:
            return torch.randn(batch_size, params['num_features'], 16, 16)

    def test_norm_act_block_structure(self):
        """Test that NormActBlock is structured correctly"""
        for _ in range(100):  # Test multiple random configurations
            block, params = self._generate_random_norm_act_block()
            
            # Check that we have both normalization and activation components
            self.assertIsNotNone(block.block.normalization, "Normalization component should not be None")
            self.assertIsNotNone(block.block.activation, "Activation component should not be None")
            
            # Check that the block has the expected internal structure
            self.assertIsInstance(block.block, nn.Sequential, "Internal block should be Sequential")
            self.assertEqual(len(block.block), 2, "Block should have exactly 2 components")
            
            # Check that normalization is the first component
            norm_layer = block.block[0]
            
            if params['norm_type'] == 'batchnorm1d':
                self.assertIsInstance(norm_layer, nn.BatchNorm1d, "First component should be BatchNorm1d")
                self.assertEqual(norm_layer.num_features, params['num_features'], 
                                 "BatchNorm1d num_features doesn't match expected value")

            elif params['norm_type'] == 'layernorm':
                self.assertIsInstance(norm_layer, nn.LayerNorm, "First component should be LayerNorm")
                self.assertEqual(norm_layer.normalized_shape[0], params['num_features'], 
                                 "LayerNorm normalized_shape doesn't match expected value")

            elif params['norm_type'] == 'instancenorm1d':
                self.assertIsInstance(norm_layer, nn.InstanceNorm1d, "First component should be InstanceNorm1d")
                self.assertEqual(norm_layer.num_features, params['num_features'], 
                                 "InstanceNorm1d num_features doesn't match expected value")
                                 
            elif params['norm_type'] == 'instancenorm2d':
                self.assertIsInstance(norm_layer, nn.InstanceNorm2d, "First component should be InstanceNorm2d")
                self.assertEqual(norm_layer.num_features, params['num_features'], 
                                 "InstanceNorm2d num_features doesn't match expected value")
                                 
            elif params['norm_type'] == 'groupnorm':
                self.assertIsInstance(norm_layer, nn.GroupNorm, "First component should be GroupNorm")
                self.assertEqual(norm_layer.num_channels, params['num_features'], 
                                 "GroupNorm num_channels doesn't match expected value")
                self.assertEqual(norm_layer.num_groups, params['norm_params']['num_groups'],
                                 "GroupNorm num_groups doesn't match expected value")
            
            # Check that activation is the second component
            act_layer = block.block[1]
            if params['activation_type'] == 'relu':
                self.assertIsInstance(act_layer, nn.ReLU, "Second component should be ReLU")
            elif params['activation_type'] == 'leaky_relu':
                self.assertIsInstance(act_layer, nn.LeakyReLU, "Second component should be LeakyReLU")
                self.assertEqual(act_layer.negative_slope, params['activation_params']['negative_slope'],
                                 "LeakyReLU negative_slope doesn't match expected value")
            elif params['activation_type'] == 'gelu':
                self.assertIsInstance(act_layer, nn.GELU, "Second component should be GELU")
            elif params['activation_type'] == 'tanh':
                self.assertIsInstance(act_layer, nn.Tanh, "Second component should be Tanh")
            elif params['activation_type'] == 'sigmoid':
                self.assertIsInstance(act_layer, nn.Sigmoid, "Second component should be Sigmoid")
    

    def test_norm_act_block_forward_1d(self):
        """Test that NormActBlock forward pass works with 1D input"""
        for _ in range(100):  # Test multiple random configurations
            block, params = self._generate_random_norm_act_block()
            
            # Skip GroupNorm2d for 1D test
            if params['norm_type'] in ['groupnorm', 'batchnorm2d', 'instancenorm2d', 'layernorm']:
                continue
                
            # Create a random batch of inputs
            batch_size = random.randint(1, 16)
            input_tensor = self._get_valid_input(params, batch_size)
            
            # Run a forward pass
            try:
                # make sure to set the model to eval mode
                block.eval()

                output = block(input_tensor)
                
                # Check that output has the same shape as input
                self.assertEqual(output.shape, input_tensor.shape, 
                                 "Output shape should match input shape")
                
                # Get predicted output shape using DimensionsAnalyser
                expected_shape = self.dim_analyser.analyse_dimensions(
                    input_shape=tuple(input_tensor.shape),
                    net=block
                )
                
                # Compare shapes
                self.assertEqual(tuple(output.shape), expected_shape, 
                                f"Mismatch in output shape: got {tuple(output.shape)}, expected {expected_shape}")
                
            except Exception as e:
                self.fail(f"Forward pass raised an exception: {e}")
    
    def test_norm_act_block_forward_2d(self):
        """Test that NormActBlock forward pass works with 2D spatial input"""
        for _ in range(100):  # Test multiple random configurations
            # Start with a 1D block
            block, params = self._generate_random_norm_act_block()
            
            if params['norm_type'] not in ['groupnorm', 'batchnorm2d', 'instancenorm2d']:
                continue
            
            # For 2D spatial input, we need to create a new block with 2D normalization
            if params['norm_type'] == 'batchnorm2d':
                # Create a block with BatchNorm2d
                block = NormActBlock(
                    normalization='batchnorm2d',
                    normalization_params=params['norm_params'],
                    activation=params['activation_type'],
                    activation_params=params['activation_params']
                )
            elif params['norm_type'] == 'instancenorm2d':
                # Create a block with InstanceNorm2d
                block = NormActBlock(
                    normalization='instancenorm2d',
                    normalization_params=params['norm_params'],
                    activation=params['activation_type'],
                    activation_params=params['activation_params']
                )

            elif params['norm_type'] == 'groupnorm':
                # Create a block with GroupNorm2d
                block = NormActBlock(
                    normalization='groupnorm',
                    normalization_params=params['norm_params'],
                    activation=params['activation_type'],
                    activation_params=params['activation_params']
                )
            
            batch_size = random.randint(1, 16)
            input_tensor = self._get_valid_input(params, batch_size)
            
            # Run a forward pass
            try:
                # make sure to set the model to eval mode
                block.eval()

                output = block(input_tensor)
                
                # Check that output has the same shape as input
                self.assertEqual(output.shape, input_tensor.shape, 
                                 "Output shape should match input shape")
                
                # Get predicted output shape using DimensionsAnalyser
                expected_shape = self.dim_analyser.analyse_dimensions(
                    input_shape=tuple(input_tensor.shape),
                    net=block
                )
                
                # Compare shapes
                self.assertEqual(tuple(output.shape), expected_shape, 
                                f"Mismatch in output shape: got {tuple(output.shape)}, expected {expected_shape}")
                
            except Exception as e:
                self.fail(f"Forward pass raised an exception: {e}")
    
    def test_eval_mode(self):
        """Test that eval mode is correctly set across the block"""
        for _ in range(100):
            block, _ = self._generate_random_norm_act_block()
            super()._test_eval_mode(block)
    
    def test_train_mode(self):
        """Test that train mode is correctly set across the block"""
        for _ in range(100):
            block, _ = self._generate_random_norm_act_block()
            super()._test_train_mode(block)
    
    def test_consistent_output_in_eval_mode(self):
        """Test that the block produces consistent output in eval mode"""
        for _ in range(100):
            block, params = self._generate_random_norm_act_block()
            input_tensor = self._get_valid_input(params, batch_size=2)            
            super()._test_consistent_output_in_eval_mode(block, input_tensor)
    
    def test_batch_size_one_in_train_mode(self):
        """Test that the block handles batch size 1 in train mode"""
        for _ in range(100):
            block, params = self._generate_random_norm_act_block()
            input_tensor = self._get_valid_input(params, batch_size=1)
            super()._test_batch_size_one_in_train_mode(block, input_tensor)
    
    def test_batch_size_one_in_eval_mode(self):
        """Test that the block handles batch size 1 in eval mode"""
        for _ in range(100):
            block, params = self._generate_random_norm_act_block()
            input_tensor = self._get_valid_input(params, batch_size=1)
            super()._test_batch_size_one_in_eval_mode(block, input_tensor)
    
    def test_named_parameters_length(self):
        """Test that named_parameters and parameters have the same length"""
        for _ in range(100):
            block, _ = self._generate_random_norm_act_block()
            super()._test_named_parameters_length(block)
    
    def test_to_device(self):
        """Test that the block can be moved between devices correctly"""
        for _ in range(100):
            block, params = self._generate_random_norm_act_block()
            input_tensor = self._get_valid_input(params, batch_size=2)
            super()._test_to_device(block, input_tensor)
    
    def test_consistent_output_without_dropout_bn(self):
        """Test that modules without dropout or batch normalization produce consistent output"""
        # Test only with normalization types that don't have stochastic behavior like dropout
        for _ in range(100):
            block, params = self._generate_random_norm_act_block()
            input_tensor = self._get_valid_input(params, batch_size=2)
            super()._test_consistent_output_without_dropout_bn(block, input_tensor)
    
    def test_wrapper_like_behavior(self):
        """Test that NormActBlock behaves like a wrapper module"""
        for _ in range(100):
            block, _ = self._generate_random_norm_act_block()
            
            # Test that it has the expected properties and methods
            self.assertTrue(hasattr(block, 'block'), "Block should have a 'block' property")
            self.assertIsInstance(block.block, nn.Sequential, "block property should return a Sequential module")
            
            # Test that modules() includes all submodules
            all_modules = list(block.modules())
            self.assertIn(block._block.normalization, all_modules, "modules() should include the normalization layer")
            self.assertIn(block._block.activation, all_modules, "modules() should include the activation layer")


if __name__ == '__main__':
    #TODO: understand layernorm and added proper tests !!!

    import mypt.code_utils.pytorch_utils as pu
    pu.seed_everything(42)
    unittest.main()
    