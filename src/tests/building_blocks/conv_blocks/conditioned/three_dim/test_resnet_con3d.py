import torch
import random
import unittest

from torch import nn
from typing import Tuple

from tests.custom_base_test import CustomModuleBaseTest
from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser
from mypt.building_blocks.conv_blocks.conditioned.three_dim.resnet_con_3d import CondThreeDimWResBlock


class TestConditionalWResBlock3D(CustomModuleBaseTest):
    """Test class for ConditionalWResBlock with 3D film conditioning"""
    
    def setUp(self):
        """Initialize test parameters"""
        self.dim_analyser = DimensionsAnalyser()
        
        # Define common test parameters
        self.in_channels_range = (1, 32)
        self.out_channels_range = (1, 64)
        self.cond_dimension_range = (8, 64)
        self.stride_options = [1, 2]
        self.dropout_options = [0.0, 0.1, 0.5]
        self.force_residual_options = [True, False]
        self.inner_dim_range = (16, 128)
        
        # For normalization and activation
        self.norm_types = ['batchnorm2d', 'groupnorm']
        self.activation_types = ['relu', 'leaky_relu', 'gelu']
    
    def _get_valid_input(self, block=None, batch_size=2, height=32, width=32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a random input tensor and condition tensor with the correct shape"""
        if block is None:
            in_channels = random.randint(*self.in_channels_range)
            cond_dimension = random.randint(*self.cond_dimension_range)
        else:
            in_channels = block._in_channels
            cond_dimension = block._cond_dimension
        
        # Generate input tensor
        x = torch.randn(batch_size, in_channels, height, width, requires_grad=False)
        
        # Generate 3D condition tensor (spatial dimensions match input)
        condition = torch.randn(batch_size, cond_dimension, height, width, requires_grad=False)
        
        return x, condition
    
    def _generate_random_conditional_resnet_block(self, 
                                                in_channels=None, 
                                                out_channels=None,
                                                cond_dimension=None,
                                                stride=None,
                                                dropout_rate=None,
                                                force_residual=None,
                                                inner_dim=None) -> CondThreeDimWResBlock:
        """Generate a random ConditionalWResBlock with configurable parameters"""
        # Set parameters or choose random values
        if in_channels is None:
            in_channels = random.randint(*self.in_channels_range)
        
        if out_channels is None:
            out_channels = random.randint(*self.out_channels_range)
        
        if cond_dimension is None:
            cond_dimension = random.randint(*self.cond_dimension_range)
        
        if stride is None:
            stride = random.choice(self.stride_options)
        
        if dropout_rate is None:
            dropout_rate = random.choice(self.dropout_options)
        
        if force_residual is None:
            force_residual = random.choice(self.force_residual_options)
            
        if inner_dim is None:
            inner_dim = random.randint(*self.inner_dim_range)
        
        # Choose normalization type
        norm_type = random.choice(self.norm_types)

        # try both None and {} as norm_params, the class should handle both cases correctly 
        norm1_params = random.choice([None, {}])
        norm2_params = random.choice([None, {}])

        # Choose activation type and parameters
        activation_type = random.choice(self.activation_types)
        activation_params = {'inplace': True} if activation_type == 'relu' else {}
        
        film_activation = random.choice(self.activation_types)
        film_activation_params = {'inplace': True} if film_activation == 'relu' else {}
        
        return CondThreeDimWResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_dimension=cond_dimension,
            inner_dim=inner_dim,
            stride=stride,
            dropout_rate=dropout_rate,
            norm1=norm_type,
            norm1_params=norm1_params,
            norm2=norm_type,
            norm2_params=norm2_params,
            activation=activation_type,
            activation_params=activation_params,
            film_activation=film_activation,
            film_activation_params=film_activation_params,
            force_residual=force_residual
        )
    
    ########################## Block Structure Tests ##########################
    # @unittest.skip("skip for now")
    def test_block_structure(self):
        """Test that the ConditionalWResBlock has the correct structure"""
        for _ in range(100):
            block = self._generate_random_conditional_resnet_block()
            
            # Check that the block components exist
            self.assertIsInstance(block._components, nn.ModuleDict)
            
            # Check required components
            self.assertIn('film1', block._components)
            self.assertIn('conv1', block._components)
            self.assertIn('dropout', block._components)
            self.assertIn('film2', block._components)
            self.assertIn('conv2', block._components)
            
            # Check if shortcut exists when needed
            if not hasattr(block, '_shortcut') or block._shortcut is None:
                # Should only be None when dimensions match and force_residual=False
                self.assertEqual(block._stride, 1)
                self.assertEqual(block._in_channels, block._out_channels)
                self.assertFalse(block._force_residual)
            else:
                # Shortcut should be a Conv2d
                self.assertIsInstance(block._shortcut, nn.Conv2d)
                self.assertEqual(block._shortcut.in_channels, block._in_channels)
                self.assertEqual(block._shortcut.out_channels, block._out_channels)
    

    # @unittest.skip("skip for now")
    def test_forward_pass_basic(self):
        """Test basic forward pass functionality"""
        for _ in range(100):
            block = self._generate_random_conditional_resnet_block()
            x, condition = self._get_valid_input(block)
            
            # Regular forward pass
            output = block(x, condition)
            
            # Check output dimensions
            batch_size, in_channels, height, width = x.shape
            expected_height = height if block._stride == 1 else (height + 1) // 2
            expected_width = width if block._stride == 1 else (width + 1) // 2
            
            self.assertEqual(output.shape, (batch_size, block._out_channels, expected_height, expected_width))
    
    # @unittest.skip("Skipping residual path creation tests for now")
    def test_forward_pass_debug_mode(self):
        """Test forward pass with debug=True"""
        for _ in range(100):
            block = self._generate_random_conditional_resnet_block()
            x, condition = self._get_valid_input(block)
            
            # Set to eval mode for consistent outputs
            block.eval()
            
            # Debug mode forward pass
            main_output, residual_output, combined_output = block(x, condition, debug=True)

            # Regular forward pass
            regular_output = block(x, condition)
            
            # Check that regular output matches combined output
            self.assertTrue(torch.allclose(regular_output, combined_output))
            
            # Check that combined output equals main + residual
            calculated_output = main_output + residual_output
            self.assertTrue(torch.allclose(calculated_output, combined_output))
    
    # @unittest.skip("Skipping residual path creation tests for now")
    def test_conditioning_effect(self):
        """Test that different conditioning tensors produce different outputs"""
        for _ in range(100):
            block = self._generate_random_conditional_resnet_block()
            x, condition1 = self._get_valid_input(block)
            
            # Create a different condition tensor
            condition2 = torch.randn_like(condition1)
            
            # Set to eval mode
            block.eval()
            
            # Get outputs with different conditions
            output1 = block(x, condition1)
            output2 = block(x, condition2)
            
            # Outputs should be different due to different conditioning
            self.assertFalse(torch.allclose(output1, output2))
    
    ########################## CustomModuleBaseTest Tests ##########################
    
    def test_eval_mode(self):
        """Test that the block can be set to evaluation mode"""
        for _ in range(100):
            block = self._generate_random_conditional_resnet_block()
            super()._test_eval_mode(block)
    
    def test_train_mode(self):
        """Test that the block can be set to training mode"""
        for _ in range(100):
            block = self._generate_random_conditional_resnet_block()
            x, condition = self._get_valid_input(block)
            super()._test_train_mode(block)
    
    def test_consistent_output_in_eval_mode(self):
        """Test consistent output in evaluation mode"""
        for _ in range(100):
            block = self._generate_random_conditional_resnet_block()
            x, condition = self._get_valid_input(block)
            super()._test_consistent_output_in_eval_mode(block, x, condition)
    
    def test_batch_size_one_in_train_mode(self):
        """Test handling of batch size 1 in training mode"""
        for _ in range(100):
            block = self._generate_random_conditional_resnet_block()
            x, condition = self._get_valid_input(block, batch_size=1)
            super()._test_batch_size_one_in_train_mode(block, x, condition)

    def test_batch_size_one_in_eval_mode(self):
        """Test handling of batch size 1 in evaluation mode"""
        for _ in range(100):
            block = self._generate_random_conditional_resnet_block()
            x, condition = self._get_valid_input(block, batch_size=1)
            super()._test_batch_size_one_in_eval_mode(block, x, condition)

    def test_named_parameters_length(self):
        """Test that named_parameters and parameters have the same length"""
        for _ in range(100):
            block = self._generate_random_conditional_resnet_block()
            super()._test_named_parameters_length(block)

    def test_to_device(self):
        """Test that the block can be moved between devices"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping device tests")
            
        for _ in range(100):
            block = self._generate_random_conditional_resnet_block()
            x, condition = self._get_valid_input(block)
            super()._test_to_device(block, x, condition)


if __name__ == '__main__':
    import mypt.code_utils.pytorch_utils as pu
    pu.seed_everything(42)
    unittest.main()
