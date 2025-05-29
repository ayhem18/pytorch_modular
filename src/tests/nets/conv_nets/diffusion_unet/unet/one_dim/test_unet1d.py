import torch
import random
import unittest

import torch.nn as nn
from typing import List, Optional, Tuple

from tests.custom_base_test import CustomModuleBaseTest
from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser
from mypt.nets.conv_nets.diffusion_unet.unet.one_dim.unet1d import UNet1DCond


class TestUNet1DCond(CustomModuleBaseTest):
    """Test class for UNet1DCond that verifies the builder design pattern and model functionality"""
    
    def setUp(self):
        """Initialize test parameters"""
        # Define common test parameters
        self.in_channels_range = (1, 16)
        self.out_channels_range = (1, 16)
        self.cond_dimension_range = (4, 32)
        self.num_resnet_blocks_range = (2, 4) # number of resnet blocks in the any Unetblock must be at least 2
        self.num_down_layers_range = (1, 4)
        self.dropout_options = [0.0, 0.1]
        self.downsample_types = ["conv", "avg_pool", "max_pool"]
        self.upsample_types = ["transpose_conv", "conv", "interpolate"]
        self.inner_dim_range = (16, 64)
        
        # For normalization and activation
        self.norm_types = ['batchnorm2d', 'groupnorm']
        self.activation_types = ['relu', 'leaky_relu', 'gelu']
    
    def _create_unet_model(self, 
                          in_channels: Optional[int] = None,
                          out_channels: Optional[int] = None,
                          cond_dimension: Optional[int] = None) -> UNet1DCond:
        """Create a basic UNet1DCond model without building it"""
        in_channels = in_channels or random.randint(*self.in_channels_range)
        out_channels = out_channels or random.randint(*self.out_channels_range)
        cond_dimension = cond_dimension or random.randint(*self.cond_dimension_range)
        
        return UNet1DCond(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_dimension=cond_dimension
        )
    
    def _get_valid_input(self, 
                        model: UNet1DCond, 
                        batch_size: int = 2, 
                        k: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate valid input for the UNet model
        
        Args:
            model: The UNet model
            batch_size: Batch size for the input
            k: Multiplier for spatial dimensions (input will be 2^num_down_layers * k)
            
        Returns:
            Tuple of (input tensor, condition tensor)
        """
        if not hasattr(model, "num_down_layers") or model.num_down_layers is None:
            # If model hasn't been built yet, use a default value
            num_down_layers = 3
        else:
            num_down_layers = model.num_down_layers
        
        # Make dimensions divisible by 2^num_down_layers
        spatial_size = 2**num_down_layers * k
        
        # Generate input tensor
        x = torch.randn(batch_size, model.input_channels, spatial_size, spatial_size)
        condition = torch.randn(batch_size, model.cond_dimension)
        
        return x, condition
    
    def _build_complete_model(self, model: Optional[UNet1DCond] = None) -> UNet1DCond:
        """Build a complete UNet model with all components"""
        if model is None:
            model = self._create_unet_model()
        
        # Random parameters
        num_down_layers = random.randint(*self.num_down_layers_range)
        num_resnet_blocks = random.randint(*self.num_resnet_blocks_range)
        mid_block_num_resnet_blocks = random.randint(*self.num_resnet_blocks_range)
        
        # Generate random channel configuration
        out_channels = [random.randint(16, 64) for _ in range(num_down_layers)]
        
        # Build down block
        model.build_down_block(
            num_down_layers=num_down_layers,
            num_resnet_blocks=num_resnet_blocks,
            out_channels=out_channels,
            downsample_types="conv",
            dropout_rate=random.choice(self.dropout_options)
        )
        
        # Build middle block
        model.build_middle_block(
            num_resnet_blocks=mid_block_num_resnet_blocks,
            dropout_rate=random.choice(self.dropout_options)
        )
        
        # Build up block
        model.build_up_block(
            num_resnet_blocks=num_resnet_blocks,
            upsample_types="transpose_conv",
            dropout_rate=random.choice(self.dropout_options)
        )
        
        return model
    
    ########################## Structure Tests ##########################
    
    def test_down_block_structure(self):
        """Test the structure of the down block after building"""
        for _ in range(20):
            model = self._create_unet_model()
            
            # Random parameters
            num_down_layers = random.randint(*self.num_down_layers_range)
            num_resnet_blocks = random.randint(*self.num_resnet_blocks_range)
            out_channels = [random.randint(16, 64) for _ in range(num_down_layers)]
            
            # Build down block
            model.build_down_block(
                num_down_layers=num_down_layers,
                num_resnet_blocks=num_resnet_blocks,
                out_channels=out_channels,
                downsample_types="conv"
            )
            
            # Verify structure
            self.assertIsNotNone(model._down_block)
            self.assertEqual(model.num_down_layers, num_down_layers)
            self.assertEqual(model.num_down_block_resnet_blocks, num_resnet_blocks)
            self.assertEqual(model.down_block_out_channels, out_channels)
            self.assertEqual(len(model._down_block.down_layers), num_down_layers)
            
            # Check the first down layer's in_channels
            self.assertEqual(model._down_block.down_layers[0].in_channels, model.input_channels)
            
            # Check the output channels of each down layer
            for i, layer in enumerate(model._down_block.down_layers):
                # the layer.out_channels is a list (modified inside the UnetDownLayer1D)
                self.assertTrue(isinstance(layer.out_channels, list))
                self.assertTrue(all((c == out_channels[i] for c in layer.out_channels)))
    
    def test_middle_block_structure(self):
        """Test the structure of the middle block after building"""
        for _ in range(20):
            model = self._create_unet_model()
            
            # Random parameters
            num_down_layers = random.randint(*self.num_down_layers_range)
            num_down_resnet_blocks = random.randint(*self.num_resnet_blocks_range)
            out_channels = [random.randint(16, 64) for _ in range(num_down_layers)]
            mid_num_resnet_blocks = random.randint(*self.num_resnet_blocks_range)
            
            # Build down block first
            model.build_down_block(
                num_down_layers=num_down_layers,
                num_resnet_blocks=num_down_resnet_blocks,
                out_channels=out_channels,
                downsample_types="conv"
            )
            
            # Build middle block
            model.build_middle_block(
                num_resnet_blocks=mid_num_resnet_blocks
            )
            
            # Verify structure
            self.assertIsNotNone(model._middle_block)
            self.assertEqual(model.middle_block_num_resnet_blocks, mid_num_resnet_blocks)
            self.assertEqual(len(model._middle_block._mid_blocks), mid_num_resnet_blocks)
            
            # Check in_channels of middle block
            self.assertEqual(model._middle_block.in_channels, out_channels[-1])
            
            # Check out_channels of middle block (should match in_channels if not specified)
            self.assertEqual(model._middle_block.out_channels, out_channels[-1])
            
            # Check each resnet block's channels
            for i, block in enumerate(model._middle_block._mid_blocks):
                self.assertEqual(block._in_channels, out_channels[-1])
                self.assertEqual(block._out_channels, out_channels[-1])
                
    def test_up_block_structure(self):
        """Test the structure of the up block after building"""
        for _ in range(20):
            model = self._create_unet_model()
            
            # Random parameters
            num_down_layers = random.randint(*self.num_down_layers_range)
            num_resnet_blocks = random.randint(*self.num_resnet_blocks_range)
            out_channels = [random.randint(16, 64) for _ in range(num_down_layers)]
            mid_num_resnet_blocks = random.randint(*self.num_resnet_blocks_range)
            
            # Build down and middle blocks first
            model.build_down_block(
                num_down_layers=num_down_layers,
                num_resnet_blocks=num_resnet_blocks,
                out_channels=out_channels,
                downsample_types="conv"
            )
            
            model.build_middle_block(
                num_resnet_blocks=mid_num_resnet_blocks
            )
            
            # Build up block
            model.build_up_block(
                num_resnet_blocks=num_resnet_blocks,
                upsample_types="transpose_conv"
            )
            
            # Verify structure
            self.assertIsNotNone(model._up_block)
            self.assertEqual(model.up_block_num_resnet_blocks, num_resnet_blocks)
            self.assertEqual(len(model._up_block.up_layers), num_down_layers)
            
            # Check the first up layer's in_channels
            self.assertEqual(model._up_block.up_layers[0].in_channels, out_channels[-1])
            
            # Check the output channels of the last up layer
            # the layer.out_channels is a list (modified inside the UnetUpLayer1D)
            self.assertTrue(isinstance(model._up_block.up_layers[-1].out_channels, list))
            self.assertTrue(all((c == model.final_out_channels for c in model._up_block.up_layers[-1].out_channels)))
            
            # The output channels should be the reversed of the input with final_out_channels at the end
            expected_up_channels = out_channels[::-1][1:] + [model.final_out_channels]
            for i, layer in enumerate(model._up_block.up_layers):
                # the layer.out_channels is a list (modified inside the UnetUpLayer1D)
                self.assertTrue(isinstance(layer.out_channels, list))
                self.assertTrue(all((c == expected_up_channels[i] for c in layer.out_channels)))
    
    ########################## Forward Pass Tests ##########################
    
    def test_forward_pass(self):
        """Test forward pass through a complete UNet model"""
        for _ in range(100):
            # Create and build a complete model
            model = self._build_complete_model()
            
            # Get valid input
            x, condition = self._get_valid_input(model)
            
            # Forward pass
            output = model(x, condition)
            
            # Check output dimensions
            self.assertEqual(output.shape[0], x.shape[0])  # Batch size preserved
            self.assertEqual(output.shape[1], model.final_out_channels)  # Output channels
            self.assertEqual(output.shape[2], x.shape[2])  # Height preserved
            self.assertEqual(output.shape[3], x.shape[3])  # Width preserved
    
    def test_invalid_input_dimensions(self):
        """Test that the model raises an error for invalid input dimensions"""
        model = self._build_complete_model()
        
        # Create input with dimensions not divisible by 2^num_down_layers
        batch_size = 2
        invalid_size = 2**model.num_down_layers - 1  # One less than a valid size
        
        x = torch.randn(batch_size, model.input_channels, invalid_size, invalid_size)
        condition = torch.randn(batch_size, model.cond_dimension)
        
        # Forward pass should raise ValueError
        with self.assertRaises(ValueError):
            model(x, condition)
    
    def test_different_input_sizes(self):
        """Test forward pass with different valid input sizes"""
        model = self._build_complete_model()
        
        for k in [1, 2, 3, 4]:
            # Generate valid input with different multiplier k
            x, condition = self._get_valid_input(model, k=k)
            
            # Forward pass
            output = model(x, condition)
            
            # Check output dimensions
            self.assertEqual(output.shape[2], x.shape[2])  # Height preserved
            self.assertEqual(output.shape[3], x.shape[3])  # Width preserved
    
    ########################## Building Order Tests ##########################
    
    def test_build_order_requirements(self):
        """Test that the model enforces the correct build order"""
        model = self._create_unet_model()
        
        # Cannot build middle block before down block
        with self.assertRaises(ValueError):
            model.build_middle_block(num_resnet_blocks=2)
        
        # Build down block
        model.build_down_block(
            num_down_layers=2,
            num_resnet_blocks=2,
            out_channels=[32, 64],
            downsample_types="conv"
        )
        
        # Now we can build middle block
        model.build_middle_block(num_resnet_blocks=2)
        
        # Cannot build up block before both down and middle blocks
        model = self._create_unet_model()
        model.build_down_block(
            num_down_layers=2,
            num_resnet_blocks=2,
            out_channels=[32, 64],
            downsample_types="conv"
        )
        
        # No middle block yet, should raise error
        with self.assertRaises(ValueError):
            model.build_up_block(num_resnet_blocks=2)
    
    def test_forward_without_build(self):
        """Test that forward raises an error if the model is not built"""
        model = self._create_unet_model()
        x, condition = torch.randn(2, model.input_channels, 32, 32), torch.randn(2, model.cond_dimension)
        
        with self.assertRaises(RuntimeError):
            model(x, condition)
    
    ########################## CustomModuleBaseTest Tests ##########################
    
    def test_eval_mode(self):
        """Test that the model can be set to evaluation mode"""
        model = self._build_complete_model()
        super()._test_eval_mode(model)
    
    def test_train_mode(self):
        """Test that the model can be set to training mode"""
        model = self._build_complete_model()
        super()._test_train_mode(model)
    
    def test_consistent_output_in_eval_mode(self):
        """Test consistent output in evaluation mode"""
        model = self._build_complete_model()
        x, condition = self._get_valid_input(model)
        super()._test_consistent_output_in_eval_mode(model, x, condition)
    
    def test_batch_size_one_in_train_mode(self):
        """Test handling of batch size 1 in training mode"""
        model = self._build_complete_model()
        x, condition = self._get_valid_input(model, batch_size=1)
        super()._test_batch_size_one_in_train_mode(model, x, condition)
    
    def test_batch_size_one_in_eval_mode(self):
        """Test handling of batch size 1 in evaluation mode"""
        model = self._build_complete_model()
        x, condition = self._get_valid_input(model, batch_size=1)
        super()._test_batch_size_one_in_eval_mode(model, x, condition)
    
    def test_named_parameters_length(self):
        """Test that named_parameters and parameters have the same length"""
        model = self._build_complete_model()
        super()._test_named_parameters_length(model)
    
    def test_to_device(self):
        """Test that the model can be moved between devices"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping device tests")
            
        model = self._build_complete_model()
        x, condition = self._get_valid_input(model)
        super()._test_to_device(model, x, condition)
    
    def test_module_is_nn_module(self):
        """Test that the module is an instance of torch.nn.Module"""
        model = self._build_complete_model()
        super()._test_module_is_nn_module(model)


if __name__ == '__main__':
    unittest.main()
