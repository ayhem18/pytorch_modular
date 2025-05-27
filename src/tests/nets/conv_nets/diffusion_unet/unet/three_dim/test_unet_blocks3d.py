import torch
import random
import unittest

import torch.nn as nn

from typing import List, Optional, Tuple, Union

from tests.custom_base_test import CustomModuleBaseTest
from mypt.nets.conv_nets.diffusion_unet.unet.three_dim.unet_blocks3d import UnetDownBlock3D, UnetUpBlock3D


# @unittest.skip("passed")
class TestUnetDownBlock3D(CustomModuleBaseTest):
    """Test class for UnetDownBlock3D that verifies downsampling behavior"""
    
    def setUp(self):
        """Initialize test parameters"""
        
        # Define common test parameters
        self.in_channels_range = (1, 32)
        self.cond_dimension_range = (4, 64)
        self.num_resnet_blocks_range = (2, 5)
        self.num_down_layers_range = (1, 4)
        self.dropout_options = [0.0, 0.1, 0.3]
        self.downsample_types = ["conv", "avg_pool", "max_pool"]
        self.inner_dim_range = (16, 128)
        
        # For normalization and activation
        self.norm_types = ['batchnorm2d', 'groupnorm']
        self.activation_types = ['relu', 'leaky_relu', 'gelu']
    
    def _get_valid_input(self, 
                        block: Optional[UnetDownBlock3D] = None, 
                        batch_size: int = 2, 
                        height: int = 32, 
                        width: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random input and condition tensors with correct shapes"""
        if block is None:
            in_channels = random.randint(*self.in_channels_range)
            cond_dimension = random.randint(*self.cond_dimension_range)
        else:
            in_channels = block.in_channels
            cond_dimension = block.cond_dimension
        
        # Generate input tensor - 3D case includes spatial dimensions for condition
        x = torch.randn(batch_size, in_channels, height, width)
        condition = torch.randn(batch_size, cond_dimension, height, width)
        
        return x, condition
    
    def _generate_random_channel_list(self, num_layers: int) -> List[int]:
        """Generate a random list of channel counts"""
        return [random.randint(16, 64) for _ in range(num_layers)]
    
    def _generate_random_down_block(self,
                                    num_down_layers=None,
                                    num_resnet_blocks=None,
                                    in_channels=None,
                                    cond_dimension=None,
                                    downsample_types=None) -> UnetDownBlock3D:
        """Generate a random UnetDownBlock3D"""
        if num_down_layers is None:
            num_down_layers = random.randint(*self.num_down_layers_range)
        
        if num_resnet_blocks is None:
            num_resnet_blocks = random.randint(*self.num_resnet_blocks_range)
        
        if in_channels is None:
            in_channels = random.randint(*self.in_channels_range)
        
        if cond_dimension is None:
            cond_dimension = random.randint(*self.cond_dimension_range)
        
        # Generate random channel counts for each layer
        out_channels = self._generate_random_channel_list(num_down_layers)
        
        # Set up downsample types
        if downsample_types is None:
            downsample_types = [random.choice(self.downsample_types) for _ in range(num_down_layers)]
        elif isinstance(downsample_types, str):
            downsample_types = [downsample_types] * num_down_layers
        
        # Choose random parameters
        dropout_rate = random.choice(self.dropout_options)
        inner_dim = random.randint(*self.inner_dim_range)
        
        # Choose norm type
        norm_type = random.choice(self.norm_types)
        norm1_params = {}
        norm2_params = {}

        if norm_type == 'batchnorm2d':
            norm1 = nn.BatchNorm2d
            norm2 = nn.BatchNorm2d
        elif norm_type == 'groupnorm':
            norm1 = nn.GroupNorm
            norm2 = nn.GroupNorm 

        # Choose activation type
        activation_type = random.choice(self.activation_types)
        activation_params = {}
        
        # Choose FiLM activation
        film_activation = random.choice(self.activation_types)
        film_activation_params = {}
        
        # Create the block
        return UnetDownBlock3D(
            num_down_layers=num_down_layers,
            num_resnet_blocks=num_resnet_blocks,
            in_channels=in_channels,
            cond_dimension=cond_dimension,
            out_channels=out_channels,
            downsample_types=downsample_types,
            inner_dim=inner_dim,
            dropout_rate=dropout_rate,
            norm1=norm1,
            norm1_params=norm1_params,
            norm2=norm2,
            norm2_params=norm2_params,
            activation=activation_type,
            activation_params=activation_params,
            film_activation=film_activation,
            film_activation_params=film_activation_params,
            force_residual=True  # Set to true to ensure the residual path is created
        )
    
    def _verify_down_dimensions(self, input_shape, output_shape, num_down_layers):
        """Verify that dimensions are properly halved for each down layer"""
        batch_size, _, in_height, in_width = input_shape
        _, _, out_height, out_width = output_shape
        
        # Calculate expected output dimensions
        expected_height = in_height // (2 ** num_down_layers)
        expected_width = in_width // (2 ** num_down_layers)
        
        self.assertEqual(out_height, expected_height, 
                         f"Expected output height {expected_height}, got {out_height}")
        self.assertEqual(out_width, expected_width, 
                         f"Expected output width {expected_width}, got {out_width}")
    
    ########################## Basic Tests ##########################
    # @unittest.skip("passed")
    def test_initialization(self):
        """Test that UnetDownBlock3D initializes correctly with various parameters"""
        for _ in range(100):
            num_down_layers = random.randint(*self.num_down_layers_range)
            num_blocks = random.randint(*self.num_resnet_blocks_range)
            
            block = self._generate_random_down_block(
                num_down_layers=num_down_layers,
                num_resnet_blocks=num_blocks
            )
            
            # Check that the block has the correct number of layers
            self.assertEqual(len(block.down_layers), num_down_layers)
            
            # Check that each layer has correct number of resnet blocks
            for layer in block.down_layers:
                self.assertEqual(len(layer._resnet_blocks), num_blocks)

    # @unittest.skip("passed")    
    def test_forward_pass(self):
        """Test basic forward pass functionality"""
        for _ in range(100):
            block = self._generate_random_down_block()
            
            # Create input with dimensions that will work with the number of down layers
            num_down_layers = len(block.down_layers)
            k = random.randint(1, 4)
            height = width = 2 ** num_down_layers * k
            
            x, condition = self._get_valid_input(block, height=height, width=width)
            
            # Forward pass
            output, skip_outputs = block(x, condition)
            
            # Check output is a tensor
            self.assertIsInstance(output, torch.Tensor)
            
            # Check skip connections list length
            self.assertEqual(len(skip_outputs), num_down_layers)
            
            # Verify output dimensions
            self._verify_down_dimensions(x.shape, output.shape, num_down_layers)
    
    ########################## Dimension Tests ##########################
    # @unittest.skip("passed")
    def test_dimension_reduction(self):
        """Test that dimensions are halved correctly through the network"""
        for _ in range(50):
            # Test with 1-4 down layers
            num_down_layers = random.randint(1, 4)
            block = self._generate_random_down_block(num_down_layers=num_down_layers)
            
            # Create input where dimensions are divisible by 2^num_down_layers
            k = random.randint(3, 8)
            height = width = 2 ** num_down_layers * k
            
            x, condition = self._get_valid_input(block, height=height, width=width)
            
            # Forward pass
            output, skip_outputs = block(x, condition)
            
            # Output dimensions should be k
            self.assertEqual(output.shape[2], k)
            self.assertEqual(output.shape[3], k)
            
            # Check skip connection dimensions
            for i, skip in enumerate(skip_outputs):
                expected_height = height // (2 ** (i + 1))
                expected_width = width // (2 ** (i + 1))
                
                self.assertEqual(skip.shape[2], expected_height)
                self.assertEqual(skip.shape[3], expected_width)
    
    # @unittest.skip("passed")
    def test_different_downsample_types(self):
        """Test with different downsample types"""
        for _ in range(50):
            num_down_layers = random.randint(2, 4)
            
            # Try all combinations of downsample types
            for downsample_type in self.downsample_types:
                downsample_types = [downsample_type] * num_down_layers
                
                block = self._generate_random_down_block(
                    num_down_layers=num_down_layers,
                    downsample_types=downsample_types
                )
                
                # Create input of suitable size
                k = random.randint(1, 8)
                height = (2 ** num_down_layers) * k
                width = (2 ** num_down_layers) * k
                
                x, condition = self._get_valid_input(block, height=height, width=width)
                
                # Forward pass
                output, skip_outputs = block(x, condition)
                
                # Verify output dimensions
                self._verify_down_dimensions(x.shape, output.shape, num_down_layers)
    
    ########################## Edge Case Tests ##########################
    # @unittest.skip("passed")
    def test_minimum_input_size(self):
        """Test with the minimum possible input size"""
        for _ in range(50):
            num_down_layers = random.randint(1, 3)
            block = self._generate_random_down_block(num_down_layers=num_down_layers)
            
            # Minimum input size is 2^num_down_layers
            height = width = 2 ** num_down_layers
            
            x, condition = self._get_valid_input(block, height=height, width=width)
            
            # Forward pass
            output, skip_outputs = block(x, condition)
            
            # Verify output dimensions
            self.assertEqual(output.shape[2], 1, "Minimum height should be 1")
            self.assertEqual(output.shape[3], 1, "Minimum width should be 1")
    
    # @unittest.skip("passed")
    def test_different_layer_counts(self):
        """Test with different numbers of down layers"""
        for num_down_layers in range(1, 5):
            block = self._generate_random_down_block(num_down_layers=num_down_layers)
            
            # Verify layer count is correct
            self.assertEqual(len(block.down_layers), num_down_layers)
            
            # Create valid input
            k = random.randint(1, 4)
            height = width = 2 ** num_down_layers * k
            
            x, condition = self._get_valid_input(block, height=height, width=width)
            
            # Forward pass
            output, skip_outputs = block(x, condition)
            
            # Verify output dimensions
            self.assertEqual(output.shape[2], k)
            self.assertEqual(output.shape[3], k)
            
            # Verify skip connection count
            self.assertEqual(len(skip_outputs), num_down_layers)
    
    ########################## CustomModuleBaseTest Tests ##########################
    # @unittest.skip("skip for now")
    def test_eval_mode(self):
        """Test that the block can be set to evaluation mode"""
        for _ in range(50):
            block = self._generate_random_down_block()
            super()._test_eval_mode(block)

    # @unittest.skip("skip for now")
    def test_train_mode(self):
        """Test that the block can be set to training mode"""
        for _ in range(50):
            block = self._generate_random_down_block()
            super()._test_train_mode(block)

    # @unittest.skip("skip for now")
    def test_consistent_output_in_eval_mode(self):
        """Test consistent output in evaluation mode"""
        for _ in range(20):
            block = self._generate_random_down_block()
            
            # Create valid input
            num_down_layers = len(block.down_layers)
            k = random.randint(1, 4)
            height = width = 2 ** num_down_layers * k
            
            x, condition = self._get_valid_input(block, height=height, width=width)
            super()._test_consistent_output_in_eval_mode(block, x, condition)

    # @unittest.skip("skip for now")  
    def test_batch_size_one(self):
        """Test handling of batch size 1"""
        for _ in range(50):
            block = self._generate_random_down_block()
            
            # Create valid input
            num_down_layers = len(block.down_layers)
            k = random.randint(1, 4)
            height = width = 2 ** num_down_layers * k
            
            x, condition = self._get_valid_input(block, batch_size=1, height=height, width=width)
            
            # Forward pass should work without errors
            output, _ = block(x, condition)
            
            # Check batch dimension is preserved
            self.assertEqual(output.shape[0], 1)

    def test_named_parameters_length(self):
        """Test that named_parameters and parameters have the same length"""
        for _ in range(20):
            block = self._generate_random_down_block()
            super()._test_named_parameters_length(block)
    
    def test_to_device(self):
        """Test that the block can be moved between devices"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping device tests")
            
        for _ in range(50): 
            block = self._generate_random_down_block()
            
            # Create valid input
            num_down_layers = len(block.down_layers)
            k = random.randint(1, 4)
            height = width = 2 ** num_down_layers * k
            
            x, condition = self._get_valid_input(block, height=height, width=width)
            super()._test_to_device(block, x, condition)




# @unittest.skip("Skipping UnetUpBlock3D tests")
class TestUnetUpBlock3D(CustomModuleBaseTest):
    """Test class for UnetUpBlock3D that verifies upsampling behavior"""
    
    def setUp(self):
        """Initialize test parameters"""
        
        # Define common test parameters
        self.in_channels_range = (1, 32)
        self.cond_dimension_range = (4, 64)
        self.num_resnet_blocks_range = (2, 5)
        self.num_up_layers_range = (1, 4)
        self.dropout_options = [0.0, 0.1, 0.3]
        self.upsample_types = ["transpose_conv", "conv", "interpolate"]
        self.inner_dim_range = (16, 128)
        
        # For normalization and activation
        self.norm_types = ['batchnorm2d', 'groupnorm']
        self.activation_types = ['relu', 'leaky_relu', 'gelu']
    
    def _get_valid_input_and_skip(self, 
                                  block: Optional[UnetUpBlock3D] = None,
                                  base_size: int = 1,
                                  batch_size: int = 2) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Generate random input, skip outputs, and condition tensors with correct shapes"""
        in_channels = block.in_channels
        cond_dimension = block.cond_dimension
        num_up_layers = len(block.up_layers)
        out_channels = block.out_channels
        
        # Initial input size
        height = width = base_size
        
        # Input tensor
        x = torch.randn(batch_size, in_channels, height, width)
        
        # Condition tensor with spatial dimensions
        condition = torch.randn(batch_size, cond_dimension, height, width)
        
        # Skip connection tensors - dimensions double at each step
        # Generate skip connection tensors
        skip_outputs = [None for _ in range(num_up_layers)]
        
        for i in range(num_up_layers):
            # Each skip connection has progressively larger spatial dimensions
            skip_height = skip_width = height * (2 ** i)
            # out_channels is a list of lists
            # skip_channels is either in_channels
            # or the input to the i-th up_layer: which is the output of the (i-1)-th up_layer 
            # which can be found as the last element in the out_channels[i - 1] list
            skip_channels = out_channels[i - 1][-1] if i > 0 else in_channels
            skip = torch.randn(batch_size, skip_channels, skip_height, skip_width)
            skip_outputs[i] = skip
        
        # Reverse skip outputs list to match expected ordering
        skip_outputs = list(reversed(skip_outputs))
        
        return x, skip_outputs, condition
    
    def _generate_random_channel_list(self, num_layers: int) -> List[int]:
        """Generate a random list of channel counts"""
        return [random.randint(16, 64) for _ in range(num_layers)]
    
    def _generate_random_up_block(self,
                                 num_up_layers=None,
                                 num_resnet_blocks=None,
                                 in_channels=None,
                                 cond_dimension=None,
                                 upsample_types=None) -> UnetUpBlock3D:
        """Generate a random UnetUpBlock3D"""
        if num_up_layers is None:
            num_up_layers = random.randint(*self.num_up_layers_range)
        
        if num_resnet_blocks is None:
            num_resnet_blocks = random.randint(*self.num_resnet_blocks_range)
        
        if in_channels is None:
            in_channels = random.randint(*self.in_channels_range)
        
        if cond_dimension is None:
            cond_dimension = random.randint(*self.cond_dimension_range)
        
        # Generate random channel counts for each layer
        out_channels = self._generate_random_channel_list(num_up_layers)
        
        # Set up upsample types
        if upsample_types is None:
            upsample_types = [random.choice(self.upsample_types) for _ in range(num_up_layers)]
        elif isinstance(upsample_types, str):
            upsample_types = [upsample_types] * num_up_layers
        
        # Choose random parameters
        dropout_rate = random.choice(self.dropout_options)
        inner_dim = random.randint(*self.inner_dim_range)
        
        # Choose norm type
        norm_type = random.choice(self.norm_types)
        if norm_type == 'batchnorm2d':
            norm1 = nn.BatchNorm2d
            norm2 = nn.BatchNorm2d
        elif norm_type == 'groupnorm':
            norm1 = nn.GroupNorm
            norm2 = nn.GroupNorm
        
        norm1_params = {}
        norm2_params = {}

        # Choose activation type
        activation_type = random.choice(self.activation_types)
        activation_params = {}
        
        # Choose FiLM activation
        film_activation = random.choice(self.activation_types)
        film_activation_params = {}
        
        # Create the block
        return UnetUpBlock3D(
            num_up_layers=num_up_layers,
            num_resnet_blocks=num_resnet_blocks,
            in_channels=in_channels,
            cond_dimension=cond_dimension,
            out_channels=out_channels,
            upsample_types=upsample_types,
            inner_dim=inner_dim,
            dropout_rate=dropout_rate,
            norm1=norm1,
            norm1_params=norm1_params,
            norm2=norm2,
            norm2_params=norm2_params,
            activation=activation_type,
            activation_params=activation_params,
            film_activation=film_activation,
            film_activation_params=film_activation_params,
            force_residual=True  # Set to true to ensure the residual path is created
        )
    
    def _verify_up_dimensions(self, input_shape, output_shape, num_up_layers):
        """Verify that dimensions are properly doubled for each up layer"""
        batch_size, _, in_height, in_width = input_shape
        _, _, out_height, out_width = output_shape
        
        # Calculate expected output dimensions
        expected_height = in_height * (2 ** num_up_layers)
        expected_width = in_width * (2 ** num_up_layers)
        
        self.assertEqual(out_height, expected_height, 
                         f"Expected output height {expected_height}, got {out_height}")
        self.assertEqual(out_width, expected_width, 
                         f"Expected output width {expected_width}, got {out_width}")
    
    ########################## Basic Tests ##########################
    def test_initialization(self):
        """Test that UnetUpBlock3D initializes correctly with various parameters"""
        for _ in range(100):
            num_up_layers = random.randint(*self.num_up_layers_range)
            num_blocks = random.randint(*self.num_resnet_blocks_range)
            
            block = self._generate_random_up_block(
                num_up_layers=num_up_layers,
                num_resnet_blocks=num_blocks
            )
            
            # Check that the block has the correct number of layers
            self.assertEqual(len(block.up_layers), num_up_layers)
            
            # Check that each layer has correct number of resnet blocks
            for layer in block.up_layers:
                self.assertEqual(len(layer._resnet_blocks), num_blocks)

    def test_forward_pass(self):
        """Test basic forward pass functionality"""
        for _ in range(100):
            block = self._generate_random_up_block()
            
            # Create valid inputs
            base_size = random.randint(1, 4)
            x, skip_outputs, condition = self._get_valid_input_and_skip(block, base_size=base_size)
            
            # Forward pass
            output = block(x, skip_outputs, condition)
            
            # Check output is a tensor
            self.assertIsInstance(output, torch.Tensor)
            
            # Verify output dimensions
            num_up_layers = len(block.up_layers)
            self._verify_up_dimensions(x.shape, output.shape, num_up_layers)
    
    ########################## Dimension Tests ##########################
    def test_dimension_doubling(self):
        """Test that dimensions are doubled correctly through the network"""
        for _ in range(50):
            # Test with 1-4 up layers
            num_up_layers = random.randint(1, 4)
            block = self._generate_random_up_block(num_up_layers=num_up_layers)
            
            # Create input with base size k
            k = random.randint(1, 4)
            x, skip_outputs, condition = self._get_valid_input_and_skip(block, base_size=k)
            
            # Forward pass
            output = block(x, skip_outputs, condition)
            
            # Output dimensions should be k * 2^num_up_layers
            expected_size = k * (2 ** num_up_layers)
            self.assertEqual(output.shape[2], expected_size)
            self.assertEqual(output.shape[3], expected_size)
    
    def test_different_upsample_types(self):
        """Test with different upsample types"""
        for _ in range(50):
            num_up_layers = random.randint(2, 4)
            
            # Try each upsample type
            for upsample_type in self.upsample_types:
                upsample_types = [upsample_type] * num_up_layers
                
                block = self._generate_random_up_block(
                    num_up_layers=num_up_layers,
                    upsample_types=upsample_types
                )
                
                # Create valid input
                base_size = random.randint(1, 4)
                x, skip_outputs, condition = self._get_valid_input_and_skip(block, base_size=base_size)
                
                # Forward pass
                output = block(x, skip_outputs, condition)
                
                # Verify output dimensions
                self._verify_up_dimensions(x.shape, output.shape, num_up_layers)
    
    ########################## Edge Case Tests ##########################
    def test_single_pixel_input(self):
        """Test with single-pixel input"""
        for _ in range(50):
            num_up_layers = random.randint(1, 4)
            block = self._generate_random_up_block(num_up_layers=num_up_layers)
            
            # Input is a single pixel
            x, skip_outputs, condition = self._get_valid_input_and_skip(block, base_size=1)
            
            # Forward pass
            output = block(x, skip_outputs, condition)
            
            # Output should be 2^num_up_layers x 2^num_up_layers
            expected_size = 2 ** num_up_layers
            self.assertEqual(output.shape[2], expected_size)
            self.assertEqual(output.shape[3], expected_size)

    def test_different_layer_counts(self):
        """Test with different numbers of up layers"""
        for num_up_layers in range(1, 5):
            block = self._generate_random_up_block(num_up_layers=num_up_layers)
            
            # Verify layer count is correct
            self.assertEqual(len(block.up_layers), num_up_layers)
            
            # Create valid input
            base_size = random.randint(1, 4)
            x, skip_outputs, condition = self._get_valid_input_and_skip(block, base_size=base_size)
            
            # Forward pass
            output = block(x, skip_outputs, condition)
            
            # Verify output dimensions
            expected_size = base_size * (2 ** num_up_layers)
            self.assertEqual(output.shape[2], expected_size)
            self.assertEqual(output.shape[3], expected_size)

    def test_skip_connection_integration(self):
        """Test that skip connections are correctly integrated"""
        for _ in range(50):
            num_up_layers = random.randint(2, 4)
            block = self._generate_random_up_block(num_up_layers=num_up_layers)
            
            # Create input with base size 1
            x, skip_outputs, condition = self._get_valid_input_and_skip(block, base_size=1)
            
            # Forward pass
            output = block(x, skip_outputs, condition)
            
            # The output should have the final expected dimensions
            expected_size = 2 ** num_up_layers
            self.assertEqual(output.shape[2], expected_size)
            self.assertEqual(output.shape[3], expected_size)
            
            # Check that skip connections are correctly integrated by modifying them
            # and verifying the output changes
            modified_skips = [s.clone() * 2 for s in skip_outputs]
            modified_output = block(x, modified_skips, condition)
            
            # The outputs should be different
            self.assertFalse(torch.allclose(output, modified_output))
    
    ########################## CustomModuleBaseTest Tests ##########################
    def test_eval_mode(self):
        """Test that the block can be set to evaluation mode"""
        for _ in range(50):
            block = self._generate_random_up_block()
            super()._test_eval_mode(block)

    def test_train_mode(self):
        """Test that the block can be set to training mode"""
        for _ in range(50):
            block = self._generate_random_up_block()
            super()._test_train_mode(block)

    def test_consistent_output_in_eval_mode(self):
        """Test consistent output in evaluation mode"""
        for _ in range(50):
            block = self._generate_random_up_block()
            base_size = random.randint(1, 4)
            
            x, skip_outputs, condition = self._get_valid_input_and_skip(
                block, base_size=base_size
            )
            super()._test_consistent_output_in_eval_mode(block, x, skip_outputs, condition, reverse_skip_connections=True)

    def test_batch_size_one_in_train_mode(self):
        """Test handling of batch size 1 in training mode"""
        for _ in range(50):
            block = self._generate_random_up_block()
            base_size = random.randint(1, 4)
            
            x, skip_outputs, condition = self._get_valid_input_and_skip(
                block, base_size=base_size, batch_size=1
            )   
            super()._test_batch_size_one_in_train_mode(block, x, skip_outputs, condition, reverse_skip_connections=True)

    def test_batch_size_one_in_eval_mode(self):
        """Test handling of batch size 1 in evaluation mode"""
        for _ in range(50):
            block = self._generate_random_up_block()
            base_size = random.randint(1, 4)
            
            x, skip_outputs, condition = self._get_valid_input_and_skip(
                block, base_size=base_size, batch_size=1
            )   
            super()._test_batch_size_one_in_eval_mode(block, x, skip_outputs, condition, reverse_skip_connections=True)

    def test_named_parameters_length(self):
        """Test that named_parameters and parameters have the same length"""
        for _ in range(50):
            block = self._generate_random_up_block()
            super()._test_named_parameters_length(block)

    def test_to_device(self):
        """Test that the block can be moved between devices"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping device tests")
            
        for _ in range(10):  # Limit for performance
            block = self._generate_random_up_block()
            base_size = random.randint(1, 4)
            
            x, skip_outputs, condition = self._get_valid_input_and_skip(
                block, base_size=base_size
            )
            super()._test_to_device(block, x, skip_outputs, condition, reverse_skip_connections=True)

    def test_module_is_nn_module(self):
        """Test that the module is an instance of torch.nn.Module"""
        block = self._generate_random_up_block()
        super()._test_module_is_nn_module(block)


if __name__ == '__main__':
    from mypt.code_utils import pytorch_utils as pu
    pu.seed_everything(42)
    unittest.main()
