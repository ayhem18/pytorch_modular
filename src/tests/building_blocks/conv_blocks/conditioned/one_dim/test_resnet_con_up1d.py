import torch
import random
import unittest

import torch.nn as nn

from typing import Optional, Tuple

from tests.custom_base_test import CustomModuleBaseTest
from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser
from mypt.building_blocks.conv_blocks.conditioned.one_dim.resnet_con1d import UpCondOneDimWResBlock


class TestUpCondOneDimWResBlock(CustomModuleBaseTest):
    """Test class for UpCondOneDimWResBlock with spatial upsampling"""
    
    def setUp(self):
        """Initialize test parameters"""
        self.dim_analyser = DimensionsAnalyser()
        
        # Define common test parameters
        self.in_channels_range = (1, 32)
        self.out_channels_range = (1, 64)
        self.cond_dimension_range = (8, 64)
        self.dropout_options = [0.0, 0.1, 0.3]
        self.upsample_types = ["transpose_conv", "conv", "interpolate"]
        self.inner_dim_range = (16, 128)

        # For normalization and activation
        self.norm_types = ['batchnorm2d', 'groupnorm']
        self.activation_types = ['relu', 'leaky_relu', 'gelu']
    
    def _get_valid_input(self, 
                         block: Optional[UpCondOneDimWResBlock] = None, 
                         batch_size: int = 2, 
                         height: int = 16, 
                         width: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random input tensor and condition tensor with correct shapes"""
        if block is None:
            in_channels = random.randint(*self.in_channels_range)
            cond_dimension = random.randint(*self.cond_dimension_range)
        else:
            in_channels = block._in_channels
            cond_dimension = block._cond_dimension
        
        # Generate input tensor
        x = torch.randn(batch_size, in_channels, height, width)
        condition = torch.randn(batch_size, cond_dimension)
        
        return x, condition
    
    def _generate_random_upcond_block(self,
                                      in_channels=None,
                                      out_channels=None,
                                      cond_dimension=None,
                                      dropout_rate=None,
                                      upsample_type=None,
                                      inner_dim=None) -> UpCondOneDimWResBlock:
        """Generate a random UpCondOneDimWResBlock with given or random parameters"""
        # Generate random parameters if not provided
        in_channels = in_channels or random.randint(*self.in_channels_range)
        out_channels = out_channels or random.randint(*self.out_channels_range)
        cond_dimension = cond_dimension or random.randint(*self.cond_dimension_range)
        dropout_rate = dropout_rate if dropout_rate is not None else random.choice(self.dropout_options)
        upsample_type = upsample_type or random.choice(self.upsample_types)
        inner_dim = inner_dim or random.randint(*self.inner_dim_range)
        
        # Choose random normalization and activation types
        activation_type = random.choice(self.activation_types)
        film_activation = random.choice(self.activation_types)
        
        norm_type = random.choice(self.norm_types)
        norm1_params, norm2_params = {}, {}


        # Set activation parameters
        activation_params = {}
        film_activation_params = {}
        
        # Create the block
        return UpCondOneDimWResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_dimension=cond_dimension,
            inner_dim=inner_dim,
            dropout_rate=dropout_rate,
            norm1=norm_type,
            norm1_params=norm1_params,
            norm2=norm_type,
            norm2_params=norm2_params,
            activation=activation_type,
            activation_params=activation_params,
            film_activation=film_activation,
            film_activation_params=film_activation_params,
            force_residual=True,
            upsample_type=upsample_type
        )
    
    ########################## Block Structure Tests ##########################
    def test_upcond_block_structure(self):
        """Test that UpCondOneDimWResBlock is structured correctly"""
        for _ in range(100):
            # Create a block with each upsampling type
            for upsample_type in self.upsample_types:
                block = self._generate_random_upcond_block(upsample_type=upsample_type)
                
                # Check block components
                self.assertTrue(hasattr(block, '_resnet_block'), "Block should have a _resnet_block component")
                self.assertTrue(hasattr(block, '_upsample'), "Block should have a _upsample component")
                
                # Check that the upsample component is the correct type
                if upsample_type == "transpose_conv":
                    self.assertIsInstance(block._upsample, nn.ConvTranspose2d)
                    self.assertEqual(block._upsample.stride, (2, 2))
                elif upsample_type == "conv":
                    self.assertIsInstance(block._upsample, nn.Sequential)
                    self.assertIsInstance(block._upsample[0], nn.Upsample)
                    self.assertIsInstance(block._upsample[1], nn.Conv2d)
                elif upsample_type == "interpolate":
                    self.assertIsInstance(block._upsample, nn.Upsample)
                    self.assertEqual(block._upsample.scale_factor, 2)
    
    ########################## Forward Pass Tests ##########################

    def test_spatial_upsampling(self):
        """Test that spatial dimensions are doubled after forward pass"""
        for _ in range(10):
            # Test with various input shapes
            heights = [8, 16, 32, 17, 33]  # Include odd dimensions
            widths = [8, 16, 32, 15, 31]  # Include odd dimensions
            
            for height in heights:
                for width in widths:
                    # Test each upsampling type
                    for upsample_type in self.upsample_types:
                        # if upsample_type == "transpose_conv":
                        #     continue
                        block = self._generate_random_upcond_block(upsample_type=upsample_type)
                        x, condition = self._get_valid_input(block, height=height, width=width)
                        
                        # Get output
                        output = block(x, condition)
                        
                        # Check output shape - dimensions should be doubled
                        expected_height = height * 2
                        expected_width = width * 2
                        
                        self.assertEqual(output.shape[2], expected_height, 
                                        f"Output height should be {expected_height} for input height {height}")
                        self.assertEqual(output.shape[3], expected_width, 
                                        f"Output width should be {expected_width} for input width {width}")
    
    
    def test_channel_dimensions(self):
        """Test that channel dimensions are set correctly in output"""
        for _ in range(100):
            in_channels = random.randint(*self.in_channels_range)
            out_channels = random.randint(*self.out_channels_range)
            
            block = self._generate_random_upcond_block(
                in_channels=in_channels, 
                out_channels=out_channels
            )
            
            x, condition = self._get_valid_input(block)
            
            # Check input has correct channels
            self.assertEqual(x.shape[1], in_channels)
            
            # Get output and check channels
            output = block(x, condition)
            self.assertEqual(output.shape[1], out_channels)
    
    def test_conditioning_effect(self):
        """Test that different conditioning tensors produce different outputs"""
        for _ in range(100):
            block = self._generate_random_upcond_block()
            x, condition1 = self._get_valid_input(block)
            
            # Create a different condition tensor with same shape
            condition2 = torch.randn_like(condition1)
            
            # Set to eval mode
            block.eval()
            
            # Get outputs with different conditions
            output1 = block(x, condition1)
            output2 = block(x, condition2)
            
            # Outputs should be different due to different conditioning
            self.assertFalse(torch.allclose(output1, output2))
    
    ########################## Upsample Type Tests ##########################

    def test_transpose_conv_upsample(self):
        """Test the transpose convolution upsampling method"""
        for _ in range(50):
            block = self._generate_random_upcond_block(upsample_type="transpose_conv")
            
            # Verify upsampling layer structure
            self.assertIsInstance(block._upsample, nn.ConvTranspose2d)
            self.assertEqual(block._upsample.stride, (2, 2))
            self.assertEqual(block._upsample.kernel_size, (3, 3))
            self.assertEqual(block._upsample.in_channels, block._out_channels)
            self.assertEqual(block._upsample.out_channels, block._out_channels)
            
            # Test forward pass
            x, condition = self._get_valid_input(block)
            output = block(x, condition)
            
            # Check output shape
            self.assertEqual(output.shape[2], x.shape[2] * 2)
            self.assertEqual(output.shape[3], x.shape[3] * 2)
    
    def test_conv_upsample(self):
        """Test the convolution after interpolation upsampling method"""
        for _ in range(50):
            block = self._generate_random_upcond_block(upsample_type="conv")
            
            # Verify upsampling layer structure
            self.assertIsInstance(block._upsample, nn.Sequential)
            self.assertIsInstance(block._upsample[0], nn.Upsample)
            self.assertIsInstance(block._upsample[1], nn.Conv2d)
            
            # Check upsampling parameters
            self.assertEqual(block._upsample[0].scale_factor, 2)

            self.assertEqual(block._upsample[0].mode, 'nearest-exact')
            
            # Check conv parameters
            self.assertEqual(block._upsample[1].kernel_size, (3, 3))
            self.assertEqual(block._upsample[1].in_channels, block._out_channels)
            self.assertEqual(block._upsample[1].out_channels, block._out_channels)
            
            # Test forward pass
            x, condition = self._get_valid_input(block)
            output = block(x, condition)
            
            # Check output shape
            self.assertEqual(output.shape[2], x.shape[2] * 2)
            self.assertEqual(output.shape[3], x.shape[3] * 2)
    
    def test_interpolate_upsample(self):
        """Test the interpolation-only upsampling method"""
        for _ in range(50):
            block = self._generate_random_upcond_block(upsample_type="interpolate")
            
            # Verify upsampling layer structure
            self.assertIsInstance(block._upsample, nn.Upsample)
            self.assertEqual(block._upsample.scale_factor, 2)
            self.assertEqual(block._upsample.mode, 'nearest-exact')
            
            # Test forward pass
            x, condition = self._get_valid_input(block)
            output = block(x, condition)
            
            # Check output shape
            self.assertEqual(output.shape[2], x.shape[2] * 2)
            self.assertEqual(output.shape[3], x.shape[3] * 2)
    
    ########################## Edge Case Tests ##########################
    
    def test_large_input_shape(self):
        """Test with unusually large input shapes"""
        for _ in range(5):
            # Test large dimensions
            height, width = 128, 128
            
            for upsample_type in self.upsample_types:
                block = self._generate_random_upcond_block(upsample_type=upsample_type)
                x, condition = self._get_valid_input(block, height=height, width=width)
                
                # Get output
                output = block(x, condition)
                
                # Check output shape
                self.assertEqual(output.shape[2], height * 2)
                self.assertEqual(output.shape[3], width * 2)
    
    def test_small_input_shape(self):
        """Test with small input shapes"""
        for _ in range(5):
            # Test small dimensions
            heights = [1, 2, 3]
            widths = [1, 2, 3]
            
            for height in heights:
                for width in widths:
                    for upsample_type in self.upsample_types:
                        block = self._generate_random_upcond_block(upsample_type=upsample_type)
                        x, condition = self._get_valid_input(block, height=height, width=width)
                        
                        # Get output
                        output = block(x, condition)
                        
                        # Check output shape
                        self.assertEqual(output.shape[2], height * 2)
                        self.assertEqual(output.shape[3], width * 2)
    
    ########################## CustomModuleBaseTest Tests ##########################
    def test_eval_mode(self):
        """Test that the block can be set to evaluation mode"""
        for _ in range(100):
            block = self._generate_random_upcond_block()
            super()._test_eval_mode(block)
    
    # @unittest.skip("skip for now")
    def test_train_mode(self):
        """Test that the block can be set to training mode"""
        for _ in range(100):
            block = self._generate_random_upcond_block()
            super()._test_train_mode(block)
    
    # @unittest.skip("skip for now")
    def test_consistent_output_in_eval_mode(self):
        """Test consistent output in evaluation mode"""
        for _ in range(100):
            block = self._generate_random_upcond_block()
            x, condition = self._get_valid_input(block)
            super()._test_consistent_output_in_eval_mode(block, x, condition)
    
    # @unittest.skip("skip for now")
    def test_batch_size_one_in_train_mode(self):
        """Test handling of batch size 1 in training mode"""
        for _ in range(100):
            block = self._generate_random_upcond_block()
            x, condition = self._get_valid_input(block, batch_size=1)
            super()._test_batch_size_one_in_train_mode(block, x, condition)
    
    # @unittest.skip("skip for now")
    def test_batch_size_one_in_eval_mode(self):
        """Test handling of batch size 1 in evaluation mode"""
        for _ in range(100):
            block = self._generate_random_upcond_block()
            x, condition = self._get_valid_input(block, batch_size=1)
            super()._test_batch_size_one_in_eval_mode(block, x, condition)
    
    def test_named_parameters_length(self):
        """Test that named_parameters and parameters have the same length"""
        for _ in range(100):
            block = self._generate_random_upcond_block()
            super()._test_named_parameters_length(block)
    
    def test_to_device(self):
        """Test that the block can be moved between devices"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping device tests")
            
        for _ in range(100):
            block = self._generate_random_upcond_block()
            x, condition = self._get_valid_input(block)
            super()._test_to_device(block, x, condition)



if __name__ == '__main__':
    unittest.main()
