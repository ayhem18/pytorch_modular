import torch
import random
import unittest

from torch import nn
from typing import Optional, Tuple

from tests.custom_base_test import CustomModuleBaseTest
from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser
from mypt.building_blocks.conv_blocks.conditioned.one_dim.resnet_con_block import DownCondOneDimWResBlock


class TestDownCondOneDimWResBlock(CustomModuleBaseTest):
    """Test class for DownCondOneDimWResBlock with spatial downsampling"""
    
    def setUp(self):
        """Initialize test parameters"""
        self.dim_analyser = DimensionsAnalyser()
        
        # Define common test parameters
        self.in_channels_range = (1, 32)
        self.out_channels_range = (1, 64)
        self.cond_dimension_range = (8, 64)
        self.dropout_options = [0.0, 0.1, 0.3]
        self.downsample_types = ["conv", "avg_pool", "max_pool"][1:]
        self.inner_dim_range = (16, 128)
        
        # For normalization and activation
        self.norm_types = ['batchnorm2d', 'groupnorm']
        self.activation_types = ['relu', 'leaky_relu', 'gelu']
    
    def _get_valid_input(self, 
                         block: Optional[DownCondOneDimWResBlock] = None, 
                         batch_size: int = 2, 
                         height: int = 32, 
                         width: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
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
    
    def _generate_random_downcond_block(self,
                                       in_channels=None,
                                       out_channels=None,
                                       cond_dimension=None,
                                       dropout_rate=None,
                                       downsample_type=None,
                                       inner_dim=None) -> DownCondOneDimWResBlock:
        """Generate a random DownCondOneDimWResBlock with configurable parameters"""
        # Set parameters or choose random values
        if in_channels is None:
            in_channels = random.randint(*self.in_channels_range)
        
        if out_channels is None:
            out_channels = random.randint(*self.out_channels_range)
        
        if cond_dimension is None:
            cond_dimension = random.randint(*self.cond_dimension_range)
        
        if dropout_rate is None:
            dropout_rate = random.choice(self.dropout_options)
        
        if downsample_type is None:
            downsample_type = random.choice(self.downsample_types)
            
        if inner_dim is None:
            inner_dim = random.randint(*self.inner_dim_range)
        
        # Choose normalization type
        norm_type = random.choice(self.norm_types)
        if norm_type == 'groupnorm':
            norm1_params = {"num_groups": 1, "num_channels": in_channels}
            norm2_params = {"num_groups": 1, "num_channels": out_channels}
        else:
            norm1_params = {"num_features": in_channels}
            norm2_params = {"num_features": out_channels}
        
        # Choose activation type and parameters
        activation_type = random.choice(self.activation_types)
        activation_params = {'inplace': True} if activation_type == 'relu' else {}
        
        film_activation = random.choice(self.activation_types)
        film_activation_params = {'inplace': True} if film_activation == 'relu' else {}
        
        # Create the block
        return DownCondOneDimWResBlock(
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
            force_residual=True,  # Set to true to ensure the residual path is created
            downsample_type=downsample_type
        )
    
    ########################## Block Structure Tests ##########################
    
    def test_downcond_block_structure(self):
        """Test that DownCondOneDimWResBlock is structured correctly"""
        for _ in range(100):
            # Create a block with each downsampling type
            for downsample_type in self.downsample_types:
                block = self._generate_random_downcond_block(downsample_type=downsample_type)
                
                # Check block components
                self.assertTrue(hasattr(block, '_resnet_block'), "Block should have a _resnet_block component")
                self.assertTrue(hasattr(block, '_downsample'), "Block should have a _downsample component")
                
                # Check that the downsample component is the correct type
                if downsample_type == "conv":
                    self.assertIsInstance(block._downsample, nn.Conv2d)
                    self.assertEqual(block._downsample.stride, (2, 2))
                elif downsample_type in ["avg_pool", "max_pool"]:
                    self.assertIsInstance(block._downsample, nn.Sequential)
                    if downsample_type == "avg_pool":
                        self.assertIsInstance(block._downsample[0], nn.AvgPool2d)
                    else:
                        self.assertIsInstance(block._downsample[0], nn.MaxPool2d)

                    self.assertIsInstance(block._downsample[1], nn.Conv2d)
                    self.assertEqual(block._downsample[0].stride, 2)
    
    ########################## Forward Pass Tests ##########################

    # @unittest.skip("Skipping spatial downsampling tests as they are not needed")
    def test_spatial_downsampling(self):
        """Test that spatial dimensions are halved after forward pass"""
        for _ in range(10):
            # Test with various input shapes
            heights = [16, 32, 64, 33, 65]  # Include odd dimensions
            widths = [16, 32, 64, 31, 63]  # Include odd dimensions
            
            for height in heights:
                for width in widths:
                    # Test each downsampling type
                    for downsample_type in self.downsample_types:
                        block = self._generate_random_downcond_block(downsample_type=downsample_type)
                        x, condition = self._get_valid_input(block, height=height, width=width)
                        
                        # Get output
                        output = block(x, condition)
                        
                        # Check output shape - dimensions should be halved
                        expected_height = height // 2
                        expected_width = width // 2
                        
                        self.assertEqual(output.shape[2], expected_height, 
                                        f"Output height should be {expected_height} for input height {height}")
                        self.assertEqual(output.shape[3], expected_width, 
                                        f"Output width should be {expected_width} for input width {width}")
    
    # @unittest.skip("Skipping odd dimension handling tests as they are not needed")
    def test_odd_dimension_handling(self):
        """Test specific handling of odd dimensions during downsampling"""
        for _ in range(10):
            # Test with odd dimensions specifically
            heights = [9, 17, 25, 33]
            widths = [11, 21, 31, 41]
            
            for height in heights:
                for width in widths:
                    # Test each downsampling type
                    for downsample_type in self.downsample_types:
                        block = self._generate_random_downcond_block(downsample_type=downsample_type)
                        x, condition = self._get_valid_input(block, height=height, width=width)
                        
                        # Get output
                        output = block(x, condition)
                        
                        # Check output shape - dimensions should be integer division by 2
                        expected_height = height // 2
                        expected_width = width // 2
                        
                        self.assertEqual(output.shape[2], expected_height)
                        self.assertEqual(output.shape[3], expected_width)
    
    # @unittest.skip("Skipping channel dimension tests as they are not needed")
    def test_channel_dimensions(self):
        """Test that channel dimensions are set correctly in output"""
        for _ in range(100):
            in_channels = random.randint(*self.in_channels_range)
            out_channels = random.randint(*self.out_channels_range)
            
            block = self._generate_random_downcond_block(
                in_channels=in_channels, 
                out_channels=out_channels
            )
            
            x, condition = self._get_valid_input(block)
            
            # Check input has correct channels
            self.assertEqual(x.shape[1], in_channels)
            
            # Get output and check channels
            output = block(x, condition)
            self.assertEqual(output.shape[1], out_channels)
    
    # @unittest.skip("Skipping conditioning effect tests as they are not needed")
    def test_conditioning_effect(self):
        """Test that different conditioning tensors produce different outputs"""
        for _ in range(100):
            block = self._generate_random_downcond_block()
            x, condition1 = self._get_valid_input(block)
            
            # Create a different condition tensor with same shape
            if len(condition1.shape) == 2:  # 1D condition
                condition2 = torch.randn_like(condition1)
            else:  # 3D condition
                condition2 = torch.randn_like(condition1)
            
            # Set to eval mode
            block.eval()
            
            # Get outputs with different conditions
            output1 = block(x, condition1)
            output2 = block(x, condition2)
            
            # Outputs should be different due to different conditioning
            self.assertFalse(torch.allclose(output1, output2))
    
    ########################## Downsample Type Tests ##########################

    # @unittest.skip("Skipping convolutional downsampling tests as they are not needed")
    def test_conv_downsample(self):
        """Test the convolutional downsampling method"""
        for _ in range(50):
            block = self._generate_random_downcond_block(downsample_type="conv")
            
            # Verify downsampling layer structure
            self.assertIsInstance(block._downsample, nn.Conv2d)
            self.assertEqual(block._downsample.stride, (2, 2))
            self.assertEqual(block._downsample.kernel_size, (3, 3))
            self.assertEqual(block._downsample.in_channels, block._out_channels)
            self.assertEqual(block._downsample.out_channels, block._out_channels)
            
            # Test forward pass
            x, condition = self._get_valid_input(block)
            output = block(x, condition)
            
            # Check output shape
            self.assertEqual(output.shape[2], x.shape[2] // 2)
            self.assertEqual(output.shape[3], x.shape[3] // 2)
    

    # @unittest.skip("Skipping average pooling downsampling tests as they are not needed")
    def test_avg_pool_downsample(self):
        """Test the average pooling downsampling method"""
        for _ in range(50):
            block = self._generate_random_downcond_block(downsample_type="avg_pool")
            
            # Verify downsampling layer structure
            self.assertIsInstance(block._downsample, nn.Sequential)
            self.assertIsInstance(block._downsample[0], nn.AvgPool2d)
            self.assertIsInstance(block._downsample[1], nn.Conv2d)
            
            # Check pooling parameters
            self.assertEqual(block._downsample[0].kernel_size, 2)
            self.assertEqual(block._downsample[0].stride, 2)
            
            # Check 1x1 conv parameters
            self.assertEqual(block._downsample[1].kernel_size, (1, 1))
            self.assertEqual(block._downsample[1].in_channels, block._out_channels)
            self.assertEqual(block._downsample[1].out_channels, block._out_channels)
            
            # Test forward pass
            x, condition = self._get_valid_input(block)
            output = block(x, condition)
            
            # Check output shape
            self.assertEqual(output.shape[2], x.shape[2] // 2)
            self.assertEqual(output.shape[3], x.shape[3] // 2)
    
    # @unittest.skip("Skipping max pooling downsampling tests as they are not needed")
    def test_max_pool_downsample(self):
        """Test the max pooling downsampling method"""
        for _ in range(50):
            block = self._generate_random_downcond_block(downsample_type="max_pool")
            
            # Verify downsampling layer structure
            self.assertIsInstance(block._downsample, nn.Sequential)
            self.assertIsInstance(block._downsample[0], nn.MaxPool2d)
            self.assertIsInstance(block._downsample[1], nn.Conv2d)
            
            # Check pooling parameters
            self.assertEqual(block._downsample[0].kernel_size, 2)
            self.assertEqual(block._downsample[0].stride, 2)
            
            # Check 1x1 conv parameters
            self.assertEqual(block._downsample[1].kernel_size, (1, 1))
            self.assertEqual(block._downsample[1].in_channels, block._out_channels)
            self.assertEqual(block._downsample[1].out_channels, block._out_channels)
            
            # Test forward pass
            x, condition = self._get_valid_input(block)
            output = block(x, condition)
            
            # Check output shape
            self.assertEqual(output.shape[2], x.shape[2] // 2)
            self.assertEqual(output.shape[3], x.shape[3] // 2)
    
    ########################## CustomModuleBaseTest Tests ##########################

    # @unittest.skip("Skipping eval mode tests as they are not needed")
    def test_eval_mode(self):
        """Test that the block can be set to evaluation mode"""
        for _ in range(100):
            block = self._generate_random_downcond_block()
            super()._test_eval_mode(block)
    
    # @unittest.skip("Skipping train mode tests as they are not needed")
    def test_train_mode(self):
        """Test that the block can be set to training mode"""
        for _ in range(100):
            block = self._generate_random_downcond_block()
            super()._test_train_mode(block)

    # @unittest.skip("Skipping consistent output in eval mode tests as they are not needed")
    def test_consistent_output_in_eval_mode(self):
        """Test consistent output in evaluation mode"""
        for _ in range(50):
            block = self._generate_random_downcond_block()
            x, condition = self._get_valid_input(block)
            super()._test_consistent_output_in_eval_mode(block, x, condition)

    # @unittest.skip("Skipping batch size one in train mode tests as they are not needed")
    def test_batch_size_one_in_train_mode(self):
        """Test handling of batch size 1 in training mode"""
        for _ in range(50):
            block = self._generate_random_downcond_block()
            x, condition = self._get_valid_input(block, batch_size=1)
            super()._test_batch_size_one_in_train_mode(block, x, condition)

    # @unittest.skip("Skipping batch size one in eval mode tests as they are not needed")
    def test_batch_size_one_in_eval_mode(self):
        """Test handling of batch size 1 in evaluation mode"""
        for _ in range(50):
            block = self._generate_random_downcond_block()
            x, condition = self._get_valid_input(block, batch_size=1)
            super()._test_batch_size_one_in_eval_mode(block, x, condition)
    
    # @unittest.skip("Skipping named parameters length tests as they are not needed")
    def test_named_parameters_length(self):
        """Test that named_parameters and parameters have the same length"""
        for _ in range(50):
            block = self._generate_random_downcond_block()
            super()._test_named_parameters_length(block)

    # @unittest.skip("Skipping device tests as they are not needed")
    def test_to_device(self):
        """Test that the block can be moved between devices"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping device tests")
            
        for _ in range(10):  # Limit for performance
            block = self._generate_random_downcond_block()
            x, condition = self._get_valid_input(block)
            super()._test_to_device(block, x, condition)
    
    ########################## Special Property Tests ##########################

    # @unittest.skip("Skipping property accessors tests as they are not needed")
    def test_property_accessors(self):
        """Test that property accessors return correct values"""
        for _ in range(50):
            in_channels = random.randint(*self.in_channels_range)
            out_channels = random.randint(*self.out_channels_range)
            cond_dimension = random.randint(*self.cond_dimension_range)
            downsample_type = random.choice(self.downsample_types)
            
            block = self._generate_random_downcond_block(
                in_channels=in_channels,
                out_channels=out_channels,
                cond_dimension=cond_dimension,
                downsample_type=downsample_type
            )
            
            # Check properties
            self.assertEqual(block._in_channels, in_channels)
            self.assertEqual(block._out_channels, out_channels)
            self.assertEqual(block._cond_dimension, cond_dimension)
            self.assertEqual(block._downsample_type, downsample_type)


if __name__ == '__main__':
    import mypt.code_utils.pytorch_utils as pu
    pu.seed_everything(42)
    unittest.main()
