import torch
import random
import unittest

from typing import List, Optional, Tuple, Union

from tests.custom_base_test import CustomModuleBaseTest
from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser
from mypt.nets.conv_nets.diffusion_unet.unet.three_dim.unet_layer3d import UnetDownLayer3D, UnetUpLayer3D

class TestUnetDownLayer3D(CustomModuleBaseTest):
    """Test class for UnetDownLayer3D that verifies downsampling behavior"""
    
    def setUp(self):
        """Initialize test parameters"""
        self.dim_analyser = DimensionsAnalyser()
        
        # Define common test parameters
        self.in_channels_range = (1, 32)
        self.cond_dimension_range = (4, 64)
        self.num_resnet_blocks_range = (2, 5)
        self.dropout_options = [0.0, 0.1, 0.3]
        self.downsample_types = ["conv", "avg_pool", "max_pool"]
        self.inner_dim_range = (16, 128)
        
        # For normalization and activation
        self.norm_types = ['batchnorm2d', 'groupnorm']
        self.activation_types = ['relu', 'leaky_relu', 'gelu']
    
    def _get_valid_input(self, 
                        layer: Optional[UnetDownLayer3D] = None, 
                        batch_size: int = 2, 
                        height: int = 32, 
                        width: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random input and condition tensors with correct shapes"""
        if layer is None:
            in_channels = random.randint(*self.in_channels_range)
            cond_dimension = random.randint(*self.cond_dimension_range)
        else:
            in_channels = layer.in_channels
            cond_dimension = layer.cond_dimension
        
        # Generate input tensor
        x = torch.randn(batch_size, in_channels, height, width)
        
        # Generate 3D condition tensor (with spatial dimensions)
        condition = torch.randn(batch_size, cond_dimension, height, width)
        
        return x, condition
    
    def _generate_random_channel_list(self, num_blocks: int) -> List[int]:
        """Generate a random list of channel counts for the resnet blocks"""
        return [random.randint(16, 64) for _ in range(num_blocks)]
    
    def _generate_random_down_layer(self,
                                   num_resnet_blocks=None,
                                   in_channels=None,
                                   cond_dimension=None,
                                   out_channels=None,
                                   dropout_rate=None,
                                   downsample_type=None,
                                   inner_dim=None) -> UnetDownLayer3D:
        """Generate a random UnetDownLayer3D with given or random parameters"""
        # Set random parameters if not provided
        if num_resnet_blocks is None:
            num_resnet_blocks = random.randint(*self.num_resnet_blocks_range)
        
        if in_channels is None:
            in_channels = random.randint(*self.in_channels_range)
        
        if cond_dimension is None:
            cond_dimension = random.randint(*self.cond_dimension_range)
        
        if out_channels is None:
            out_channels = self._generate_random_channel_list(num_resnet_blocks)
        
        if dropout_rate is None:
            dropout_rate = random.choice(self.dropout_options)
        
        if downsample_type is None:
            downsample_type = random.choice(self.downsample_types)
        
        if inner_dim is None:
            inner_dim = random.randint(*self.inner_dim_range)

        return UnetDownLayer3D(
            num_resnet_blocks=num_resnet_blocks,
            in_channels=in_channels,
            cond_dimension=cond_dimension,
            out_channels=out_channels,
            inner_dim=inner_dim,
            dropout_rate=dropout_rate,
            downsample_type=downsample_type
        )
    
    ########################## Basic Tests ##########################
    
    def test_initialization(self):
        """Test that UnetDownLayer3D can be initialized with various parameters"""
        for _ in range(100):
            # Test initialization with random parameters
            layer = self._generate_random_down_layer()
            
            # Check that the layer has the expected attributes
            self.assertEqual(layer.in_channels, layer._resnet_blocks[0]._in_channels)
            self.assertEqual(layer.out_channels[-1], layer._resnet_blocks[-1]._out_channels)
            
            # Check that the last block is a downsampling block
            self.assertTrue(hasattr(layer._resnet_blocks[-1], '_downsample'))
    
    # @unittest.skip("skip for now")
    def test_forward_pass(self):
        """Test forward pass through UnetDownLayer3D"""
        for _ in range(100):
            # Create layer with random parameters
            layer = self._generate_random_down_layer()
            
            # Generate input and condition tensors
            x, condition = self._get_valid_input(layer)
            
            # Forward pass
            output = layer(x, condition)
            
            # Check output shape - should be halved in spatial dimensions
            self.assertEqual(output.shape[0], x.shape[0])  # Batch size unchanged
            self.assertEqual(output.shape[1], layer.out_channels[-1])  # Channels match
            self.assertEqual(output.shape[2], x.shape[2] // 2)  # Height halved
            self.assertEqual(output.shape[3], x.shape[3] // 2)  # Width halved
    
    ########################## Spatial Dimension Tests ##########################
    
    def _verify_correct_output_shape(self, down_sample_type: str, in_height: int, in_width: int, out_height: int, out_width: int) -> None:
        """Verify that the output shape is correct"""
        if down_sample_type == "conv":
            self.assertEqual(out_height, (in_height + 1) // 2)
            self.assertEqual(out_width, (in_width + 1) // 2)
        else:
            self.assertEqual(out_height, in_height // 2)
            self.assertEqual(out_width, in_width // 2)

    # @unittest.skip("skip for now")
    def test_exact_halving(self):
        """Test that spatial dimensions are exactly halved for all types of downsampling"""
        for downsample_type in self.downsample_types:
            # Create layer with specific downsampling type
            layer = self._generate_random_down_layer(downsample_type=downsample_type)
            
            # Test different spatial dimensions
            for height in [8, 16, 32, 64, 15, 31, 63]:
                for width in [8, 16, 32, 64, 17, 33, 65]:
                    x, condition = self._get_valid_input(layer, height=height, width=width)
                    
                    # Forward pass
                    output = layer(x, condition)
                    
                    # Check output dimensions - should be exactly halved
                    self._verify_correct_output_shape(downsample_type, height, width, output.shape[2], output.shape[3])

    # @unittest.skip("skip for now")
    def test_odd_dimensions(self):
        """Test behavior with odd spatial dimensions"""
        for _ in range(50):
            layer = self._generate_random_down_layer()
            
            # Test with odd dimensions
            for height, width in [(15, 15), (31, 31), (63, 63), (17, 33)]:
                x, condition = self._get_valid_input(layer, height=height, width=width)
                
                # Forward pass
                output = layer(x, condition)
                
                # Check output dimensions - should be exactly halved (integer division)
                self._verify_correct_output_shape(layer.downsample_type, height, width, output.shape[2], output.shape[3])
    
    ########################## Edge Case Tests ##########################
    
    # @unittest.skip("skip for now")
    def test_small_dimensions(self):
        """Test downsampling from very small dimensions"""
        for _ in range(50):
            layer = self._generate_random_down_layer()
            
            # Test with small dimensions
            for height, width in [(2, 2), (4, 4), (6, 6), (4, 6)]:
                x, condition = self._get_valid_input(layer, height=height, width=width)
                
                # Forward pass
                output = layer(x, condition)
                
                # Check output dimensions - should be exactly halved
                self._verify_correct_output_shape(layer.downsample_type, height, width, output.shape[2], output.shape[3])
    
    # @unittest.skip("skip for now")
    def test_num_blocks(self):
        """Test with varying numbers of resnet blocks"""
        for num_blocks in range(2, 6):  # Test 2 to 5 blocks
            layer = self._generate_random_down_layer(num_resnet_blocks=num_blocks)
            
            # Verify correct number of blocks
            self.assertEqual(len(layer._resnet_blocks), num_blocks)
            
            # Check that only the last block does downsampling
            for i in range(num_blocks - 1):
                self.assertFalse(hasattr(layer._resnet_blocks[i], '_downsample'))
            
            self.assertTrue(hasattr(layer._resnet_blocks[-1], '_downsample'))
            
            # Test forward pass
            x, condition = self._get_valid_input(layer)
            output = layer(x, condition)
            
            # Check output dimensions
            self._verify_correct_output_shape(layer.downsample_type, x.shape[2], x.shape[3], output.shape[2], output.shape[3])
    
    ########################## CustomModuleBaseTest Tests ##########################
    
    def test_eval_mode(self):
        """Test that the block can be set to evaluation mode"""
        for _ in range(100):
            block = self._generate_random_down_layer()
            super()._test_eval_mode(block)
    
    def test_train_mode(self):
        """Test that the block can be set to training mode"""
        for _ in range(100):
            block = self._generate_random_down_layer()
            super()._test_train_mode(block)

    def test_consistent_output_in_eval_mode(self):
        """Test consistent output in evaluation mode"""
        for _ in range(50): 
            block = self._generate_random_down_layer()
            x, condition = self._get_valid_input(block)
            super()._test_consistent_output_in_eval_mode(block, x, condition)

    def test_batch_size_one_in_train_mode(self):
        """Test handling of batch size 1 in training mode"""
        for _ in range(50):
            block = self._generate_random_down_layer()
            x, condition = self._get_valid_input(block, batch_size=1)
            super()._test_batch_size_one_in_train_mode(block, x, condition)

    def test_batch_size_one_in_eval_mode(self):
        """Test handling of batch size 1 in evaluation mode"""
        for _ in range(50):
            block = self._generate_random_down_layer()
            x, condition = self._get_valid_input(block, batch_size=1)
            super()._test_batch_size_one_in_eval_mode(block, x, condition)
    
    def test_named_parameters_length(self):
        """Test that named_parameters and parameters have the same length"""
        for _ in range(50):
            block = self._generate_random_down_layer()
            super()._test_named_parameters_length(block)

    def test_to_device(self):
        """Test that the block can be moved between devices"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping device tests")
            
        for _ in range(100): 
            block = self._generate_random_down_layer()
            x, condition = self._get_valid_input(block)
            super()._test_to_device(block, x, condition)

    def test_module_is_nn_module(self):
        """Test that the module is an instance of torch.nn.Module"""
        block = self._generate_random_down_layer()
        super()._test_module_is_nn_module(block)




# @unittest.skip("Skipping UnetUpLayer3D tests")
class TestUnetUpLayer3D(CustomModuleBaseTest):
    """Test class for UnetUpLayer3D that verifies upsampling behavior"""
    
    def setUp(self):
        """Initialize test parameters"""
        self.dim_analyser = DimensionsAnalyser()
        
        # Define common test parameters
        self.in_channels_range = (1, 32)
        self.cond_dimension_range = (4, 64)
        self.num_resnet_blocks_range = (2, 5)
        self.dropout_options = [0.0, 0.1, 0.3]
        self.upsample_types = ["transpose_conv", "conv", "interpolate"]
        self.inner_dim_range = (16, 128)
        
        # For normalization and activation
        self.norm_types = ['batchnorm2d', 'groupnorm']
        self.activation_types = ['relu', 'leaky_relu', 'gelu']
    
    def _get_valid_input(self, 
                        layer: Optional[UnetUpLayer3D] = None, 
                        batch_size: int = 2, 
                        height: int = 16, 
                        width: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random input and condition tensors with correct shapes"""
        if layer is None:
            in_channels = random.randint(*self.in_channels_range)
            cond_dimension = random.randint(*self.cond_dimension_range)
        else:
            in_channels = layer.in_channels
            cond_dimension = layer.cond_dimension
        
        # Generate input tensor
        x = torch.randn(batch_size, in_channels, height, width)
        
        # Generate 3D condition tensor (with spatial dimensions)
        condition = torch.randn(batch_size, cond_dimension, height, width)
        
        return x, condition
    
    def _generate_random_channel_list(self, num_blocks: int) -> List[int]:
        """Generate a random list of channel counts for the resnet blocks"""
        return [random.randint(16, 64) for _ in range(num_blocks)]
    
    def _generate_random_up_layer(self,
                                 num_resnet_blocks=None,
                                 in_channels=None,
                                 cond_dimension=None,
                                 out_channels=None,
                                 dropout_rate=None,
                                 upsample_type=None,
                                 inner_dim=None) -> UnetUpLayer3D:
        """Generate a random UnetUpLayer3D with given or random parameters"""
        # Set random parameters if not provided
        if num_resnet_blocks is None:
            num_resnet_blocks = random.randint(*self.num_resnet_blocks_range)
        
        if in_channels is None:
            in_channels = random.randint(*self.in_channels_range)
        
        if cond_dimension is None:
            cond_dimension = random.randint(*self.cond_dimension_range)
        
        if out_channels is None:
            out_channels = self._generate_random_channel_list(num_resnet_blocks)
        
        if dropout_rate is None:
            dropout_rate = random.choice(self.dropout_options)
        
        if upsample_type is None:
            upsample_type = random.choice(self.upsample_types)
        
        if inner_dim is None:
            inner_dim = random.randint(*self.inner_dim_range)
        
        # Create layer
        return UnetUpLayer3D(
            num_resnet_blocks=num_resnet_blocks,
            in_channels=in_channels,
            cond_dimension=cond_dimension,
            out_channels=out_channels,
            inner_dim=inner_dim,
            dropout_rate=dropout_rate,
            upsample_type=upsample_type
        )
    
    ########################## Basic Tests ##########################
    
    def test_initialization(self):
        """Test that UnetUpLayer3D can be initialized with various parameters"""
        for _ in range(100):
            # Test initialization with random parameters
            layer = self._generate_random_up_layer()
            
            # Check that the layer has the expected attributes
            self.assertEqual(layer.in_channels, layer._resnet_blocks[0]._in_channels)
            self.assertEqual(layer.out_channels[-1], layer._resnet_blocks[-1]._out_channels)
            
            # Check that the last block is an upsampling block
            self.assertTrue(hasattr(layer._resnet_blocks[-1], '_upsample'))
    
    def test_forward_pass(self):
        """Test forward pass through UnetUpLayer3D"""
        for _ in range(100):
            # Create layer with random parameters
            layer = self._generate_random_up_layer()
            
            # Generate input and condition tensors
            x, condition = self._get_valid_input(layer)
            
            # Forward pass
            output = layer(x, condition)
            
            # Check output shape - should be doubled in spatial dimensions
            self.assertEqual(output.shape[0], x.shape[0])  # Batch size unchanged
            self.assertEqual(output.shape[1], layer.out_channels[-1])  # Channels match
            self.assertEqual(output.shape[2], x.shape[2] * 2)  # Height doubled
            self.assertEqual(output.shape[3], x.shape[3] * 2)  # Width doubled
    
    ########################## Spatial Dimension Tests ##########################
    
    def test_exact_doubling(self):
        """Test that spatial dimensions are exactly doubled for all types of upsampling"""
        for upsample_type in self.upsample_types:
            # Create layer with specific upsampling type
            layer = self._generate_random_up_layer(upsample_type=upsample_type)
            
            # Test different spatial dimensions
            for height in [8, 16, 32, 15, 31]:
                for width in [8, 16, 32, 17, 33]:
                    x, condition = self._get_valid_input(layer, height=height, width=width)
                    
                    # Forward pass
                    output = layer(x, condition)
                    
                    # Check output dimensions - should be exactly doubled
                    self.assertEqual(output.shape[2], height * 2, 
                                    f"Height {height} not correctly doubled with {upsample_type}")
                    self.assertEqual(output.shape[3], width * 2,
                                    f"Width {width} not correctly doubled with {upsample_type}")
    
    def test_odd_dimensions(self):
        """Test behavior with odd spatial dimensions"""
        for _ in range(50):
            layer = self._generate_random_up_layer()
            
            # Test with odd dimensions
            for height, width in [(15, 15), (31, 31), (17, 33)]:
                x, condition = self._get_valid_input(layer, height=height, width=width)
                
                # Forward pass
                output = layer(x, condition)
                
                # Check output dimensions - should be exactly doubled
                self.assertEqual(output.shape[2], height * 2)
                self.assertEqual(output.shape[3], width * 2)
    
    ########################## Edge Case Tests ##########################
    
    def test_small_dimensions(self):
        """Test upsampling from very small dimensions"""
        for _ in range(50):
            layer = self._generate_random_up_layer()
            
            # Test with small dimensions
            for height, width in [(1, 1), (2, 3), (3, 2), (4, 5)]:
                x, condition = self._get_valid_input(layer, height=height, width=width)
                
                # Forward pass
                output = layer(x, condition)
                
                # Check output dimensions - should be exactly doubled
                self.assertEqual(output.shape[2], height * 2)
                self.assertEqual(output.shape[3], width * 2)
    
    def test_num_blocks(self):
        """Test with varying numbers of resnet blocks"""
        for num_blocks in range(2, 6):  # Test 2 to 5 blocks
            layer = self._generate_random_up_layer(num_resnet_blocks=num_blocks)
            
            # Verify correct number of blocks
            self.assertEqual(len(layer._resnet_blocks), num_blocks)
            
            # Check that only the last block does upsampling
            for i in range(num_blocks - 1):
                self.assertFalse(hasattr(layer._resnet_blocks[i], '_upsample'))
            
            self.assertTrue(hasattr(layer._resnet_blocks[-1], '_upsample'))
            
            # Test forward pass
            x, condition = self._get_valid_input(layer)
            output = layer(x, condition)
            
            # Check output dimensions
            self.assertEqual(output.shape[2], x.shape[2] * 2)
            self.assertEqual(output.shape[3], x.shape[3] * 2)
    
    ########################## CustomModuleBaseTest Tests ##########################
    
    def test_eval_mode(self):
        """Test that the block can be set to evaluation mode"""
        for _ in range(100):
            block = self._generate_random_up_layer()
            super()._test_eval_mode(block)
    
    def test_train_mode(self):
        """Test that the block can be set to training mode"""
        for _ in range(100):
            block = self._generate_random_up_layer()
            super()._test_train_mode(block)

    def test_consistent_output_in_eval_mode(self):
        """Test consistent output in evaluation mode"""
        for _ in range(50): 
            block = self._generate_random_up_layer()
            x, condition = self._get_valid_input(block)
            super()._test_consistent_output_in_eval_mode(block, x, condition)

    def test_batch_size_one_in_train_mode(self):
        """Test handling of batch size 1 in training mode"""
        for _ in range(50):
            block = self._generate_random_up_layer()
            x, condition = self._get_valid_input(block, batch_size=1)
            super()._test_batch_size_one_in_train_mode(block, x, condition)

    def test_batch_size_one_in_eval_mode(self):
        """Test handling of batch size 1 in evaluation mode"""
        for _ in range(50):
            block = self._generate_random_up_layer()
            x, condition = self._get_valid_input(block, batch_size=1)
            super()._test_batch_size_one_in_eval_mode(block, x, condition)
    
    def test_named_parameters_length(self):
        """Test that named_parameters and parameters have the same length"""
        for _ in range(50):
            block = self._generate_random_up_layer()
            super()._test_named_parameters_length(block)

    def test_to_device(self):
        """Test that the block can be moved between devices"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping device tests")
            
        for _ in range(100): 
            block = self._generate_random_up_layer()
            x, condition = self._get_valid_input(block)
            super()._test_to_device(block, x, condition)

    def test_module_is_nn_module(self):
        """Test that the module is an instance of torch.nn.Module"""
        block = self._generate_random_up_layer()
        super()._test_module_is_nn_module(block)


if __name__ == '__main__':
    unittest.main()
