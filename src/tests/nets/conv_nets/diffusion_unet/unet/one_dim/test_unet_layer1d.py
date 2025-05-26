import torch
import random
import unittest

from typing import List, Optional, Tuple

from tests.custom_base_test import CustomModuleBaseTest
from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser
from mypt.nets.conv_nets.diffusion_unet.unet.one_dim.unet_layer1d import UnetDownLayer1D, UnetUpLayer1D


# @unittest.skip("skip for now")
class TestUnetDownLayer1D(CustomModuleBaseTest):
    """Test class for UnetDownLayer1D that verifies downsampling behavior"""
    
    def setUp(self):
        """Initialize test parameters"""
        self.dim_analyser = DimensionsAnalyser()
        
        # Define common test parameters
        self.in_channels_range = (1, 32)
        self.cond_channels_range = (4, 64)
        self.num_resnet_blocks_range = (2, 5)
        self.dropout_options = [0.0, 0.1, 0.3]
        self.downsample_types = ["conv", "avg_pool", "max_pool"]
        self.inner_dim_range = (16, 128)
        
        # For normalization and activation
        self.norm_types = ['batchnorm2d', 'groupnorm']
        self.activation_types = ['relu', 'leaky_relu', 'gelu']
    
    def _get_valid_input(self, 
                        layer: Optional[UnetDownLayer1D] = None, 
                        batch_size: int = 2, 
                        height: int = 32, 
                        width: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random input and condition tensors with correct shapes"""
        if layer is None:
            in_channels = random.randint(*self.in_channels_range)
            cond_dimension = random.randint(*self.cond_channels_range)
        else:
            in_channels = layer.in_channels
            cond_dimension = layer.cond_dimension
        
        # Generate input tensor
        x = torch.randn(batch_size, in_channels, height, width)
        condition = torch.randn(batch_size, cond_dimension)
        
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
                                   inner_dim=None) -> UnetDownLayer1D:
        """Generate a random UnetDownLayer1D with given or random parameters"""
        # Set random parameters if not provided
        if num_resnet_blocks is None:
            num_resnet_blocks = random.randint(*self.num_resnet_blocks_range)
        
        if in_channels is None:
            in_channels = random.randint(*self.in_channels_range)
        
        if cond_dimension is None:
            cond_dimension = random.randint(*self.cond_channels_range)
        
        if out_channels is None:
            out_channels = self._generate_random_channel_list(num_resnet_blocks)
        
        if dropout_rate is None:
            dropout_rate = random.choice(self.dropout_options)
        
        if downsample_type is None:
            downsample_type = random.choice(self.downsample_types)
        
        if inner_dim is None:
            inner_dim = random.randint(*self.inner_dim_range)
        
        # Create layer
        return UnetDownLayer1D(
            num_resnet_blocks=num_resnet_blocks,
            in_channels=in_channels,
            cond_dimension=cond_dimension,
            out_channels=out_channels,
            inner_dim=inner_dim,
            dropout_rate=dropout_rate,
            downsample_type=downsample_type
        )
    
    ########################## Basic Tests ##########################
    
    def _verify_correct_output_shape(self, down_sample_type: str, in_height: int, in_width: int, out_height: int, out_width: int) -> None:
        """Verify that the output shape is correct"""
        if down_sample_type == "conv":
            self.assertEqual(out_height, (in_height + 1) // 2)
            self.assertEqual(out_width, (in_width + 1) // 2)
        else:
            self.assertEqual(out_height, in_height // 2)
            self.assertEqual(out_width, in_width // 2)

    def test_initialization(self):
        """Test that UnetDownLayer1D initializes correctly with various parameters"""
        for _ in range(100):
            num_blocks = random.randint(*self.num_resnet_blocks_range)
            layer = self._generate_random_down_layer(num_resnet_blocks=num_blocks)
            
            # Check that the layer has the correct number of resnet blocks
            self.assertEqual(len(layer._resnet_blocks), num_blocks) 
            
            # Check that the last block is a DownCondOneDimWResBlock
            from mypt.building_blocks.conv_blocks.conditioned.one_dim.resnet_con1d import DownCondOneDimWResBlock
            self.assertIsInstance(layer._resnet_blocks[-1], DownCondOneDimWResBlock)
    
    def test_forward_pass(self):    
        """Test basic forward pass functionality"""
        for _ in range(100):
            layer = self._generate_random_down_layer()
            x, condition = self._get_valid_input(layer)
            
            # Forward pass should work without errors
            output = layer(x, condition)
            
            # Output should be a tensor
            self.assertIsInstance(output, torch.Tensor)
            
            # Output should have the same batch size and expected channel count
            self.assertEqual(output.shape[0], x.shape[0])
            self.assertEqual(output.shape[1], layer.out_channels[-1])
    
    ########################## Dimension Tests ##########################
    
    def test_dimensions_halved(self):
        """Test that spatial dimensions are exactly halved in the output"""
        for _ in range(100):
            layer = self._generate_random_down_layer()
            
            # Test with various shapes
            for height, width in [(16, 16), (32, 24), (17, 31)]:
                x, condition = self._get_valid_input(layer, height=height, width=width)
                
                # Forward pass
                output = layer(x, condition)
                
                # Check output dimensions
                self._verify_correct_output_shape(layer.downsample_type, height, width, output.shape[2], output.shape[3])
    
    def test_odd_dimensions(self):
        """Test handling of odd input dimensions"""
        for _ in range(50):
            layer = self._generate_random_down_layer()
            
            # Test with odd dimensions
            heights = [15, 17, 21, 33]
            widths = [15, 23, 31, 41]
            
            for height in heights:
                for width in widths:
                    x, condition = self._get_valid_input(layer, height=height, width=width)
                    
                    # Forward pass
                    output = layer(x, condition)
                    
                    # Check output dimensions
                    self._verify_correct_output_shape(layer.downsample_type, height, width, output.shape[2], output.shape[3])
    

    def test_different_block_counts(self):
        """Test with different numbers of resnet blocks"""
        for num_blocks in range(2, 6):
            for _ in range(20):
                layer = self._generate_random_down_layer(num_resnet_blocks=num_blocks)
                x, condition = self._get_valid_input(layer)
                
                # Forward pass
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


# @unittest.skip("Skipping UnetUpLayer1D tests")
class TestUnetUpLayer1D(CustomModuleBaseTest):
    """Test class for UnetUpLayer1D that verifies upsampling behavior"""
    
    def setUp(self):
        """Initialize test parameters"""
        self.dim_analyser = DimensionsAnalyser()
        
        # Define common test parameters
        self.in_channels_range = (1, 32)
        self.cond_channels_range = (4, 64)
        self.num_resnet_blocks_range = (2, 5)
        self.dropout_options = [0.0, 0.1, 0.3]
        self.upsample_types = ["transpose_conv", "conv", "interpolate"]
        self.inner_dim_range = (16, 128)
        
        # For normalization and activation
        self.norm_types = ['batchnorm2d', 'groupnorm']
        self.activation_types = ['relu', 'leaky_relu', 'gelu']
    
    def _get_valid_input(self, 
                        layer: Optional[UnetUpLayer1D] = None, 
                        batch_size: int = 2, 
                        height: int = 16, 
                        width: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random input and condition tensors with correct shapes"""
        if layer is None:
            in_channels = random.randint(*self.in_channels_range)
            cond_dimension = random.randint(*self.cond_channels_range)
        else:
            in_channels = layer.in_channels
            cond_dimension = layer.cond_dimension
        
        # Generate input tensor
        x = torch.randn(batch_size, in_channels, height, width)
        condition = torch.randn(batch_size, cond_dimension)
        
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
                                 inner_dim=None) -> UnetUpLayer1D:
        """Generate a random UnetUpLayer1D with given or random parameters"""
        # Set random parameters if not provided
        if num_resnet_blocks is None:
            num_resnet_blocks = random.randint(*self.num_resnet_blocks_range)
        
        if in_channels is None:
            in_channels = random.randint(*self.in_channels_range)
        
        if cond_dimension is None:
            cond_dimension = random.randint(*self.cond_channels_range)
        
        if out_channels is None:
            out_channels = self._generate_random_channel_list(num_resnet_blocks)
        
        if dropout_rate is None:
            dropout_rate = random.choice(self.dropout_options)
        
        if upsample_type is None:
            upsample_type = random.choice(self.upsample_types)
        
        if inner_dim is None:
            inner_dim = random.randint(*self.inner_dim_range)
        
        # Create layer
        return UnetUpLayer1D(
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
        """Test that UnetUpLayer1D initializes correctly with various parameters"""
        for _ in range(100):
            num_blocks = random.randint(*self.num_resnet_blocks_range)
            layer = self._generate_random_up_layer(num_resnet_blocks=num_blocks)
            
            # Check that the layer has the correct number of resnet blocks
            self.assertEqual(len(layer._resnet_blocks), num_blocks)
            
            # Check that the last block is an UpCondOneDimWResBlock
            from mypt.building_blocks.conv_blocks.conditioned.one_dim.resnet_con1d import UpCondOneDimWResBlock
            self.assertIsInstance(layer._resnet_blocks[-1], UpCondOneDimWResBlock)
    
    def test_forward_pass(self):
        """Test basic forward pass functionality"""
        for _ in range(100):
            layer = self._generate_random_up_layer()
            x, condition = self._get_valid_input(layer)
            
            # Forward pass should work without errors
            output = layer(x, condition)
            
            # Output should be a tensor
            self.assertIsInstance(output, torch.Tensor)
            
            # Output should have the same batch size and expected channel count
            self.assertEqual(output.shape[0], x.shape[0])
            self.assertEqual(output.shape[1], layer.out_channels[-1])
    
    ########################## Dimension Tests ##########################
    
    def test_dimensions_doubled(self):
        """Test that spatial dimensions are exactly doubled in the output"""
        for _ in range(100):
            layer = self._generate_random_up_layer()
            
            # Test with various shapes
            for height, width in [(16, 16), (32, 24), (17, 31)]:
                x, condition = self._get_valid_input(layer, height=height, width=width)
                
                # Forward pass
                output = layer(x, condition)
                
                # Check output dimensions - should be exactly doubled
                self.assertEqual(output.shape[2], height * 2)
                self.assertEqual(output.shape[3], width * 2)
    
    def test_odd_dimensions(self):
        """Test handling of odd input dimensions"""
        for _ in range(50):
            layer = self._generate_random_up_layer()
            
            # Test with odd dimensions
            heights = [15, 17, 21, 33]
            widths = [15, 23, 31, 41]
            
            for height in heights:
                for width in widths:
                    x, condition = self._get_valid_input(layer, height=height, width=width)
                    
                    # Forward pass
                    output = layer(x, condition)
                    
                    # Check output dimensions - should be exactly doubled
                    self.assertEqual(output.shape[2], height * 2)
                    self.assertEqual(output.shape[3], width * 2)
    
    def test_different_block_counts(self):
        """Test with different numbers of resnet blocks"""
        for num_blocks in range(2, 6):
            for _ in range(20):
                layer = self._generate_random_up_layer(num_resnet_blocks=num_blocks)
                x, condition = self._get_valid_input(layer)
                
                # Forward pass
                output = layer(x, condition)
                
                # Check output dimensions - should be exactly doubled
                self.assertEqual(output.shape[2], x.shape[2] * 2)
                self.assertEqual(output.shape[3], x.shape[3] * 2)
    
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
