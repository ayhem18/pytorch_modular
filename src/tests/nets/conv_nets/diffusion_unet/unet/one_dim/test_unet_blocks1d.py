import torch
import random
import unittest
import math

from typing import List, Optional, Tuple, Union

from tests.custom_base_test import CustomModuleBaseTest
from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser
from mypt.nets.conv_nets.diffusion_unet.unet.one_dim.unet_block1d import UnetDownBlock1D, UnetUpBlock1D


# @unittest.skip("Skipping UnetDownBlock1D tests")
class TestUnetDownBlock1D(CustomModuleBaseTest):
    """Test class for UnetDownBlock1D that verifies downsampling behavior"""
    
    def setUp(self):
        """Initialize test parameters"""
        self.dim_analyser = DimensionsAnalyser()
        
        # Define common test parameters
        self.in_channels_range = (1, 32)
        self.cond_dimension_range = (4, 64)
        self.num_resnet_blocks_range = (2, 4)
        self.num_down_layers_range = (1, 4)
        self.dropout_options = [0.0, 0.1, 0.3]
        self.downsample_types = ["conv", "avg_pool", "max_pool"]
        self.inner_dim_range = (16, 128)
        
        # For normalization and activation
        self.norm_types = ['batchnorm2d', 'groupnorm']
        self.activation_types = ['relu', 'leaky_relu', 'gelu']
    
    def _get_valid_input(self, 
                         block: Optional[UnetDownBlock1D] = None, 
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
        
        # Generate input tensor
        x = torch.randn(batch_size, in_channels, height, width)
        condition = torch.randn(batch_size, cond_dimension)
        
        return x, condition
    
    def _generate_random_channel_list(self, num_layers: int) -> List[int]:
        """Generate a random list of channel counts"""
        return [random.randint(16, 64) for _ in range(num_layers)]
    
    def _generate_random_down_block(self,
                                   num_down_layers=None,
                                   num_resnet_blocks=None,
                                   in_channels=None,
                                   cond_dimension=None,
                                   out_channels=None,
                                   downsample_types=None,
                                   dropout_rate=None,
                                   inner_dim=None) -> UnetDownBlock1D:
        """Generate a random UnetDownBlock1D with given or random parameters"""
        # Set random parameters if not provided
        if num_down_layers is None:
            num_down_layers = random.randint(*self.num_down_layers_range)
        
        if num_resnet_blocks is None:
            num_resnet_blocks = random.randint(*self.num_resnet_blocks_range)
        
        if in_channels is None:
            in_channels = random.randint(*self.in_channels_range)
        
        if cond_dimension is None:
            cond_dimension = random.randint(*self.cond_dimension_range)
        
        if out_channels is None:
            out_channels = self._generate_random_channel_list(num_down_layers)
        
        if downsample_types is None:
            downsample_types = [random.choice(self.downsample_types) for _ in range(num_down_layers)]
        
        if dropout_rate is None:
            dropout_rate = random.choice(self.dropout_options)
        
        if inner_dim is None:
            inner_dim = random.randint(*self.inner_dim_range)

        # Create block
        return UnetDownBlock1D(
            num_down_layers=num_down_layers,
            num_resnet_blocks=num_resnet_blocks,
            in_channels=in_channels,
            cond_dimension=cond_dimension,
            out_channels=out_channels,
            downsample_types=downsample_types,
            inner_dim=inner_dim,
            dropout_rate=dropout_rate
        )
    
    def _verify_down_dimensions(self, input_shape, output_shape, num_down_layers):
        """Verify that dimensions are reduced by factor of 2^num_down_layers"""
        expected_height = input_shape[2] // (2 ** num_down_layers)
        expected_width = input_shape[3] // (2 ** num_down_layers)
        
        self.assertEqual(output_shape[2], expected_height, 
                        f"Expected height {expected_height}, got {output_shape[2]}")
        self.assertEqual(output_shape[3], expected_width, 
                        f"Expected width {expected_width}, got {output_shape[3]}")
    
    ########################## Basic Functionality Tests ##########################
    
    def test_downsampling(self):
        """Test that the block correctly downsamples inputs by 2^n"""
        for _ in range(50):
            num_down_layers = random.randint(*self.num_down_layers_range)
            block = self._generate_random_down_block(num_down_layers=num_down_layers)
            
            # Calculate input dimensions that can be properly downsampled
            # Must be of the form 2^n * k where k is any number >= 1
            k = random.randint(1, 8)
            height = (2 ** num_down_layers) * k
            width = (2 ** num_down_layers) * k
            
            x, condition = self._get_valid_input(block, height=height, width=width)
            
            # Forward pass
            output, skip_outputs = block(x, condition)
            
            # Verify output dimensions
            self._verify_down_dimensions(x.shape, output.shape, num_down_layers)
            
            # Verify final dimensions are exactly k
            self.assertEqual(output.shape[2], k, f"Expected final height {k}, got {output.shape[2]}")
            self.assertEqual(output.shape[3], k, f"Expected final width {k}, got {output.shape[3]}")
            
            # Verify skip connections are correctly stored
            self.assertEqual(len(skip_outputs), num_down_layers, 
                            f"Expected {num_down_layers} skip outputs, got {len(skip_outputs)}")
    
    # @unittest.skip("Skipping skip connection tests")
    def test_skip_connections(self):
        """Test that skip connections are properly created"""
        for _ in range(50):
            num_down_layers = random.randint(*self.num_down_layers_range)
            block = self._generate_random_down_block(num_down_layers=num_down_layers)
            
            # Create input of suitable size
            k = random.randint(1, 8)
            height = (2 ** num_down_layers) * k
            width = (2 ** num_down_layers) * k
            
            x, condition = self._get_valid_input(block, height=height, width=width)
            
            # Forward pass
            output, skip_outputs = block(x, condition)
            
            # Verify skip connections have correct shape progression
            expected_heights = [height // (2 ** i) for i in range(1, num_down_layers + 1)]
            expected_widths = [width // (2 ** i) for i in range(1, num_down_layers + 1)]
            
            for i, skip in enumerate(skip_outputs):
                self.assertEqual(skip.shape[2], expected_heights[i], 
                                f"Skip {i} height mismatch: expected {expected_heights[i]}, got {skip.shape[2]}")
                self.assertEqual(skip.shape[3], expected_widths[i], 
                                f"Skip {i} width mismatch: expected {expected_widths[i]}, got {skip.shape[3]}")
    
    # @unittest.skip("Skipping downsample type tests")
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
    # @unittest.skip("Skipping minimum input size tests")
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
    
    # @unittest.skip("Skipping different layer counts tests")
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
    # @unittest.skip("Skipping eval mode tests")
    def test_eval_mode(self):
        """Test that the block can be set to evaluation mode"""
        for _ in range(50):
            block = self._generate_random_down_block()
            super()._test_eval_mode(block)
    
    # @unittest.skip("Skipping train mode tests")
    def test_train_mode(self):
        """Test that the block can be set to training mode"""
        for _ in range(50):
            block = self._generate_random_down_block()
            super()._test_train_mode(block)
    
    # @unittest.skip("Skipping consistent output in eval mode tests")
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
    
    # @unittest.skip("Skipping batch size one tests")
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
    
    # @unittest.skip("Skipping named parameters length tests")
    def test_named_parameters_length(self):
        """Test that named_parameters and parameters have the same length"""
        for _ in range(20):
            block = self._generate_random_down_block()
            super()._test_named_parameters_length(block)
    
    # @unittest.skip("Skipping device tests")
    def test_to_device(self):
        """Test that the block can be moved between devices"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping device tests")
            
        for _ in range(10):  # Limit for performance
            block = self._generate_random_down_block()
            
            # Create valid input
            num_down_layers = len(block.down_layers)
            k = random.randint(1, 4)
            height = width = 2 ** num_down_layers * k
            
            x, condition = self._get_valid_input(block, height=height, width=width)
            super()._test_to_device(block, x, condition)



class TestUnetUpBlock1D(CustomModuleBaseTest):
    """Test class for UnetUpBlock1D that verifies upsampling behavior"""
    
    def setUp(self):
        """Initialize test parameters"""
        self.dim_analyser = DimensionsAnalyser()
        
        # Define common test parameters
        self.in_channels_range = (16, 64)
        self.cond_dimension_range = (4, 64)
        self.num_resnet_blocks_range = (2, 4)
        self.num_up_layers_range = (1, 4)
        self.dropout_options = [0.0, 0.1, 0.3]
        self.upsample_types = ["transpose_conv", "conv", "interpolate"]
        self.inner_dim_range = (16, 128)
        
        # For normalization and activation
        self.norm_types = ['batchnorm2d', 'groupnorm']
        self.activation_types = ['relu', 'leaky_relu', 'gelu']
    
    def _generate_random_channel_list(self, num_blocks: int) -> List[int]:
        """Generate a random list of channel counts"""
        return [random.randint(16, 64) for _ in range(num_blocks)]
    
    def _get_valid_input_and_skip(self, 
                                 block: UnetUpBlock1D, 
                                 batch_size: int = 2, 
                                 base_size: int = 4) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Generate random input, skip connections, and condition with correct shapes"""
        in_channels = block.in_channels
        cond_dimension = block.cond_dimension
        num_up_layers = len(block.up_layers)
        out_channels = block.out_channels
        
        # Calculate appropriate input size
        input_height = input_width = base_size
        
        # Generate input tensor
        x = torch.randn(batch_size, in_channels, input_height, input_width)
        
        # Generate skip connection tensors
        skip_outputs = [None for _ in range(num_up_layers)]
        
        for i in range(num_up_layers):
            # Each skip connection has progressively larger spatial dimensions
            skip_height = skip_width = input_height * (2 ** i)
            # out_channels is a list of lists
            # skip_channels is either in_channels
            # or the input to the i-th up_layer: which is the output of the (i-1)-th up_layer 
            # which can be found as the last element in the out_channels[i - 1] list
            skip_channels = out_channels[i - 1][-1] if i > 0 else in_channels
            skip = torch.randn(batch_size, skip_channels, skip_height, skip_width)
            skip_outputs[i] = skip
        
        # Generate condition tensor
        condition = torch.randn(batch_size, cond_dimension)
        
        return x, skip_outputs, condition
    
    def _generate_random_up_block(self,
                                 num_up_layers=None,
                                 num_resnet_blocks=None,
                                 in_channels=None,
                                 cond_dimension=None,
                                 out_channels=None,
                                 upsample_types=None,
                                 dropout_rate=None,
                                 inner_dim=None) -> UnetUpBlock1D:
        """Generate a random UnetUpBlock1D with given or random parameters"""
        # Set random parameters if not provided
        if num_up_layers is None:
            num_up_layers = random.randint(*self.num_up_layers_range)
        
        if num_resnet_blocks is None:
            num_resnet_blocks = random.randint(*self.num_resnet_blocks_range)
        
        if in_channels is None:
            in_channels = random.randint(*self.in_channels_range)
        
        if cond_dimension is None:
            cond_dimension = random.randint(*self.cond_dimension_range)
        
        if out_channels is None:
            out_channels = self._generate_random_channel_list(num_up_layers)
        
        if upsample_types is None:
            upsample_types = [random.choice(self.upsample_types) for _ in range(num_up_layers)]
        
        if dropout_rate is None:
            dropout_rate = random.choice(self.dropout_options)
        
        if inner_dim is None:
            inner_dim = random.randint(*self.inner_dim_range)

        # Create block
        return UnetUpBlock1D(
            num_up_layers=num_up_layers,
            num_resnet_blocks=num_resnet_blocks,
            in_channels=in_channels,
            cond_dimension=cond_dimension,
            out_channels=out_channels,
            upsample_types=upsample_types,
            inner_dim=inner_dim,
            dropout_rate=dropout_rate
        )
    
    def _verify_up_dimensions(self, input_shape, output_shape, num_up_layers):
        """Verify that dimensions are increased by factor of 2^num_up_layers"""
        expected_height = input_shape[2] * (2 ** num_up_layers)
        expected_width = input_shape[3] * (2 ** num_up_layers)
        
        self.assertEqual(output_shape[2], expected_height, 
                        f"Expected height {expected_height}, got {output_shape[2]}")
        self.assertEqual(output_shape[3], expected_width, 
                        f"Expected width {expected_width}, got {output_shape[3]}")
    
    ########################## Basic Functionality Tests ##########################
    def test_upsampling(self):
        """Test that the block correctly upsamples inputs by 2^n"""
        for _ in range(50):
            num_up_layers = random.randint(*self.num_up_layers_range)
            block = self._generate_random_up_block(num_up_layers=num_up_layers)
            
            # Base size can be anything
            base_size = random.randint(1, 8)
            
            x, skip_outputs, condition = self._get_valid_input_and_skip(
                block, base_size=base_size
            )
            
            # Forward pass
            output = block(x, skip_outputs, condition, reverse_skip_connections=False)
            
            # Verify output dimensions
            self._verify_up_dimensions(x.shape, output.shape, num_up_layers)
            
            # Verify final dimensions are exactly 2^n * base_size
            expected_size = base_size * (2 ** num_up_layers)
            self.assertEqual(output.shape[2], expected_size, 
                            f"Expected final height {expected_size}, got {output.shape[2]}")
            self.assertEqual(output.shape[3], expected_size, 
                            f"Expected final width {expected_size}, got {output.shape[3]}")
    
    # @unittest.skip("passed")
    def test_different_upsample_types(self):
        """Test with different upsample types"""
        for _ in range(50):
            num_up_layers = random.randint(2, 4)
            
            # Try all upsample types
            for upsample_type in self.upsample_types:
                upsample_types = [upsample_type] * num_up_layers
                
                block = self._generate_random_up_block(
                    num_up_layers=num_up_layers,
                    upsample_types=upsample_types
                )
                
                # Base size can be anything
                base_size = random.randint(1, 8)
                
                x, skip_outputs, condition = self._get_valid_input_and_skip(
                    block, base_size=base_size
                )
                
                # Forward pass
                output = block(x, skip_outputs, condition, reverse_skip_connections=False)
                
                # Verify output dimensions
                self._verify_up_dimensions(x.shape, output.shape, num_up_layers)
    
    ########################## Edge Case Tests ##########################
    # @unittest.skip("passed")
    def test_single_pixel_input(self):
        """Test upsampling from a single pixel"""
        for _ in range(50):
            num_up_layers = random.randint(1, 4)
            block = self._generate_random_up_block(num_up_layers=num_up_layers)
            
            # Input is 1x1
            x, skip_outputs, condition = self._get_valid_input_and_skip(
                block, base_size=1
            )
            
            # Forward pass
            output = block(x, skip_outputs, condition, reverse_skip_connections=False)
            
            # Verify output dimensions
            expected_size = 2 ** num_up_layers
            self.assertEqual(output.shape[2], expected_size)
            self.assertEqual(output.shape[3], expected_size)
    
    # @unittest.skip("passed")
    def test_different_layer_counts(self):
        """Test with different numbers of up layers"""
        for num_up_layers in range(1, 5):
            block = self._generate_random_up_block(num_up_layers=num_up_layers)
            
            # Verify layer count is correct
            self.assertEqual(len(block.up_layers), num_up_layers)
            
            # Create valid input
            base_size = random.randint(1, 4)
            
            x, skip_outputs, condition = self._get_valid_input_and_skip(
                block, base_size=base_size
            )
            
            # Forward pass
            output = block(x, skip_outputs, condition, reverse_skip_connections=False)
            
            # Verify output dimensions
            expected_size = base_size * (2 ** num_up_layers)
            self.assertEqual(output.shape[2], expected_size)
            self.assertEqual(output.shape[3], expected_size)
    
    ########################## CustomModuleBaseTest Tests ##########################
    # @unittest.skip("passed")
    def test_eval_mode(self):
        """Test that the block can be set to evaluation mode"""
        for _ in range(100):
            block = self._generate_random_up_block()
            super()._test_eval_mode(block)

    # @unittest.skip("passed")
    def test_train_mode(self):
        """Test that the block can be set to training mode"""
        for _ in range(100):
            block = self._generate_random_up_block()
            super()._test_train_mode(block)

    # @unittest.skip("passed")
    def test_consistent_output_in_eval_mode(self):
        """Test consistent output in evaluation mode"""
        for _ in range(50): 
            block = self._generate_random_up_block()
                        # Create valid input
            base_size = random.randint(1, 4)
            
            x, skip_outputs, condition = self._get_valid_input_and_skip(
                block, base_size=base_size
            )
            # the forward method for the up block expects the arguments in the following order:
            # input tensor: x
            # skip outputs: skip_outputs 
            # condition: condition 
            # reverse_skip_connections: reverse_skip_connections
            super()._test_consistent_output_in_eval_mode(block, x, skip_outputs, condition, reverse_skip_connections=False)

    # @unittest.skip("passed")
    def test_batch_size_one_in_train_mode(self):
        """Test handling of batch size 1 in training mode"""
        for _ in range(50):
            block = self._generate_random_up_block()
            # Create valid input
            base_size = random.randint(1, 4)
            
            x, skip_outputs, condition = self._get_valid_input_and_skip(
                block, base_size=base_size, batch_size=1
            )   
            # the forward method for the up block expects the arguments in the following order:
            # input tensor: x
            # skip outputs: skip_outputs 
            # condition: condition 
            # reverse_skip_connections: reverse_skip_connections
            super()._test_batch_size_one_in_train_mode(block, x, skip_outputs, condition, reverse_skip_connections=False)

    # @unittest.skip("passed")
    def test_batch_size_one_in_eval_mode(self):
        """Test handling of batch size 1 in evaluation mode"""
        for _ in range(50):
            block = self._generate_random_up_block()
            # Create valid input
            base_size = random.randint(1, 4)
            
            x, skip_outputs, condition = self._get_valid_input_and_skip(
                block, base_size=base_size, batch_size=1
            )   
            # the forward method for the up block expects the arguments in the following order:
            # input tensor: x
            # skip outputs: skip_outputs 
            # condition: condition 
            # reverse_skip_connections: reverse_skip_connections
            super()._test_batch_size_one_in_eval_mode(block, x, skip_outputs, condition, reverse_skip_connections=False)

    # @unittest.skip("passed")
    def test_named_parameters_length(self):
        """Test that named_parameters and parameters have the same length"""
        for _ in range(50):
            block = self._generate_random_up_block()
            super()._test_named_parameters_length(block)

    # @unittest.skip("Skipping device tests")
    def test_to_device(self):
        """Test that the block can be moved between devices"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping device tests")
            
        for _ in range(100): 
            block = self._generate_random_up_block()
            # Create valid input
            base_size = random.randint(1, 4)
            
            x, skip_outputs, condition = self._get_valid_input_and_skip(
                block, base_size=base_size
            )
            # the forward method for the up block expects the arguments in the following order:
            # input tensor: x
            # skip outputs: skip_outputs 
            # condition: condition 
            # reverse_skip_connections: reverse_skip_connections
            super()._test_to_device(block, x, skip_outputs, condition, reverse_skip_connections=False)

    # @unittest.skip("passed")
    def test_module_is_nn_module(self):
        """Test that the module is an instance of torch.nn.Module"""
        block = self._generate_random_up_block()
        super()._test_module_is_nn_module(block)




if __name__ == '__main__':
    unittest.main()
