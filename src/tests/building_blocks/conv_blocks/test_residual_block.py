import torch
import unittest
import random

from torch import nn
from random import randint as ri
from typing import Tuple

import mypt.code_utils.pytorch_utils as pu
from tests.custom_base_test import CustomModuleBaseTest
from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser
from mypt.building_blocks.conv_blocks.residual_conv_block import ResidualConvBlock
from mypt.building_blocks.conv_blocks.conv_block import BasicConvBlock


class TestResidualConvBlock(CustomModuleBaseTest):
    """Test class for ResidualConvBlock implementation"""
    
    def setUp(self):
        """Initialize test parameters"""
        self.dim_analyser = DimensionsAnalyser()
        
        # Define standard activation functions for testing
        self.activation_modules = [nn.ReLU(), nn.LeakyReLU(), nn.Tanh()]
        
        # Define common test parameters
        self.input_channels_range = (1, 16)
        self.output_channels_range = (1, 32)
        self.num_layers_range = (2, 4)
        self.kernel_sizes_options = [3, 5, 7]
        self.strides_options = [1, 2]
        self.padding_options = ['same', 0, 1]
    
    def _get_valid_input(self, block=None, batch_size=2, height=32, width=32) -> torch.Tensor:
        """Generate a random input tensor with the correct shape for the given block"""
        if block is None:
            in_channels = ri(*self.input_channels_range)
        else:
            in_channels = block._channels[0]
        
        return torch.randn(batch_size, in_channels, height, width)
    
    def _generate_random_residual_conv_block(self, 
                                            num_conv_layers=None, 
                                            matching_dimensions=None,
                                            force_residual=None,
                                            activation=None,
                                            kernel_sizes=None):
        """
        Generate a random ResidualConvBlock with configurable parameters
        
        Args:
            num_conv_layers: Number of convolutional layers (default: random)
            matching_dimensions: Whether input and output dimensions should match (default: random)
            force_residual: Whether to force a residual connection (default: random)
            activation: Activation function to use (default: random)
            
        Returns:
            A randomly configured ResidualConvBlock
        """
        # Generate random parameters if not specified
        if num_conv_layers is None:
            num_conv_layers = ri(*self.num_layers_range)
            
        if matching_dimensions is None:
            matching_dimensions = random.choice([True, False])
            
        if force_residual is None:
            force_residual = random.choice([True, False])
            
        if activation is None:
            activation = random.choice(self.activation_modules)
        
        # Generate channel dimensions
        in_channels = ri(*self.input_channels_range)
        
        if matching_dimensions:
            out_channels = in_channels
            paddings = 'same'
        else:
            out_channels = ri(*self.output_channels_range)
            while out_channels == in_channels:
                out_channels = ri(*self.output_channels_range)
            
            # the exact value here does not matter, since the number of output channels is different
            paddings = random.choice(self.padding_options)

        # Create a list of channels for all layers
        channels = [in_channels] * num_conv_layers + [out_channels]
        # Generate other parameters
        if kernel_sizes is None:
            kernel_sizes = random.choice(self.kernel_sizes_options)
        else:
            if isinstance(kernel_sizes, int):
                kernel_sizes = [kernel_sizes] * num_conv_layers
            else:
                assert len(kernel_sizes) == num_conv_layers

        # the `same` padding is not supported for strides > 1
        if paddings == 'same':
            strides = 1
        else:
            strides = random.choice(self.strides_options)

        use_bn = random.choice([True, False])
        activation_after_each_layer = random.choice([True, False])
        final_bn_layer = random.choice([True, False])
        
        # Create the block
        return ResidualConvBlock(
            num_conv_layers=num_conv_layers,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            use_bn=use_bn,
            activation_after_each_layer=activation_after_each_layer,
            activation=activation,
            activation_params=None,
            final_bn_layer=final_bn_layer,
            force_residual=force_residual
        )
    
    ########################## CustomModuleBaseTest tests ##########################
    
    def test_eval_mode(self):
        """Test that the block can be set to evaluation mode"""
        for _ in range(100):
            block = self._generate_random_residual_conv_block()
            super()._test_eval_mode(block)
    
    def test_train_mode(self):
        """Test that the block can be set to training mode"""
        for _ in range(100):
            block = self._generate_random_residual_conv_block()
            super()._test_train_mode(block)
    
    def test_consistent_output_without_dropout_bn(self):
        """Test consistent output for non-stochastic blocks"""
        for _ in range(100):
            # Create a block without batch norm or dropout
            block = self._generate_random_residual_conv_block()
            block.use_bn = False
            
            input_tensor = self._get_valid_input(block)
            super()._test_consistent_output_without_dropout_bn(block, input_tensor)
    
    def test_consistent_output_in_eval_mode(self):
        """Test consistent output in evaluation mode"""
        for _ in range(100):
            block = self._generate_random_residual_conv_block()
            input_tensor = self._get_valid_input(block)
            super()._test_consistent_output_in_eval_mode(block, input_tensor)
    
    def test_batch_size_one_in_train_mode(self):
        """Test handling of batch size 1 in training mode"""
        for _ in range(100):
            block = self._generate_random_residual_conv_block()
            input_tensor = self._get_valid_input(block, batch_size=1)
            super()._test_batch_size_one_in_train_mode(block, input_tensor)
    
    def test_batch_size_one_in_eval_mode(self):
        """Test handling of batch size 1 in evaluation mode"""
        for _ in range(100):
            block = self._generate_random_residual_conv_block()
            input_tensor = self._get_valid_input(block, batch_size=1)
            super()._test_batch_size_one_in_eval_mode(block, input_tensor)
    
    def test_named_parameters_length(self):
        """Test that named_parameters and parameters have the same length"""
        for _ in range(100):
            block = self._generate_random_residual_conv_block()
            super()._test_named_parameters_length(block)
    
    def test_to_device(self):
        """Test that the block can be moved between devices"""
        for _ in range(100):
            block = self._generate_random_residual_conv_block()
            super()._test_to_device(block)
    
    ########################## ResidualConvBlock specific tests ##########################

    @unittest.skip("skip for now")
    def test_block_structure(self):
        """Test that the ResidualConvBlock has the correct structure"""
        for _ in range(100):
            num_layers = ri(*self.num_layers_range)
            block = self._generate_random_residual_conv_block(num_conv_layers=num_layers)
            
            # Check main stream is a BasicConvBlock
            self.assertIsInstance(block._block, BasicConvBlock)
            self.assertEqual(block._block.num_conv_layers, num_layers)
            
            # Check if adaptive layer is present when needed
            if block._channels[0] != block._channels[-1] or block._force_residual:
                self.assertIsNotNone(block._adaptive_layer)
                self.assertIsInstance(block._adaptive_layer, nn.Conv2d)
            else:
                self.assertIsNone(block._adaptive_layer)
    

    @unittest.skip("skip for now")
    def test_adaptive_layer_creation(self):
        """Test that adaptive layer is created when necessary"""
        # Test with matching dimensions and force_residual=False
        block = self._generate_random_residual_conv_block(matching_dimensions=True, force_residual=False)
        self.assertIsNone(block._adaptive_layer, 
                         "Adaptive layer should not be created when dimensions match and force_residual=False")
        
        # Test with non-matching dimensions
        block = self._generate_random_residual_conv_block(matching_dimensions=False, force_residual=False)
        self.assertIsNotNone(block._adaptive_layer, 
                            "Adaptive layer should be created when dimensions don't match")
        self.assertEqual(block._adaptive_layer.in_channels, block._channels[0])
        self.assertEqual(block._adaptive_layer.out_channels, block._channels[-1])
        
        # Test with force_residual=True
        block = self._generate_random_residual_conv_block(matching_dimensions=True, force_residual=True)
        self.assertIsNotNone(block._adaptive_layer, 
                            "Adaptive layer should be created when force_residual=True")
    
    
    @unittest.skip("skip for now")
    def test_method_overrides(self):
        """Test that all overridden methods work correctly"""
        block = self._generate_random_residual_conv_block()
        
        # Test children and named_children
        children = list(block.children())
        named_children = list(block.named_children())
        
        # Should contain at least the main block and possibly the adaptive layer
        self.assertGreaterEqual(len(children), 1)
        self.assertGreaterEqual(len(named_children), 1)
        
        # Test modules
        modules = list(block.modules())
        self.assertGreater(len(modules), 1)  # Should contain at least the block itself
        
        # Test parameters
        params = list(block.parameters())
        named_params = list(block.named_parameters())
        self.assertEqual(len(params), len(named_params))
        
        # Test to, train, and eval
        # These methods should return self
        self.assertIs(block.to(torch.float), block)
        self.assertIs(block.train(), block)
        self.assertIs(block.eval(), block)


    @unittest.skip("skip for now")
    def test_error_on_large_kernel_without_input_shape(self):
        """Test that an error is raised when kernel_sizes > 1 and input_shape is not provided"""
        # Test with scalar kernel_size > 1
        with self.assertRaises(ValueError):
            ResidualConvBlock(
                num_conv_layers=2,
                channels=[16, 16, 32],
                kernel_sizes=3,
                strides=1,
                paddings='same',
                use_bn=True,
                force_residual=False
            )
            
        # Test with list containing kernel_sizes > 1
        kernel_size_combinations = [
            [3, 1],  # First layer has kernel_size > 1
            [1, 3],  # Second layer has kernel_size > 1
            [3, 5],  # All layers have kernel_size > 1
            [5, 7]   # All layers have larger kernel_size
        ]
        
        for kernel_sizes in kernel_size_combinations:
            with self.subTest(kernel_sizes=kernel_sizes):
                with self.assertRaises(ValueError):
                    ResidualConvBlock(
                        num_conv_layers=len(kernel_sizes),
                        channels=[16] * len(kernel_sizes) + [32],
                        kernel_sizes=kernel_sizes,
                        strides=1,
                        paddings='same',
                        use_bn=True,
                        force_residual=False
                    )
        
        # Test that it works with input_shape provided
        for kernel_sizes in kernel_size_combinations:
            try:
                ResidualConvBlock(
                    num_conv_layers=len(kernel_sizes),
                    channels=[16] * len(kernel_sizes) + [32],
                    kernel_sizes=kernel_sizes,
                    strides=1,
                    paddings='same',
                    use_bn=True,
                    force_residual=False,
                    input_shape=(16, 64, 64)
                )
            except ValueError:
                self.fail(f"ResidualConvBlock with kernel_sizes={kernel_sizes} and input_shape should not raise ValueError")
                
        # Test that kernel_size=1 works without input_shape
        try:
            ResidualConvBlock(
                num_conv_layers=2,
                channels=[16, 16, 32],
                kernel_sizes=1,
                strides=1,
                paddings='same',
                use_bn=True,
                force_residual=False
            )
        except ValueError:
            self.fail("ResidualConvBlock with kernel_size=1 should not require input_shape")


    @unittest.skip("skip for now")
    def test_input_shape_validation(self):
        """Test that the forward method validates input shape when input_shape is specified"""
        # Create a block with input_shape specified
        input_shape = (16, 64, 64)
        block = ResidualConvBlock(
            num_conv_layers=2,
            channels=[16, 16, 32],
            kernel_sizes=3,
            strides=1,
            paddings='same',
            use_bn=True,
            force_residual=False,
            input_shape=input_shape
        )
        
        # Valid input should work
        valid_input = torch.randn(2, *input_shape)
        try:
            block(valid_input)
        except ValueError:
            self.fail("Forward pass should accept input with correct shape")
        
        # Invalid input should raise error
        invalid_shapes = [
            (2, 16, 32, 64),  # Wrong spatial dimensions
            (2, 8, 64, 64),   # Wrong channel dimension
            (2, 16, 64, 32)   # Wrong width
        ]
        
        for shape in invalid_shapes:
            with self.subTest(shape=shape):
                invalid_input = torch.randn(shape)
                with self.assertRaises(ValueError):
                    block(invalid_input)



    @unittest.skip("skip for now")
    def test_forward_md_true_fr_true(self):
        """
        Test forward pass with matching dimensions and force_residual=True
        
        In this case:
        - Input and output have the same number of channels
        - An adaptive layer with 1x1 kernel should be created
        - Output should be main_stream + residual_stream
        """
        for _ in range(100):
            block = self._generate_random_residual_conv_block(matching_dimensions=True, force_residual=True)
            block.eval()  # Set to eval mode for consistent outputs
            
            # Generate random input with valid shape
            batch_size = random.randint(1, 4)
            height = random.randint(256, 512)
            width = random.randint(256, 512)
            x = self._get_valid_input(block, batch_size=batch_size, height=height, width=width)
            
            # Check that adaptive layer has a 1x1 kernel
            self.assertIsNotNone(block._adaptive_layer)
            self.assertEqual(tuple(block._adaptive_layer.kernel_size), (1, 1), 
                            "Adaptive layer should have a 1x1 kernel when dimensions match and force_residual=True")
            
            # Verify forward pass computation
            with torch.no_grad():
                main_output = block._block(x)
                residual_output = block._adaptive_layer(x)
                expected_output = main_output + residual_output
                
                # Debug mode should return individual components
                main, residual, combined = block(x, debug=True)
                
                # Check that all outputs match expectations
                self.assertTrue(torch.allclose(main, main_output))
                self.assertTrue(torch.allclose(residual, residual_output))
                self.assertTrue(torch.allclose(combined, expected_output))
                
                # Regular forward should match combined output
                output = block(x)
                self.assertTrue(torch.allclose(output, expected_output))


    def test_forward_md_true_fr_false(self):
        """
        Test forward pass with matching dimensions and force_residual=False
        
        In this case:
        - Input and output have the same number of channels
        - No adaptive layer should be created (direct identity connection)
        - Output should be main_stream + input
        """
        for _ in range(100):
            block = self._generate_random_residual_conv_block(matching_dimensions=True, force_residual=False)
            block.eval()  # Set to eval mode for consistent outputs
            
            # Generate random input with valid shape
            batch_size = random.randint(1, 4)
            height = random.randint(256, 512)
            width = random.randint(256, 512)
            x = self._get_valid_input(block, batch_size=batch_size, height=height, width=width)
            
            # Check that no adaptive layer exists
            self.assertIsNone(block._adaptive_layer)
            
            # Verify forward pass computation
            with torch.no_grad():
                main_output = block._block(x)
                expected_output = main_output + x
                
                # Debug mode should return individual components
                main, residual, combined = block(x, debug=True)
                
                # Check that all outputs match expectations
                self.assertTrue(torch.allclose(main, main_output))
                self.assertTrue(torch.allclose(residual, x))
                self.assertTrue(torch.allclose(combined, expected_output))
                
                # Regular forward should match combined output
                output = block(x)
                self.assertTrue(torch.allclose(output, expected_output))


    @unittest.skip("skip for now")
    def test_forward_md_false_fr_true(self):
        """
        Test forward pass with non-matching dimensions and force_residual=True
        
        In this case:
        - Input and output have different numbers of channels
        - An adaptive layer must be created to match dimensions
        - Output should be main_stream + residual_stream
        """
        for _ in range(100):
            block = self._generate_random_residual_conv_block(matching_dimensions=False, force_residual=True)
            block.eval()  # Set to eval mode for consistent outputs
            
            # Generate random input with valid shape
            batch_size = random.randint(1, 4)
            height = random.randint(256, 512)
            width = random.randint(256, 512)
            x = self._get_valid_input(block, batch_size=batch_size, height=height, width=width)
            
            # Check that adaptive layer exists and the kernel size is different from 1x1
            self.assertIsNotNone(block._adaptive_layer)
            self.assertNotEqual(tuple(block._adaptive_layer.kernel_size), (1, 1))
            
            # Verify forward pass computation
            with torch.no_grad():
                main_output = block._block(x)
                residual_output = block._adaptive_layer(x)
                expected_output = main_output + residual_output
                
                # Debug mode should return individual components
                main, residual, combined = block(x, debug=True)
                
                # Check that all outputs match expectations
                self.assertTrue(torch.allclose(main, main_output))
                self.assertTrue(torch.allclose(residual, residual_output))
                self.assertTrue(torch.allclose(combined, expected_output))
                
                # Regular forward should match combined output
                output = block(x)
                self.assertTrue(torch.allclose(output, expected_output))

    
    @unittest.skip("skip for now")
    def test_forward_md_false_fr_false(self):
        """
        Test forward pass with non-matching dimensions and force_residual=False
        
        In this case:
        - Input and output have different numbers of channels
        - An adaptive layer must be created to match dimensions
        - Output should be main_stream + residual_stream
        """
        for _ in range(100):
            block = self._generate_random_residual_conv_block(matching_dimensions=False, force_residual=False)
            block.eval()  # Set to eval mode for consistent outputs
            
            # Generate random input with valid shape
            batch_size = random.randint(1, 4)
            height = random.randint(256, 512)
            width = random.randint(256, 512)
            x = self._get_valid_input(block, batch_size=batch_size, height=height, width=width)
            
            # Check that adaptive layer exists and is not a 1x1 kernel
            self.assertIsNotNone(block._adaptive_layer)
            self.assertNotEqual(tuple(block._adaptive_layer.kernel_size), (1, 1))

            # Verify forward pass computation
            with torch.no_grad():
                main_output = block._block(x)
                residual_output = block._adaptive_layer(x)
                expected_output = main_output + residual_output
                
                # Debug mode should return individual components
                main, residual, combined = block(x, debug=True)
                
                # Check that all outputs match expectations
                self.assertTrue(torch.allclose(main, main_output))
                self.assertTrue(torch.allclose(residual, residual_output))
                self.assertTrue(torch.allclose(combined, expected_output))
                
                # Regular forward should match combined output
                output = block(x)
                self.assertTrue(torch.allclose(output, expected_output))


if __name__ == "__main__":
    pu.seed_everything(42)
    unittest.main()
