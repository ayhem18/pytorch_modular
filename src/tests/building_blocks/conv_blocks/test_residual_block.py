import torch
import random
import unittest

from torch import nn
from random import randint as ri
from typing import Tuple

from tqdm import tqdm

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
                                            strides=None,
                                            input_shape=None):
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

        if strides is not None:
            if isinstance(strides, int):
                strides = [strides] * num_conv_layers
            else:
                assert len(strides) == num_conv_layers

        if strides is not None and any(s > 1 for s in strides):
            matching_dimensions=False

            if input_shape is None:
                input_shape = (16, 100, 100)    

        if strides is None:
            strides = random.choice(self.strides_options) 
            
            if strides != 1:
                matching_dimensions = False
                input_shape = (16, 100, 100)


        if matching_dimensions is None:
            matching_dimensions = random.choice([True, False])
            
        if force_residual is None:
            force_residual = random.choice([True, False])
            
        if activation is None:
            activation = random.choice(self.activation_modules)
        
        # Generate channel dimensions
        if input_shape is not None:
            in_channels = input_shape[0]
        else:
            in_channels = ri(*self.input_channels_range)
        
        if matching_dimensions:
            out_channels = in_channels
            paddings = 'same'
            strides = 1
        else:
            out_channels = ri(*self.output_channels_range)
            while out_channels == in_channels:
                out_channels = ri(*self.output_channels_range)
            
            # if the dimensions are not matching, then no need for padding
            paddings = 0


        # Create a list of channels for all layers
        channels = [in_channels] * num_conv_layers + [out_channels]

        kernel_sizes = random.choice(self.kernel_sizes_options)
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
            force_residual=force_residual,
            input_shape=input_shape
        )
    
    ########################## CustomModuleBaseTest tests ##########################
    
    @unittest.skip("passed!!")
    def test_eval_mode(self):
        """Test that the block can be set to evaluation mode"""
        for _ in tqdm(range(100), desc="Testing eval mode"):
            block = self._generate_random_residual_conv_block()
            super()._test_eval_mode(block)
    
    @unittest.skip("passed!!")
    def test_train_mode(self):
        """Test that the block can be set to training mode"""
        for _ in tqdm(range(100), desc="Testing train mode"):
            block = self._generate_random_residual_conv_block()
            super()._test_train_mode(block)
    
    @unittest.skip("passed!!")
    def test_consistent_output_without_dropout_bn(self):
        """Test consistent output for non-stochastic blocks"""
        for _ in tqdm(range(100), desc="Testing consistent output without dropout bn"):
            block = self._generate_random_residual_conv_block()
            
            if block._input_shape is not None:
                input_tensor = self._get_valid_input(block, batch_size=1, height=block._input_shape[1], width=block._input_shape[2])
            else:
                input_tensor = self._get_valid_input(block)

            super()._test_consistent_output_without_dropout_bn(block, input_tensor)
    
    @unittest.skip("passed!!")
    def test_consistent_output_in_eval_mode(self):
        """Test consistent output in evaluation mode"""
        for _ in range(100):
            block = self._generate_random_residual_conv_block()
            if block._input_shape is not None:
                input_tensor = self._get_valid_input(block, batch_size=1, height=block._input_shape[1], width=block._input_shape[2])
            else:   
                input_tensor = self._get_valid_input(block)
            super()._test_consistent_output_in_eval_mode(block, input_tensor)

    @unittest.skip("passed!!")
    def test_batch_size_one_in_train_mode(self):
        """Test handling of batch size 1 in training mode"""
        for _ in range(100):
            block = self._generate_random_residual_conv_block()
            if block._input_shape is not None:
                input_tensor = self._get_valid_input(block, batch_size=1, height=block._input_shape[1], width=block._input_shape[2])
            else:
                input_tensor = self._get_valid_input(block)
            super()._test_batch_size_one_in_train_mode(block, input_tensor)

    # @unittest.skip("skip for now")
    def test_batch_size_one_in_eval_mode(self):
        """Test handling of batch size 1 in evaluation mode"""
        for _ in range(100):
            block = self._generate_random_residual_conv_block()
            if block._input_shape is not None:
                input_tensor = self._get_valid_input(block, batch_size=1, height=block._input_shape[1], width=block._input_shape[2])
            else:
                input_tensor = self._get_valid_input(block)
            super()._test_batch_size_one_in_eval_mode(block, input_tensor)

    # @unittest.skip("skip for now")
    def test_named_parameters_length(self):
        """Test that named_parameters and parameters have the same length"""
        for _ in range(100):
            block = self._generate_random_residual_conv_block()
            super()._test_named_parameters_length(block)
    
    # @unittest.skip("skip for now")
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
                strides=[1, 2, 1],
                paddings='same',
                use_bn=True,
                force_residual=False
            )
            
        # Test with list containing strides > 1
        strides_combinations = [
            [3, 1],  # First layer has stride > 1
            [1, 3],  # Second layer has stride > 1
            [3, 5],  # All layers have stride > 1
            [5, 7]   # All layers have larger stride
        ]
        
        for strides in strides_combinations:
            with self.subTest(strides=strides):
                with self.assertRaises(ValueError):
                    ResidualConvBlock(
                        num_conv_layers=len(strides),
                        channels=[16] * len(strides) + [32],
                        kernel_sizes=[1] * len(strides),
                        strides=strides,
                        paddings='same',
                        use_bn=True,
                        force_residual=False
                    )
        
        # Test that it works with input_shape provided
        for strides in strides_combinations:
            try:
                ResidualConvBlock(
                    num_conv_layers=len(strides),
                    channels=[16] * len(strides) + [32],
                    kernel_sizes=[1] * len(strides),
                    strides=strides,
                    paddings='same',
                    use_bn=True,
                    force_residual=False,
                    input_shape=(16, 100, 100)
                )
            except ValueError:
                self.fail(f"ResidualConvBlock with strides={strides} and input_shape should not raise ValueError")
                
        # Test that kernel_size=1 works without input_shape
        try:
            for k in range(1, 11, 2):
                ResidualConvBlock(
                    num_conv_layers=2,
                    channels=[16, 16, 32],
                    kernel_sizes=k,
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
        input_shape = (16, 100, 100)
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
        

        for _ in range(100):
            # change one of the dimensions of the input shape
            shape = input_shape.copy()
            i = random.randint(0, 2) 
            shape[i] = random.randint(2, input_shape[i] - 1)
            
            with self.subTest(shape=shape):
                invalid_input = torch.randn(2, *shape)
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
            
            # matching dimensions does not work with strides > 1
            block = self._generate_random_residual_conv_block(matching_dimensions=True, force_residual=True, strides=1)
            block.eval()
            # Generate random input with valid shape
            batch_size = random.randint(1, 4)
            height = random.randint(100, 150)
            width = random.randint(100, 150)

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

    @unittest.skip("skip for now")
    def test_forward_md_true_fr_false(self):
        """
        Test forward pass with matching dimensions and force_residual=False
        
        In this case:
        - Input and output have the same number of channels
        - No adaptive layer should be created (direct identity connection)
        - Output should be main_stream + input
        """
        for _ in range(100):
            block = self._generate_random_residual_conv_block(matching_dimensions=True, force_residual=False, strides=1)
            block.eval()  # Set to eval mode for consistent outputs
            
            # Generate random input with valid shape
            batch_size = random.randint(1, 4)
            height = random.randint(100, 150)
            width = random.randint(100, 150)
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
            # matching dimensions is set to False; then we can pass input_shape and strides > 1

            blocks_inputs = [None, None]

            b1 = self._generate_random_residual_conv_block(matching_dimensions=False, force_residual=True, strides=1)
            b1.eval()  # Set to eval mode for consistent outputs

            batch_size = random.randint(1, 4)
            height = random.randint(100, 150)
            width = random.randint(100, 150)
            x = self._get_valid_input(b1, batch_size=batch_size, height=height, width=width)
            blocks_inputs[0] = (b1, x)

            # generate the second block with a larger input shape
            batch_size = random.randint(1, 4)
            height = random.randint(100, 150)
            width = random.randint(100, 150)

            b2 = self._generate_random_residual_conv_block(matching_dimensions=False, force_residual=True, strides=2, input_shape=(16, height, width))
            b2.eval()  # Set to eval mode for consistent outputs

            x = self._get_valid_input(b2, batch_size=batch_size, height=height, width=width)
            blocks_inputs[1] = (b2, x)

            for block, x in blocks_inputs:
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
            blocks_inputs = [None, None]

            b1 = self._generate_random_residual_conv_block(matching_dimensions=False, force_residual=True, strides=1)
            b1.eval()  # Set to eval mode for consistent outputs

            batch_size = random.randint(1, 4)
            height = random.randint(100, 150)
            width = random.randint(100, 150)
            x = self._get_valid_input(b1, batch_size=batch_size, height=height, width=width)
            blocks_inputs[0] = (b1, x)

            # generate the second block with a larger input shape
            batch_size = random.randint(1, 4)
            height = random.randint(100, 150)
            width = random.randint(100, 150)

            b2 = self._generate_random_residual_conv_block(matching_dimensions=False, force_residual=True, strides=2, input_shape=(16, height, width))
            b2.eval()  # Set to eval mode for consistent outputs

            x = self._get_valid_input(b2, batch_size=batch_size, height=height, width=width)
            blocks_inputs[1] = (b2, x)


            for block, x in blocks_inputs:
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
