import torch
import random
import unittest

from torch import nn
from tqdm import tqdm
from random import randint as ri
from typing import List, Optional, Tuple, Union


import mypt.code_utils.pytorch_utils as pu

from tests.custom_base_test import CustomModuleBaseTest
from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser
from mypt.nets.conv_nets.adaptive_unet.uni_multi_residual_conv import UniformMultiResidualNet
from mypt.building_blocks.conv_blocks.residual.residual_conv_block import ResidualConvBlock
from mypt.building_blocks.conv_blocks.adaptive.conv_block_design.conv_design_utils import compute_log_linear_sequence


@unittest.skip("skip for now")
class TestUniformMultiResidualNet(CustomModuleBaseTest):
    """Test class for UniformMultiResidualNet implementation"""
    
    def setUp(self):
        """Initialize test parameters"""
        self.dim_analyser = DimensionsAnalyser()
        
        # Define standard activation functions for testing
        self.activation_modules = [nn.ReLU(), nn.LeakyReLU(), nn.Tanh()]
        
        # Define common test parameters
        self.in_channels_range = (1, 16)
        self.out_channels_range = (16, 32)
        self.num_blocks_range = (2, 4)
        self.num_layers_per_block_range = (2, 3)
        self.kernel_sizes_options = [1, 3, 5, 7]
        self.strides_options = [1, 2]
        self.padding_options = ['same', 0, 1]
    
    def _get_valid_input(self, network=None, batch_size=2, height=64, width=64) -> torch.Tensor:
        """Generate a random input tensor with the correct shape for the given network"""
        if network is None:
            in_channels = ri(*self.in_channels_range)
        else:
            in_channels = network.in_channels
            
            # If network has input_shape defined, use those dimensions
            if hasattr(network, 'input_shape') and network.input_shape is not None:
                in_channels, height, width = network.input_shape
        
        return torch.randn(batch_size, in_channels, height, width)
    
    def _generate_random_uniform_multi_residual_net(self, 
                                                    strides: Union[int, List[int]], 
                                                    input_shape: Optional[Tuple[int, int, int]] = None,
                                                    layers_per_block: Optional[Union[int, List[int]]] = None,
                                                    num_blocks: Optional[int] = None,
                                                    ):
        """Generate a random UniformMultiResidualNet for testing"""

        if isinstance(strides, List):
            num_blocks = len(strides) 
        else:
            if num_blocks is None: 
                num_blocks = ri(*self.num_blocks_range)
        
        if isinstance(strides, int):
            strides = [strides] * num_blocks

        if any(s > 1 for s in strides) and input_shape is None:
            raise ValueError("input_shape must be provided when using strides > 1")

        if input_shape is not None:
            in_channels, _, _ = input_shape
        else:
            in_channels = ri(*self.in_channels_range)

        out_channels = ri(*self.out_channels_range)
        layers_per_block = ri(*self.num_layers_per_block_range) if layers_per_block is None else layers_per_block
        
        # Random configurations
        use_bn = random.choice([True, False])
        activation_after_each_layer = random.choice([True, False])
        final_bn_layer = random.choice([True, False])
        force_residual = random.choice([True, False])
        
        # Choose a random activation function
        activation = random.choice(self.activation_modules)
        
        # Choose kernel sizes - either all 1 or meaningful values
        kernel_sizes = random.choice(self.kernel_sizes_options)
        
        if any(s > 1 for s in strides):
            paddings = 0
        else:
            paddings = random.choice(self.padding_options)
            if paddings != 'same':
                paddings = min(paddings, kernel_sizes - 1)
        
        # Create and return the network
        return UniformMultiResidualNet(
            num_conv_blocks=num_blocks,
            in_channels=in_channels,
            out_channels=out_channels,
            conv_layers_per_block=layers_per_block,
            channels=None,  # Use auto-generated log-linear sequence
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
    @unittest.skip("passed")
    def test_eval_mode(self):
        """Test that the network can be set to evaluation mode"""
        for _ in tqdm(range(50), desc="Testing eval mode: strides=1, input_shape=(8, 64, 64)"):
            network = self._generate_random_uniform_multi_residual_net(
                strides=1,
                input_shape=(8, 64, 64),
                num_blocks=3,
                layers_per_block=3
            )
            super()._test_eval_mode(network)

        for _ in tqdm(range(50), desc="Testing eval mode: strides=1, "):
            network = self._generate_random_uniform_multi_residual_net(
                strides=1,
            )
            super()._test_eval_mode(network) 

        for _ in tqdm(range(50), desc="Testing eval mode: strides=2, input_shape=(8, 64, 64)"):
            network = self._generate_random_uniform_multi_residual_net(
                strides=2,
                input_shape=(8, 100, 100),  
                layers_per_block=2,
                num_blocks=2
            )
            super()._test_eval_mode(network)

    @unittest.skip("passed")
    def test_train_mode(self):
        """Test that the network can be set to training mode"""
        for _ in tqdm(range(50), desc="Testing train mode: strides=1, input_shape=(8, 64, 64)"):
            network = self._generate_random_uniform_multi_residual_net(
                strides=1,
                input_shape=(8, 64, 64),
                num_blocks=3,
                layers_per_block=3
            )
            super()._test_train_mode(network)

        for _ in tqdm(range(50), desc="Testing train mode: strides=1, "):
            network = self._generate_random_uniform_multi_residual_net(
                strides=1,
            )
            super()._test_train_mode(network)

        for _ in tqdm(range(50), desc="Testing train mode: strides=2, input_shape=(8, 64, 64)"):
            network = self._generate_random_uniform_multi_residual_net(
                strides=2,
                input_shape=(8, 100, 100),
                layers_per_block=2,
                num_blocks=2
            )
            super()._test_train_mode(network)

    @unittest.skip("passed")
    def test_consistent_output_without_dropout_bn(self):
        """Test consistent output for non-stochastic networks"""
        for _ in tqdm(range(50), desc="Testing consistent output without dropout/bn: strides=1, input_shape=(8, 64, 64)"):
            network = self._generate_random_uniform_multi_residual_net(
                strides=1,
                input_shape=(8, 64, 64)
            )
            input_tensor = self._get_valid_input(network, height=64, width=64)
            super()._test_consistent_output_without_dropout_bn(network, input_tensor)

        for _ in tqdm(range(50), desc="Testing consistent output without dropout/bn: strides=1, "):
            network = self._generate_random_uniform_multi_residual_net(
                strides=1,
            )
            input_tensor = self._get_valid_input(network, height=64, width=64)
            super()._test_consistent_output_without_dropout_bn(network, input_tensor)

        for _ in tqdm(range(50), desc="Testing consistent output without dropout/bn: strides=2, input_shape=(8, 64,     64)"):
            network = self._generate_random_uniform_multi_residual_net(
                strides=2,
                input_shape=(8, 100, 100),
                layers_per_block=2,
                num_blocks=2
            )
            input_tensor = self._get_valid_input(network, height=100, width=100)
            super()._test_consistent_output_without_dropout_bn(network, input_tensor)

    
    @unittest.skip("passed")
    def test_consistent_output_in_eval_mode(self):
        """Test consistent output in evaluation mode"""
        for _ in tqdm(range(50), desc="Testing consistent output in eval mode: strides=1, input_shape=(8, 64, 64)"):
            network = self._generate_random_uniform_multi_residual_net(
                strides=1,
                input_shape=(8, 64, 64)
            )
            input_tensor = self._get_valid_input(network, height=64, width=64)
            super()._test_consistent_output_in_eval_mode(network, input_tensor)

        for _ in tqdm(range(50), desc="Testing consistent output in eval mode: strides=1, "):
            network = self._generate_random_uniform_multi_residual_net(
                strides=1,
            )
            input_tensor = self._get_valid_input(network, height=64, width=64)  
            super()._test_consistent_output_in_eval_mode(network, input_tensor)

        for _ in tqdm(range(50), desc="Testing consistent output in eval mode: strides=2, input_shape=(8, 64, 64)"):
            network = self._generate_random_uniform_multi_residual_net(
                strides=2,
                input_shape=(8, 100, 100),
                layers_per_block=2,
                num_blocks=2
            )
            input_tensor = self._get_valid_input(network, height=100, width=100)
            super()._test_consistent_output_in_eval_mode(network, input_tensor)


    @unittest.skip("passed")
    def test_batch_size_one_in_train_mode(self):
        """Test handling of batch size 1 in training mode"""
        for _ in tqdm(range(50), desc="Testing batch size 1 in train mode: strides=1, input_shape=(8, 64, 64)"):
            network = self._generate_random_uniform_multi_residual_net(
                strides=1,
                input_shape=(8, 64, 64)
            )
            input_tensor = self._get_valid_input(network, batch_size=1)
            super()._test_batch_size_one_in_train_mode(network, input_tensor)

        for _ in tqdm(range(50), desc="Testing batch size 1 in train mode: strides=1, "):
            network = self._generate_random_uniform_multi_residual_net(
                strides=1,
            )
            input_tensor = self._get_valid_input(network, batch_size=1)
            super()._test_batch_size_one_in_train_mode(network, input_tensor)   

        for _ in tqdm(range(50), desc="Testing batch size 1 in train mode: strides=2, "):
            network = self._generate_random_uniform_multi_residual_net(
                strides=2,
                input_shape=(8, 100, 100),
                layers_per_block=2,
                num_blocks=2
            )   
            input_tensor = self._get_valid_input(network, batch_size=1)
            super()._test_batch_size_one_in_train_mode(network, input_tensor)



    @unittest.skip("passed")
    def test_batch_size_one_in_eval_mode(self):
        """Test handling of batch size 1 in evaluation mode"""
        for _ in tqdm(range(50), desc="Testing batch size 1 in eval mode: strides=1, input_shape=(8, 64, 64)"):
            network = self._generate_random_uniform_multi_residual_net(
                strides=1,
                input_shape=(8, 64, 64)
            )
            input_tensor = self._get_valid_input(network, batch_size=1)
            super()._test_batch_size_one_in_eval_mode(network, input_tensor)

        for _ in tqdm(range(50), desc="Testing batch size 1 in eval mode: strides=1, "):
            network = self._generate_random_uniform_multi_residual_net(
                strides=1,
            )
            input_tensor = self._get_valid_input(network, batch_size=1)
            super()._test_batch_size_one_in_eval_mode(network, input_tensor)        

        for _ in tqdm(range(50), desc="Testing batch size 1 in eval mode: strides=2, input_shape=(8, 64, 64)"):
            network = self._generate_random_uniform_multi_residual_net(
                strides=2,
                input_shape=(8, 100, 100),
                layers_per_block=2,
                num_blocks=2
            )   
            input_tensor = self._get_valid_input(network, batch_size=1)
            super()._test_batch_size_one_in_eval_mode(network, input_tensor)



    @unittest.skip("passed")
    def test_named_parameters_length(self):
        """Test that named_parameters and parameters have the same length"""
        for _ in tqdm(range(50), desc="Testing named_parameters and parameters have the same length"):
            network = self._generate_random_uniform_multi_residual_net(
                strides=1,  
                input_shape=(8, 64, 64)
            )
            super()._test_named_parameters_length(network)

        for _ in tqdm(range(50), desc="Testing named_parameters and parameters have the same length: strides=1, "):
            network = self._generate_random_uniform_multi_residual_net(
                strides=1,
            )
            super()._test_named_parameters_length(network)

        for _ in tqdm(range(50), desc="Testing named_parameters and parameters have the same length: strides=2, input_shape=(8, 64, 64)"):
            network = self._generate_random_uniform_multi_residual_net(
                strides=2,
                input_shape=(8, 100, 100),
                layers_per_block=2,
                num_blocks=2
            )
            super()._test_named_parameters_length(network)

    
            
    # @unittest.skip("passed")
    def test_to_device(self):
        """Test device transfer functionality"""
        for _ in tqdm(range(5), desc="Testing device transfer functionality: strides=1, input_shape=(8, 64, 64)"):  # Fewer tests as this is expensive
            network = self._generate_random_uniform_multi_residual_net(
                strides=1,
                input_shape=(8, 64, 64)
            )
            input_tensor = self._get_valid_input(network, height=64, width=64)
            super()._test_to_device(network, input_tensor)

        for _ in tqdm(range(5), desc="Testing device transfer functionality: strides=1, "):
            network = self._generate_random_uniform_multi_residual_net(
                strides=1,
            )
            input_tensor = self._get_valid_input(network, height=100, width=100)
            super()._test_to_device(network, input_tensor)
        
        for _ in tqdm(range(5), desc="Testing device transfer functionality: strides=2, input_shape=(8, 64, 64)"):
            network = self._generate_random_uniform_multi_residual_net(
                strides=2,
                input_shape=(8, 100, 100),
                layers_per_block=2,
                num_blocks=2
            )
            input_tensor = self._get_valid_input(network, height=100, width=100)
            super()._test_to_device(network, input_tensor)

    ########################## UniformMultiResidualNet specific tests ##########################
    @unittest.skip("passed")
    def test_strides_validation(self):
        """Test that the network requires input_shape when strides > 1"""
        # Test with strides > 1 without input_shape (should raise ValueError)
        with self.assertRaises(ValueError):
            UniformMultiResidualNet(
                num_conv_blocks=3,
                in_channels=8,
                out_channels=16,
                conv_layers_per_block=2,
                kernel_sizes=3,
                strides=2,  # strides > 1
                paddings=0,
                use_bn=True,
                input_shape=None  # No input_shape
            )
        
        # Test with strides > 1 with input_shape (should work)
        try:
            network = UniformMultiResidualNet(
                num_conv_blocks=3,
                in_channels=8,
                out_channels=16,
                conv_layers_per_block=2,
                kernel_sizes=3,
                strides=2,  # strides > 1
                paddings=0,
                use_bn=True,
                input_shape=(8, 128, 128)  # pass a large input_shape
            )
        except ValueError:
            self.fail("UniformMultiResidualNet with strides > 1 and valid input_shape should not raise ValueError")
        
        # Test with strides = 1 without input_shape (should work regardless of kernel_size)
        try:
            for k in [1, 3, 5]:
                UniformMultiResidualNet(
                    num_conv_blocks=3,
                    in_channels=8,
                    out_channels=16,
                    conv_layers_per_block=2,
                    kernel_sizes=k,
                    strides=1,  # strides = 1
                    paddings=0,
                    use_bn=True,
                    input_shape=None  # No input_shape
                )
        except ValueError:
            self.fail("UniformMultiResidualNet with strides=1 should not require input_shape")
    
    @unittest.skip("passed")
    def test_network_structure(self):
        """Test that the network structure is correctly created"""
        for _ in range(50):
            # Create networks with different numbers of blocks and layers
            num_blocks = ri(*self.num_blocks_range)
            layers_per_block = ri(*self.num_layers_per_block_range)
            
            network = UniformMultiResidualNet(
                num_conv_blocks=num_blocks,
                in_channels=8,
                out_channels=16,
                conv_layers_per_block=layers_per_block,
                kernel_sizes=3,
                strides=1,
                paddings='same',
                use_bn=True,
                input_shape=(8, 64, 64)
            )
            
            # Check that the number of blocks is correct
            block_children = list(network._block.children())
            self.assertEqual(len(block_children), num_blocks, 
                         f"Expected {num_blocks} blocks, got {len(block_children)}")
            
            # Check that all blocks are ResidualConvBlocks
            for i, block in enumerate(block_children):
                self.assertIsInstance(block, ResidualConvBlock, 
                                   f"Block {i} is not a ResidualConvBlock")
                
                # Check that each block has the correct number of layers
                if isinstance(layers_per_block, int):
                    expected_layers = layers_per_block
                else:
                    expected_layers = layers_per_block[i]
                
                self.assertEqual(block._num_conv_layers, expected_layers,
                             f"Block {i} has {block._num_conv_layers} layers, expected {expected_layers}")
                
                # Check that each block has input_shape set
                self.assertIsNotNone(block._input_shape, 
                                  f"Block {i} does not have input_shape set")

    # @unittest.skip("passed")
    def test_channel_progression(self):
        """Test that channels follow the expected progression"""
        for _ in range(50):
            num_blocks = ri(*self.num_blocks_range)
            in_channels = ri(*self.in_channels_range)
            out_channels = ri(*self.out_channels_range)
            
            # Create a network with auto-generated channels
            network = UniformMultiResidualNet(
                num_conv_blocks=num_blocks,
                in_channels=in_channels,
                out_channels=out_channels,
                conv_layers_per_block=2,
                kernel_sizes=3,
                strides=1,
                paddings='same',
                use_bn=True,
                input_shape=(in_channels, 64, 64)
            )
            
            # Calculate the expected channel progression
            expected_channels = compute_log_linear_sequence(in_channels, out_channels, num_blocks + 1)
            
            # Check that the stored channels match
            self.assertEqual(network.channels, expected_channels,
                         f"Channel progression mismatch: got {network.channels}, expected {expected_channels}")
            
            # Check that each block has the correct channels
            blocks = list(network._block.children())
            for i, block in enumerate(blocks):
                block_input_channels = block._channels[0]
                block_output_channels = block._channels[-1]
                
                self.assertEqual(block_input_channels, expected_channels[i],
                             f"Block {i} input channels: got {block_input_channels}, expected {expected_channels[i]}")
                
                self.assertEqual(block_output_channels, expected_channels[i+1],
                             f"Block {i} output channels: got {block_output_channels}, expected {expected_channels[i+1]}")

    @unittest.skip("passed")
    def test_normalize_parameter(self):
        """Test the _normalize_parameter method"""
        network = UniformMultiResidualNet(
            num_conv_blocks=3,
            in_channels=8,
            out_channels=16,
            conv_layers_per_block=2,
            kernel_sizes=3,
            strides=1,
            paddings='same',
            use_bn=True,
            input_shape=(8, 64, 64)
        )
        
        # Test with a single value
        param = 5
        normalized = network._normalize_parameter(param, 3, "test_param")
        self.assertEqual(normalized, [5, 5, 5])
        
        # Test with a list of the right length
        param = [1, 2, 3]
        normalized = network._normalize_parameter(param, 3, "test_param")
        self.assertEqual(normalized, [1, 2, 3])
        
        # Test with a single-element list
        param = [7]
        normalized = network._normalize_parameter(param, 3, "test_param")
        self.assertEqual(normalized, [7, 7, 7])
        
        # Test with a list of the wrong length
        param = [1, 2]
        with self.assertRaises(ValueError):
            normalized = network._normalize_parameter(param, 3, "test_param")
        
        # Test with None
        param = None
        normalized = network._normalize_parameter(param, 3, "test_param")
        self.assertEqual(normalized, [None, None, None])

    @unittest.skip("passed")
    def test_forward_pass_different_input_shapes(self):
        """Test forward pass with different input shapes"""
        input_shapes = [
            (8, 64, 64),
            (16, 128, 128),
            (3, 224, 224)
        ]
        
        for input_shape in input_shapes:
            with self.subTest(input_shape=input_shape):
                network = UniformMultiResidualNet(
                    num_conv_blocks=3,
                    in_channels=input_shape[0],
                    out_channels=32,
                    conv_layers_per_block=2,
                    kernel_sizes=3,
                    strides=1,
                    paddings='same',
                    use_bn=True,
                    input_shape=input_shape
                )
                
                # Create input tensor
                batch_size = 2
                x = torch.randn(batch_size, *input_shape)
                
                # Forward pass
                output = network(x)
                
                # Check output shape - channels should be out_channels, spatial dims may change
                self.assertEqual(output.shape[0], batch_size)
                self.assertEqual(output.shape[1], 32)  # out_channels

    @unittest.skip("passed")
    def test_input_shape_validation(self):
        """Test that the forward method validates input shape when input_shape is specified"""
        # Create a network with specific input_shape
        input_shape = (16, 64, 64)
        network = UniformMultiResidualNet(
            num_conv_blocks=3,
            in_channels=input_shape[0],
            out_channels=32,
            conv_layers_per_block=2,
            kernel_sizes=3,
            strides=1,
            paddings='same',
            use_bn=True,
            input_shape=input_shape
        )
        
        # Valid input should work
        valid_input = torch.randn(2, *input_shape)
        try:
            network(valid_input)
        except ValueError:
            self.fail("Forward pass should accept input with correct shape")
        
        # Invalid input shape should raise ValueError
        invalid_shapes = [
            (2, 8, 64, 64),    # Wrong number of channels
            (2, 16, 32, 64),   # Wrong height
            (2, 16, 64, 32),   # Wrong width
            (2, 16, 32, 32)    # Both height and width wrong
        ]
        
        for shape in invalid_shapes:
            with self.subTest(shape=shape):
                invalid_input = torch.randn(shape)
                with self.assertRaises(ValueError):
                    network(invalid_input)



if __name__ == '__main__':
    pu.seed_everything(42)
    unittest.main() 