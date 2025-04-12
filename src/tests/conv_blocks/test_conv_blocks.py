import unittest
import torch
import random
from torch import nn

import mypt.code_utilities.pytorch_utilities as pu
from mypt.convBlocks.convBlock import ConvBlock
from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser


class TestConvBlocks(unittest.TestCase):
    def setUp(self):
        self.dim_analyser = DimensionsAnalyser()

    def test_conv_block_single_activation_no_bn(self):
        """Test that conv block with single activation at the end and no batch norm works correctly"""
        for _ in range(100):  # Test multiple random configurations
            # Generate random but valid parameters
            num_layers = random.randint(1, 4)
            channels = [random.randint(1, 64) for _ in range(num_layers + 1)]
            kernel_sizes = random.randint(1, 5)
            
            # Create block with single activation and no batch norm
            block = ConvBlock(
                num_conv_layers=num_layers,
                channels=channels,
                kernel_sizes=kernel_sizes,
                use_bn=False,
                activation_after_each_layer=False,
                activation=nn.ReLU
            )
            
            # Check that block was created correctly
            children = list(block.children())
            # We expect num_layers conv layers followed by 1 activation layer
            self.assertEqual(len(children), num_layers + 1)
            
            # Check that all except the last layer are Conv2d
            for i in range(num_layers):
                self.assertIsInstance(children[i], nn.Conv2d)
                self.assertEqual(children[i].in_channels, channels[i])
                self.assertEqual(children[i].out_channels, channels[i+1])
            
            # Check the last layer is an activation
            self.assertIsInstance(children[-1], nn.ReLU)

    def test_conv_block_single_activation_bn(self):
        """Test that conv block with single activation at the end and batch norm works correctly"""
        for _ in range(100):  # Test multiple random configurations
            # Generate random but valid parameters
            num_layers = random.randint(1, 4)
            channels = [random.randint(1, 64) for _ in range(num_layers + 1)]
            kernel_sizes = [2 * random.randint(1, 5) + 1 for _ in range(num_layers)] # have odd kernel sizes in general
            
            # Create block with single activation and batch norm
            block = ConvBlock(
                num_conv_layers=num_layers,
                channels=channels,
                kernel_sizes=kernel_sizes,
                use_bn=True,
                activation_after_each_layer=False,
                activation=nn.ReLU
            )
            
            # Check that block was created correctly
            children = list(block.children())
            # We expect (conv + bn) * num_layers + 1 activation layer
            self.assertEqual(len(children), num_layers * 2 + 1)
            
            # Check the layers
            for i in range(num_layers):
                # Every even layer should be Conv2d
                self.assertIsInstance(children[i*2], nn.Conv2d)
                self.assertEqual(children[i*2].in_channels, channels[i])
                self.assertEqual(children[i*2].out_channels, channels[i+1])
                
                # Every odd layer should be BatchNorm2d
                self.assertIsInstance(children[i*2+1], nn.BatchNorm2d)
                self.assertEqual(children[i*2+1].num_features, channels[i+1])
            
            # Check the last layer is an activation
            self.assertIsInstance(children[-1], nn.ReLU)

    def test_conv_block_activation_after_each_layer_no_bn(self):
        """Test that conv block with activation after each layer and no batch norm works correctly"""
        for _ in range(100):  # Test multiple random configurations
            # Generate random but valid parameters
            num_layers = random.randint(1, 4)
            channels = [random.randint(1, 64) for _ in range(num_layers + 1)]
            kernel_sizes = [2 * random.randint(1, 5) + 1 for _ in range(num_layers)] # have odd kernel sizes in general
            
            # Create block with activation after each layer and no batch norm
            block = ConvBlock(
                num_conv_layers=num_layers,
                channels=channels,
                kernel_sizes=kernel_sizes,
                use_bn=False,
                activation_after_each_layer=True,
                activation=nn.ReLU
            )
            
            # Check that block was created correctly
            children = list(block.children())
            # We expect (conv + activation) * num_layers layers
            self.assertEqual(len(children), num_layers * 2)
            
            # Check the layers
            for i in range(num_layers):
                # Every even layer should be Conv2d
                self.assertIsInstance(children[i*2], nn.Conv2d)
                self.assertEqual(children[i*2].in_channels, channels[i])
                self.assertEqual(children[i*2].out_channels, channels[i+1])
                
                # Every odd layer should be activation
                self.assertIsInstance(children[i*2+1], nn.ReLU)

    def test_conv_block_activation_after_each_layer_bn(self):
        """Test that conv block with activation after each layer and batch norm works correctly"""
        for _ in range(100):  # Test multiple random configurations
            # Generate random but valid parameters
            num_layers = random.randint(1, 4)
            channels = [random.randint(1, 64) for _ in range(num_layers + 1)]
            kernel_sizes = [2 * random.randint(1, 5) + 1 for _ in range(num_layers)] # have odd kernel sizes in general
            
            # Create block with activation after each layer and batch norm
            block = ConvBlock(
                num_conv_layers=num_layers,
                channels=channels,
                kernel_sizes=kernel_sizes,
                use_bn=True,
                activation_after_each_layer=True,
                activation=nn.ReLU
            )
            
            # Check that block was created correctly
            children = list(block.children())
            # We expect (conv + bn + activation) * num_layers layers
            self.assertEqual(len(children), num_layers * 3)
            
            # Check the layers
            for i in range(num_layers):
                # Check Conv2d layer
                self.assertIsInstance(children[i*3], nn.Conv2d)
                self.assertEqual(children[i*3].in_channels, channels[i])
                self.assertEqual(children[i*3].out_channels, channels[i+1])
                
                # Check BatchNorm2d layer
                self.assertIsInstance(children[i*3+1], nn.BatchNorm2d)
                self.assertEqual(children[i*3+1].num_features, channels[i+1])
                
                # Check activation layer
                self.assertIsInstance(children[i*3+2], nn.ReLU)

    def test_forward_pass_shape(self):
        """Test that forward pass produces output with expected shape"""
        for _ in range(1000):  # Test multiple random configurations
            # Generate random but valid parameters
            num_layers = random.randint(1, 4)
            channels = [random.randint(1, 64) for _ in range(num_layers + 1)]
            kernel_sizes = [2 * random.randint(1, 5) + 1 for _ in range(num_layers)] # have odd kernel sizes in general
            batch_size = random.randint(1, 8)
            img_size = random.randint(16, 64)  # Small-ish image for speed
            
            # Create random input tensor
            input_tensor = torch.randn(batch_size, channels[0], img_size, img_size)
            
            # Create block
            block = ConvBlock(
                num_conv_layers=num_layers,
                channels=channels,
                kernel_sizes=kernel_sizes,
                use_bn=True,
                activation_after_each_layer=True,
                activation=nn.ReLU
            )
            
            # Get actual output shape
            output = block(input_tensor)
            actual_shape = tuple(output.shape)
            
            # Get predicted output shape using DimensionsAnalyser
            expected_shape = self.dim_analyser.analyse_dimensions(
                input_shape=tuple(input_tensor.shape),
                net=block
            )
            
            # Compare shapes
            self.assertEqual(actual_shape, expected_shape, 
                            f"Mismatch in output shape: got {actual_shape}, expected {expected_shape}")


if __name__ == '__main__':
    pu.seed_everything(69)
    unittest.main()