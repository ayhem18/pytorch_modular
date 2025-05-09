import torch
import random
import unittest
from torch import nn
from typing import List, Tuple

import mypt.code_utils.pytorch_utils as pu
from mypt.building_blocks.conv_blocks.transpose_conv_block import TransposeConvBlock
from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser
from tests.custom_base_test import CustomModuleBaseTest


class TestTransposeConvBlocks(CustomModuleBaseTest):
    def setUp(self):
        self.dim_analyser = DimensionsAnalyser()

    def _generate_random_transpose_conv_block(self, 
                                             use_bn=True, 
                                             activation_after_each_layer=False, 
                                             activation=nn.ReLU,
                                             final_bn_layer=False) -> Tuple[TransposeConvBlock, int, List[int]]:
        num_layers = random.randint(1, 4)
        channels = [random.randint(1, 64) for _ in range(num_layers + 1)]
        # For transpose convs, we typically go from smaller to larger feature maps
        # So we'll reverse the channel list to go from more channels to fewer
        channels = sorted(channels, reverse=True)  
        kernel_sizes = random.randint(2, 5)  # Typical upsampling kernel sizes are >= 2
        
        return TransposeConvBlock(num_transpose_conv_layers=num_layers, 
                                 channels=channels, 
                                 kernel_sizes=kernel_sizes, 
                                 strides=2,  # Default upsampling stride
                                 paddings=1,
                                 output_paddings=0,
                                 use_bn=use_bn, 
                                 activation_after_each_layer=activation_after_each_layer, 
                                 activation=activation,
                                 final_bn_layer=final_bn_layer), num_layers, channels


    def test_transpose_conv_block_single_activation_no_bn(self):
        """Test that transpose conv block with single activation at the end and no batch norm works correctly"""
        # Case 1: Without final batch normalization
        with self.subTest("Without final batch normalization"):
            for _ in range(10):
                block, num_layers, channels = self._generate_random_transpose_conv_block(
                    use_bn=False, 
                    activation_after_each_layer=False, 
                    activation=nn.ReLU,
                    final_bn_layer=False
                )
                
                # Check that block was created correctly
                children = list(block.children())
                # We expect num_layers transpose_conv layers followed by 1 activation layer
                self.assertEqual(len(children), num_layers + 1)
                
                # Check that all except the last layer are ConvTranspose2d
                for i in range(num_layers):
                    self.assertIsInstance(children[i], nn.ConvTranspose2d)
                    self.assertEqual(children[i].in_channels, channels[i])
                    self.assertEqual(children[i].out_channels, channels[i+1])
                
                # Check the last layer is an activation
                self.assertIsInstance(children[-1], nn.ReLU)
        
        # Case 2: With final batch normalization
        with self.subTest("With final batch normalization"):
            for _ in range(10):
                block, num_layers, channels = self._generate_random_transpose_conv_block(
                    use_bn=False, 
                    activation_after_each_layer=False, 
                    activation=nn.ReLU,
                    final_bn_layer=True
                )
                
                # Check that block was created correctly
                children = list(block.children())
                # We expect num_layers conv layers + 1 final batch norm + 1 activation layer
                self.assertEqual(len(children), num_layers + 2)
                
                # Check that initial layers are ConvTranspose2d
                for i in range(num_layers):
                    self.assertIsInstance(children[i], nn.ConvTranspose2d)
                    self.assertEqual(children[i].in_channels, channels[i])
                    self.assertEqual(children[i].out_channels, channels[i+1])
                
                # Check for final batch norm before activation
                self.assertIsInstance(children[-2], nn.BatchNorm2d)
                self.assertEqual(children[-2].num_features, channels[-1])
                
                # Check the last layer is an activation
                self.assertIsInstance(children[-1], nn.ReLU)

    def test_transpose_conv_block_single_activation_bn(self):
        """Test that transpose conv block with single activation at the end and batch norm works correctly"""
        # Test with batch norm and without additional final BN (BN is added after each layer anyway)
        with self.subTest("With batch norm after each transpose conv"):
            for v in [True, False]:  # Test both final_bn_layer settings
                for _ in range(10):
                    block, num_layers, channels = self._generate_random_transpose_conv_block(
                        use_bn=True, 
                        activation_after_each_layer=False, 
                        activation=nn.ReLU,
                        final_bn_layer=v
                    )

                    # Check that block was created correctly
                    children = list(block.children())
                    # We expect (transpose_conv + bn) * num_layers + 1 activation layer
                    self.assertEqual(len(children), num_layers * 2 + 1)
                    
                    # Check the layers
                    for i in range(num_layers):
                        # Every even layer should be ConvTranspose2d
                        self.assertIsInstance(children[i*2], nn.ConvTranspose2d)
                        self.assertEqual(children[i*2].in_channels, channels[i])
                        self.assertEqual(children[i*2].out_channels, channels[i+1])
                        
                        # Every odd layer should be BatchNorm2d
                        self.assertIsInstance(children[i*2+1], nn.BatchNorm2d)
                        self.assertEqual(children[i*2+1].num_features, channels[i+1])
                    
                    # Check the last layer is an activation
                    self.assertIsInstance(children[-1], nn.ReLU)

    def test_transpose_conv_block_activation_after_each_layer_no_bn(self):
        """Test that transpose conv block with activation after each layer and no batch norm works correctly"""
        # Case 1: Without final batch normalization
        with self.subTest("Without final batch normalization"):
            for _ in range(10):
                block, num_layers, channels = self._generate_random_transpose_conv_block(
                    use_bn=False, 
                    activation_after_each_layer=True, 
                    activation=nn.ReLU,
                    final_bn_layer=False
                )
                
                # Check that block was created correctly
                children = list(block.children())
                # We expect (transpose_conv + activation) * num_layers layers
                self.assertEqual(len(children), num_layers * 2)
                
                # Check the layers
                for i in range(num_layers):
                    # Every even layer should be ConvTranspose2d
                    self.assertIsInstance(children[i*2], nn.ConvTranspose2d)
                    self.assertEqual(children[i*2].in_channels, channels[i])
                    self.assertEqual(children[i*2].out_channels, channels[i+1])
                    
                    # Every odd layer should be an activation
                    self.assertIsInstance(children[i*2+1], nn.ReLU)
        
        # Case 2: With final batch normalization
        with self.subTest("With final batch normalization"):
            for _ in range(10):
                block, num_layers, channels = self._generate_random_transpose_conv_block(
                    use_bn=False, 
                    activation_after_each_layer=True, 
                    activation=nn.ReLU,
                    final_bn_layer=True
                )
                
                # Check that block was created correctly
                children = list(block.children())
                # We expect (transpose_conv + activation) * (num_layers-1) + transpose_conv + bn + activation
                expected_layers = (num_layers-1) * 2 + 3
                self.assertEqual(len(children), expected_layers)
                
                # Check the regular layers (except the last transpose_conv)
                for i in range(num_layers-1):
                    # Every even layer should be ConvTranspose2d
                    self.assertIsInstance(children[i*2], nn.ConvTranspose2d)
                    self.assertEqual(children[i*2].in_channels, channels[i])
                    self.assertEqual(children[i*2].out_channels, channels[i+1])
                    
                    # Every odd layer should be an activation
                    self.assertIsInstance(children[i*2+1], nn.ReLU)
                
                # Check the final layers
                # The final transpose_conv
                self.assertIsInstance(children[expected_layers-3], nn.ConvTranspose2d)
                # The final bn
                self.assertIsInstance(children[expected_layers-2], nn.BatchNorm2d)
                # The final activation
                self.assertIsInstance(children[expected_layers-1], nn.ReLU)

    def test_transpose_conv_block_activation_after_each_layer_bn(self):
        """Test that transpose conv block with activation after each layer and batch norm works correctly"""
        # Case 1: Without extra final batch normalization (already included with use_bn=True)
        with self.subTest("With batch norm after each transpose conv"):
            for _ in range(10):
                block, num_layers, channels = self._generate_random_transpose_conv_block(
                    use_bn=True, 
                    activation_after_each_layer=True, 
                    activation=nn.ReLU,
                    final_bn_layer=False
                )
                
                # Check that block was created correctly
                children = list(block.children())
                # We expect (transpose_conv + bn + activation) * num_layers layers
                self.assertEqual(len(children), num_layers * 3)
                
                # Check the layers
                for i in range(num_layers):
                    # Check ConvTranspose2d layer
                    self.assertIsInstance(children[i*3], nn.ConvTranspose2d)
                    self.assertEqual(children[i*3].in_channels, channels[i])
                    self.assertEqual(children[i*3].out_channels, channels[i+1])
                    
                    # Check BatchNorm2d layer
                    self.assertIsInstance(children[i*3+1], nn.BatchNorm2d)
                    self.assertEqual(children[i*3+1].num_features, channels[i+1])
                    
                    # Check activation layer
                    self.assertIsInstance(children[i*3+2], nn.ReLU)
        
        # Case 2: With additional final batch normalization (redundant with use_bn=True)
        with self.subTest("With batch norm after each transpose conv and final_bn_layer=True (redundant)"):
            for _ in range(10):
                block, num_layers, channels = self._generate_random_transpose_conv_block(
                    use_bn=True, 
                    activation_after_each_layer=True, 
                    activation=nn.ReLU,
                    final_bn_layer=True
                )
                
                # Check that block was created correctly - should be same as above
                children = list(block.children())
                # We expect (transpose_conv + bn + activation) * num_layers layers
                self.assertEqual(len(children), num_layers * 3)
                
                # Check the layers (same as above)
                for i in range(num_layers):
                    # Check ConvTranspose2d layer
                    self.assertIsInstance(children[i*3], nn.ConvTranspose2d)
                    # Check BatchNorm2d layer
                    self.assertIsInstance(children[i*3+1], nn.BatchNorm2d)
                    # Check activation layer
                    self.assertIsInstance(children[i*3+2], nn.ReLU)


    # Helper method for CustomModuleBaseTest
    def _get_valid_input(self, batch_size:int, in_features:int) -> torch.Tensor:
        """Generate a valid input tensor for the transpose conv block tests"""
        return torch.randn(batch_size, in_features, 16, 16)  # Smaller spatial dims since we're upsampling

    # Custom module base tests
    def test_eval_mode(self):
        """Test that eval mode is correctly set across the transpose conv block"""
        with self.subTest("Test eval mode with ReLU activation"):
            for _ in range(10):
                activation = nn.ReLU
                block, _, _ = self._generate_random_transpose_conv_block(use_bn=True, activation_after_each_layer=True, activation=activation)   
                input_tensor = self._get_valid_input(random.randint(1, 10), block.channels[0])
                super()._test_eval_mode(block)

        with self.subTest("Test eval mode with Sigmoid activation"):
            for _ in range(10):
                activation = nn.Sigmoid
                block, _, _ = self._generate_random_transpose_conv_block(use_bn=True, activation_after_each_layer=False, activation=activation)   
                super()._test_eval_mode(block)

    def test_train_mode(self):
        """Test that train mode is correctly set across the transpose conv block"""
        with self.subTest("Test train mode with ReLU activation"):
            for _ in range(10):
                activation = nn.ReLU
                block, _, _ = self._generate_random_transpose_conv_block(use_bn=True, activation_after_each_layer=True, activation=activation)   
                super()._test_train_mode(block)

        with self.subTest("Test train mode with Sigmoid activation"):
            for _ in range(10):
                activation = nn.Sigmoid
                block, _, _ = self._generate_random_transpose_conv_block(use_bn=True, activation_after_each_layer=False, activation=activation)   
                super()._test_train_mode(block)

    def test_consistent_output_in_eval_mode(self):
        """Test that the transpose conv block produces consistent output in eval mode"""
        with self.subTest("Test consistent output in eval mode with ReLU activation"):
            for _ in range(10):
                activation = nn.ReLU
                block, _, _ = self._generate_random_transpose_conv_block(use_bn=True, activation_after_each_layer=True, activation=activation)   
                input_tensor = self._get_valid_input(random.randint(1, 10), block.channels[0])
                super()._test_consistent_output_in_eval_mode(block, input_tensor)

        with self.subTest("Test consistent output in eval mode with Sigmoid activation"):
            for _ in range(10):
                activation = nn.Sigmoid
                block, _, _ = self._generate_random_transpose_conv_block(use_bn=False, activation_after_each_layer=False, activation=activation)   
                input_tensor = self._get_valid_input(random.randint(1, 10), block.channels[0])
                super()._test_consistent_output_in_eval_mode(block, input_tensor)
        
    def test_batch_size_one_in_train_mode(self):
        """Test that the transpose conv block handles batch size 1 in train mode"""
        with self.subTest("Test batch size 1 in train mode with ReLU activation"):
            for _ in range(10):
                activation = nn.ReLU
                block, _, _ = self._generate_random_transpose_conv_block(use_bn=True, activation_after_each_layer=True, activation=activation)   
                input_tensor = self._get_valid_input(1, block.channels[0])
                super()._test_batch_size_one_in_train_mode(block, input_tensor)

        with self.subTest("Test batch size 1 in train mode with Sigmoid activation"):
            for _ in range(10):
                activation = nn.Sigmoid
                block, _, _ = self._generate_random_transpose_conv_block(use_bn=False, activation_after_each_layer=False, activation=activation)   
                input_tensor = self._get_valid_input(1, block.channels[0])
                super()._test_batch_size_one_in_train_mode(block, input_tensor)

    def test_batch_size_one_in_eval_mode(self):
        """Test that the transpose conv block handles batch size 1 in eval mode"""
        with self.subTest("Test batch size 1 in eval mode with ReLU activation"):
            for _ in range(10):
                activation = nn.ReLU
                block, _, _ = self._generate_random_transpose_conv_block(use_bn=True, activation_after_each_layer=True, activation=activation)   
                input_tensor = self._get_valid_input(1, block.channels[0])
                super()._test_batch_size_one_in_eval_mode(block, input_tensor)

        with self.subTest("Test batch size 1 in eval mode with Sigmoid activation"):
            for _ in range(10):
                activation = nn.Sigmoid
                block, _, _ = self._generate_random_transpose_conv_block(use_bn=False, activation_after_each_layer=False, activation=activation)   
                input_tensor = self._get_valid_input(1, block.channels[0])
                super()._test_batch_size_one_in_eval_mode(block, input_tensor)
    
    def test_named_parameters_length(self):
        """Test that named_parameters and parameters have the same length"""
        with self.subTest("Test named_parameters with ReLU activation"):
            for _ in range(10):
                activation = nn.ReLU
                block, _, _ = self._generate_random_transpose_conv_block(use_bn=True, activation_after_each_layer=True, activation=activation)   
                super()._test_named_parameters_length(block)

        with self.subTest("Test named_parameters with Sigmoid activation"):
            for _ in range(10):
                activation = nn.Sigmoid
                block, _, _ = self._generate_random_transpose_conv_block(use_bn=False, activation_after_each_layer=False, activation=activation)   
                super()._test_named_parameters_length(block)


if __name__ == '__main__':
    pu.seed_everything(69)
    unittest.main() 