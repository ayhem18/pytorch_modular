import torch
import random
import unittest
import numpy as np

from torch import nn
from tqdm import tqdm
from typing import List, Tuple, Optional, Union

from tests.custom_base_test import CustomModuleBaseTest
from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser
from mypt.building_blocks.conv_blocks.residual.resnet_block import WideResnetBlock


class TestWideResnetBlock(CustomModuleBaseTest):
    """Test class for WideResnetBlock implementation"""
    
    def setUp(self):
        """Initialize test parameters"""
        self.dim_analyser = DimensionsAnalyser()
        
        # Define common test parameters
        self.in_channels_range = (1, 32)
        self.out_channels_range = (1, 64)
        self.stride_options = [1, 2]
        self.dropout_options = [0.0, 0.1, 0.5]
        self.force_residual_options = [True, False]
        
        # For normalization and activation
        self.norm_types = ['batchnorm2d', 'groupnorm']
        self.activation_types = ['relu', 'leaky_relu', 'gelu']
    
    def _get_valid_input(self, block=None, batch_size=2, height=32, width=32) -> torch.Tensor:
        """Generate a random input tensor with the correct shape for the given block"""
        if block is None:
            in_channels = random.randint(*self.in_channels_range)
        else:
            in_channels = block.in_channels
        
        return torch.randn(batch_size, in_channels, height, width)
    
    def _generate_random_wideresnet_block(self, 
                                        in_channels=None, 
                                        out_channels=None,
                                        stride=None,
                                        dropout_rate=None,
                                        force_residual=None) -> WideResnetBlock:
        """Generate a random WideResnetBlock with configurable parameters"""
        # Set parameters or choose random values
        if in_channels is None:
            in_channels = random.randint(*self.in_channels_range)
        
        if out_channels is None:
            out_channels = random.randint(*self.out_channels_range)
        
        if stride is None:
            stride = random.choice(self.stride_options)
        
        if dropout_rate is None:
            dropout_rate = random.choice(self.dropout_options)
        
        if force_residual is None:
            force_residual = random.choice(self.force_residual_options)
        
        # Choose normalization type and parameters
        norm_type = random.choice(self.norm_types)
        
        if norm_type == 'groupnorm':
            in_channels = in_channels * 3
            out_channels = out_channels * 3
            norm1_params = {'num_channels': in_channels, 'num_groups': 3}
            norm2_params = {'num_channels': out_channels, 'num_groups': 3}
        else:
            norm1_params = {'num_features': in_channels}
            norm2_params = {'num_features': out_channels}

        # Choose activation type and parameters
        activation_type = random.choice(self.activation_types)
        activation_params = {'inplace': True} if activation_type == 'relu' else {}
        
        # Create the block
        return WideResnetBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dropout_rate=dropout_rate,
            norm1=norm_type,
            norm1_params=norm1_params,
            norm2=norm_type,
            norm2_params=norm2_params,
            activation=activation_type,
            activation_params=activation_params,
            force_residual=force_residual
        )
    
    ########################## Block Structure Tests ##########################
    
    def test_block_structure(self):
        """Test that the WideResnetBlock has the correct structure"""
        for _ in range(1000):
            block = self._generate_random_wideresnet_block()
            
            # Check main block components
            self.assertIsInstance(block._block, nn.Sequential)
            
            # Check that _block contains expected components
            main_block_children = dict(block._block.named_children())
            self.assertIn('norm_act_1', main_block_children)
            self.assertIn('conv1', main_block_children)
            self.assertIn('dropout', main_block_children)
            self.assertIn('norm_act_2', main_block_children)
            self.assertIn('conv2', main_block_children)
            
            # Check if shortcut exists when needed
            if block._shortcut is None:
                # Shortcut should be None only when stride=1 and in_channels=out_channels
                # and force_residual=False
                self.assertEqual(block.stride, 1)
                self.assertEqual(block.in_channels, block.out_channels)
                self.assertFalse(block._force_residual)
            else:
                # Shortcut should be a Conv2d
                self.assertIsInstance(block._shortcut, nn.Conv2d)
                self.assertEqual(block._shortcut.in_channels, block.in_channels)
                self.assertEqual(block._shortcut.out_channels, block.out_channels)

    def test_residual_path_creation(self):
        """Test that residual path is created correctly based on parameters"""

        # Test with matching dimensions and force_residual=False
        for _ in range(100):
            c = random.randint(1, 100)
            block = self._generate_random_wideresnet_block(
                in_channels=c, 
                out_channels=c, 
                stride=1, 
                force_residual=False
            )
            self.assertIsNone(block._shortcut, 
                            "Shortcut should not be created when dimensions match and force_residual=False")
        
        # Test with non-matching channels
        for _ in range(100):
            c1 = random.randint(1, 100)            
            block = self._generate_random_wideresnet_block(
                in_channels=c1, 
                out_channels=random.randint(c1 + 1, c1 + 100), 
                stride=1, 
                force_residual=False
            )
            self.assertIsNotNone(block._shortcut, 
                                "Shortcut should be created when channels don't match")
        
        
        # with a stride of 2
        for _ in range(100):
            c = random.randint(1, 100)
            block = self._generate_random_wideresnet_block(
                in_channels=c, 
                out_channels=c, 
                stride=2, 
                force_residual=False
            )
            self.assertIsNotNone(block._shortcut, 
                                "Shortcut should be created when stride is not 1")



        # Test with force_residual=True
        for _ in range(100):
            c = random.randint(1, 100)
            block = self._generate_random_wideresnet_block(
                in_channels=c, 
                out_channels=c, 
                stride=1, 
                force_residual=True
            )
            self.assertIsNotNone(block._shortcut, 
                            "Shortcut should be created when force_residual=True")
    
    ########################## Forward Pass Tests ##########################
    
    def test_forward_pass_basic(self):
        """Test basic forward pass functionality"""
        for _ in range(100):
            block = self._generate_random_wideresnet_block()
            x = self._get_valid_input(block)
            
            # Regular forward pass
            output = block(x)
            
            # Check output dimensions
            batch_size, in_channels, height, width = x.shape
            expected_height = height if block.stride == 1 else (height + 1) // 2
            expected_width = width if block.stride == 1 else (width + 1) // 2
            
            self.assertEqual(output.shape, (batch_size, block.out_channels, expected_height, expected_width))
    
    def test_forward_pass_debug_mode(self):
        """Test forward pass with debug=True"""
        for _ in range(100):
            block = self._generate_random_wideresnet_block()
            x = self._get_valid_input(block)
            
            # make sure to set the block to the eval mode: otherwise calling the block twice might not give the same output
            block.eval()
            # Debug mode forward pass   
            main_output, residual_output, combined_output = block(x, debug=True)
            
            # Regular forward pass
            regular_output = block(x)
            
            # Check that regular output matches combined output
            self.assertTrue(torch.allclose(regular_output, combined_output))
            
            calculated_output = main_output + residual_output
            self.assertTrue(torch.allclose(calculated_output, combined_output))
    
    def test_dropout_effect(self):
        """Test that dropout has an effect in training mode but not in eval mode"""
        # Create a block with significant dropout
        for rate in np.linspace(0, 0.8, 11):
            rate = rate.item()
            block = self._generate_random_wideresnet_block(dropout_rate=rate)
            x = self._get_valid_input(block)
            
            # In eval mode, output should be deterministic
            block.eval()
            
            with torch.no_grad():
                output1 = block(x)
                output2 = block(x)
            
            self.assertTrue(torch.allclose(output1, output2), 
                        "In eval mode, outputs should be identical despite dropout")
            
            # In training mode with high dropout, outputs should differ
            block.train()
            output1 = block(x)
            output2 = block(x)
            
            if rate > 0:
                self.assertFalse(torch.allclose(output1, output2), 
                            "In training mode with dropout, outputs should differ")
    
    def test_stride_behavior(self):
        """Test that stride properly affects output dimensions"""
        for _ in range(100):
            # Test with stride=1
            block = self._generate_random_wideresnet_block(stride=1)
            x = self._get_valid_input(block)
            output = block(x)
            
            self.assertEqual(output.shape[2], x.shape[2], 
                            "With stride=1, height should remain unchanged")
            self.assertEqual(output.shape[3], x.shape[3], 
                            "With stride=1, width should remain unchanged")
            
            # Test with stride=2
            block = self._generate_random_wideresnet_block(stride=2)
            x = self._get_valid_input(block)
            output = block(x)
            
            self.assertEqual(output.shape[2], (x.shape[2] + 1) // 2, 
                            "With stride=2, height should be (h+1)//2")
            self.assertEqual(output.shape[3], (x.shape[3] + 1) // 2, 
                            "With stride=2, width should be (w+1)//2")
    
    ########################## CustomModuleBaseTest Tests ##########################
    
    def test_eval_mode(self):
        """Test that the block can be set to evaluation mode"""
        for _ in range(100):
            block = self._generate_random_wideresnet_block()
            super()._test_eval_mode(block)
    
    def test_train_mode(self):
        """Test that the block can be set to training mode"""
        for _ in range(100):
            block = self._generate_random_wideresnet_block()
            super()._test_train_mode(block)
    
    def test_consistent_output_without_dropout_bn(self):
        """Test consistent output for non-stochastic blocks"""
        # Skip for blocks with dropout
        for _ in range(100):
            block = self._generate_random_wideresnet_block(dropout_rate=0.0)
            input_tensor = self._get_valid_input(block)
            super()._test_consistent_output_without_dropout_bn(block, input_tensor)
    
    def test_consistent_output_in_eval_mode(self):
        """Test consistent output in evaluation mode"""
        for _ in range(100):
            block = self._generate_random_wideresnet_block()
            input_tensor = self._get_valid_input(block)
            super()._test_consistent_output_in_eval_mode(block, input_tensor)
    
    def test_batch_size_one_in_train_mode(self):
        """Test handling of batch size 1 in training mode"""
        for _ in range(100):
            block = self._generate_random_wideresnet_block()
            input_tensor = self._get_valid_input(block, batch_size=1)
            super()._test_batch_size_one_in_train_mode(block, input_tensor)
    
    def test_batch_size_one_in_eval_mode(self):
        """Test handling of batch size 1 in evaluation mode"""
        for _ in range(100):
            block = self._generate_random_wideresnet_block()
            input_tensor = self._get_valid_input(block, batch_size=1)
            super()._test_batch_size_one_in_eval_mode(block, input_tensor)
    
    def test_named_parameters_length(self):
        """Test that named_parameters and parameters have the same length"""
        for _ in range(100):
            block = self._generate_random_wideresnet_block()
            super()._test_named_parameters_length(block)
    
    def test_to_device(self):
        """Test that the block can be moved between devices"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping device tests")
            
        for _ in range(100): 
            block = self._generate_random_wideresnet_block()
            input_tensor = self._get_valid_input(block)
            super()._test_to_device(block, input_tensor)
    
    ########################## Special Cases Tests ##########################
    
    def test_property_accessors(self):
        """Test property accessors of the block"""
        for _ in range(100):
            in_channels = random.randint(*self.in_channels_range)
            out_channels = random.randint(*self.out_channels_range)
            stride = random.choice(self.stride_options)
            dropout_rate = random.choice(self.dropout_options)
            
            block = self._generate_random_wideresnet_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                dropout_rate=dropout_rate
            )
            
            # Check properties return correct values
            self.assertTrue(block.in_channels in [in_channels, in_channels * 3])
            self.assertTrue(block.out_channels in [out_channels, out_channels * 3])

            self.assertEqual(block.stride, stride)
            self.assertEqual(block.dropout_rate, dropout_rate)
    
    def test_dimensions_analyser_compatibility(self):
        """Test compatibility with DimensionsAnalyser"""
        for _ in range(100):
            block = self._generate_random_wideresnet_block()
            x = self._get_valid_input(block)
            
            # Get output from block
            output = block(x)
            
            # Get expected output shape from dimensions analyser
            expected_shape1 = self.dim_analyser.analyse_dimensions(
                input_shape=x.shape,
                net=block._block
            )

            # Get expected output shape from dimensions analyser
            expected_shape2= self.dim_analyser.analyse_dimensions(
                input_shape=x.shape,
                net=block._shortcut
            )
            
            # Check that shapes match
            self.assertEqual(tuple(output.shape), expected_shape1)
            self.assertEqual(tuple(output.shape), expected_shape2)


if __name__ == '__main__':
    import mypt.code_utils.pytorch_utils as pu
    pu.seed_everything(42)
    unittest.main()
