import torch
import unittest
import random
from typing import Tuple

import mypt.code_utils.pytorch_utils as pu
from tests.custom_base_test import CustomModuleBaseTest
from mypt.nets.conv_nets.adaptive_unet import AdaptiveUNet


class TestAdaptiveUNet(CustomModuleBaseTest):
    """Test class for AdaptiveUNet implementation"""
    
    def setUp(self):
        """Initialize test parameters"""
        # Define sample shapes for testing
        self.input_shapes = [
            (3, 256, 256),    # (channels, height, width)
            (1, 100, 100),
            (16, 128, 128)
        ]
        self.output_shapes = [
            (1, 256, 256),
            (3, 100, 100),
            (4, 128, 128)
        ]
        self.bottleneck_shapes = [
            (64, 16, 16),
            (32, 8, 8),
            (128, 28, 28)
        ]
        
        # Create a default UNet for testing
        self.default_idx = 0
        
        if not hasattr(self, 'unet'):
            self.unet = self._create_unet(self.default_idx)
    
    def _create_unet(self, index=0) -> AdaptiveUNet:
        """Helper to create a UNet with the specified parameter set"""
        unet = AdaptiveUNet(
            input_shape=self.input_shapes[index],
            output_shape=self.output_shapes[index],
            bottleneck_shape=self.bottleneck_shapes[index]
        )
        
        # Build the complete UNet
        unet.build_contracting_path()
        unet.build_bottleneck(kernel_sizes=3, num_blocks=3, conv_layers_per_block=2)
        unet.build_expanding_path()
        unet.build()
        
        return unet
    
    def _get_valid_input(self, unet=None, batch_size=2) -> torch.Tensor:
        """Generate a valid input tensor for the given UNet"""
        if unet is None:
            unet = self.unet
            
        return torch.randn(batch_size, *unet.input_shape)
    
    # @unittest.skip("passed")
    def test_initialization(self):
        """Test that UNet initializes correctly"""
        for i in range(len(self.input_shapes)):
            with self.subTest(i=i):
                unet = AdaptiveUNet(
                    input_shape=self.input_shapes[i],
                    output_shape=self.output_shapes[i],
                    bottleneck_shape=self.bottleneck_shapes[i]
                )
                
                # Check that attributes were set correctly
                self.assertEqual(unet.input_shape, self.input_shapes[i])
                self.assertEqual(unet.output_shape, self.output_shapes[i])
                self.assertEqual(unet.bottleneck_input_shape, self.bottleneck_shapes[i])
                
                # Before building, components should be None
                self.assertIsNone(unet.contracting_path)
                self.assertIsNone(unet.bottleneck)
                self.assertIsNone(unet.expanding_path)
                self.assertIsNone(unet.skip_connections)
                self.assertFalse(unet._is_built)
    
    # @unittest.skip("skip for now")
    def test_build_methods(self):
        """Test the individual build methods"""
        for i in range(len(self.input_shapes)):
            with self.subTest(i=i):
                unet = AdaptiveUNet(
                    input_shape=self.input_shapes[i],
                    output_shape=self.output_shapes[i],
                    bottleneck_shape=self.bottleneck_shapes[i]
                )
        
            # Test build_contracting_path
            unet.build_contracting_path()
            self.assertIsNotNone(unet.contracting_path)
            
            # Test build_bottleneck
            unet.build_bottleneck(kernel_sizes=3, num_blocks=3, conv_layers_per_block=2)
            self.assertIsNotNone(unet.bottleneck)
            
            # Test build_expanding_path
            unet.build_expanding_path()
            self.assertIsNotNone(unet.expanding_path)
            
            # Test build
            unet.build()
            self.assertIsNotNone(unet.skip_connections)
            self.assertTrue(unet._is_built)
    
    # @unittest.skip("skip for now")
    def test_forward(self):
        """Test forward pass with different inputs"""
        for i in range(len(self.input_shapes)):
            with self.subTest(i=i):
                unet = self._create_unet(i)
                batch_size = random.randint(1, 4)
                x = self._get_valid_input(unet, batch_size)
                
                # Forward pass should work
                y = unet(x)
                
                # Output shape should match
                expected_shape = (batch_size,) + unet.output_shape
                self.assertEqual(y.shape, expected_shape)
    
    # @unittest.skip("skip for now")
    def test_forward_error_before_build(self):
        """Test that forward raises error if called before build"""
        unet = AdaptiveUNet(
            input_shape=self.input_shapes[0],
            output_shape=self.output_shapes[0],
            bottleneck_shape=self.bottleneck_shapes[0]
        )
        
        x = self._get_valid_input(self.unet)
        
        with self.assertRaises(RuntimeError):
            unet(x)

    # @unittest.skip("skip for now")
    def test_forward_invalid_input_shape(self):
        """Test that forward raises error with invalid input shape"""
        # Create a tensor with wrong channel count
        wrong_channels = torch.randn(2, self.unet.input_shape[0] + 1, 
                                    self.unet.input_shape[1], 
                                    self.unet.input_shape[2])
        
        with self.assertRaises(ValueError):
            self.unet(wrong_channels)
        
        # Create a tensor with wrong spatial dimensions
        wrong_spatial = torch.randn(2, self.unet.input_shape[0], 
                                   self.unet.input_shape[1] + 2, 
                                   self.unet.input_shape[2] + 2)
        
        with self.assertRaises(ValueError):
            self.unet(wrong_spatial)

    
    # Tests from CustomModuleBaseTest
    
    # @unittest.skip("passed")
    def test_eval_mode(self):
        """Test that eval mode is properly propagated to all components"""
        self._test_eval_mode(self.unet)
    
    # @unittest.skip("passed")
    def test_train_mode(self):
        """Test that train mode is properly propagated to all components"""
        self._test_train_mode(self.unet)

    # @unittest.skip("passed")
    def test_named_parameters(self):
        """Test that named_parameters() matches parameters()"""
        self._test_named_parameters_length(self.unet)

    # @unittest.skip("skip for now")
    def test_batch_size_one_in_eval_mode(self):
        """Test that the model can handle batch size 1 in eval mode"""
        x = self._get_valid_input(batch_size=1)
        self._test_batch_size_one_in_eval_mode(self.unet, x)

    # @unittest.skip("skip for now")
    def test_consistent_output_in_eval_mode(self):
        """Test that the model produces consistent output in eval mode"""
        x = self._get_valid_input()
        self._test_consistent_output_in_eval_mode(self.unet, x)


if __name__ == '__main__':
    pu.seed_everything(42)
    unittest.main() 