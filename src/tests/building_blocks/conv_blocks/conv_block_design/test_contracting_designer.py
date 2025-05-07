import unittest
import torch
from torch import nn
from typing import Tuple, List

from mypt.building_blocks.conv_blocks.conv_block_design.contracting_designer import ContractingCbDesigner
from mypt.building_blocks.conv_blocks.conv_block import BasicConvBlock


class TestContractingDesigner(unittest.TestCase):
    """
    Test class for the ContractingCbDesigner which designs optimal convolutional blocks
    that reduce dimensions from input_shape to output_shape.
    """
    
    def setUp(self):
        # Define test cases: (input_shape, output_shape) tuples
        self.test_cases = [
            ((3, 32, 32), (16, 8, 8)),

            ((3, 64, 64), (16, 8, 8)),

            ((3, 128, 128), (32, 4, 4)),

            ((3, 32, 32), (16, 8, 8)),

            # the code works for square input and output shapes

            ((3, 64, 32), (16, 8, 8)),

            ((3, 128, 64), (16, 8, 4)),

            ((3, 256, 128), (32, 4, 4)),

            ((3, 100, 64), (32, 4, 4)),

            ((3, 100, 200), (32, 4, 4)),

            ((1, 16, 12), (4, 4, 4)),
        ]
    
    def test_designer_initialization(self):
        """Test that the designer initializes correctly with valid inputs"""
        for input_shape, output_shape in self.test_cases:
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                designer = ContractingCbDesigner(input_shape, output_shape)
                self.assertIsInstance(designer, ContractingCbDesigner)
    

    def test_designer_invalid_input(self):
        """Test that the designer raises ValueError with invalid inputs"""
        invalid_cases = [
            # Output height > input height
            ((3, 32, 32), (16, 64, 8)),
            # Output width > input width
            ((3, 32, 32), (16, 8, 64)),
            # Both dimensions larger
            ((3, 32, 32), (16, 64, 64)),
            # Equal dimensions
            ((3, 32, 32), (16, 32, 32)),
        ]
        
        for input_shape, output_shape in invalid_cases:
            with self.subTest(f"Invalid - Input: {input_shape}, Output: {output_shape}"):
                with self.assertRaises(ValueError):
                    ContractingCbDesigner(input_shape, output_shape)
    

    def test_decreasing_kernel_sizes(self):
        """Test that kernel sizes are decreasing within convolutional blocks"""
        for input_shape, output_shape in self.test_cases:
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                designer = ContractingCbDesigner(input_shape, output_shape)
                merged_blocks = designer.get_contracting_block()

                # Check kernel sizes
                prev_size_h = float('inf')
                prev_size_w = float('inf')

                for i, block in enumerate(merged_blocks):
                    # Check that block is a Sequential
                    self.assertIsInstance(block, nn.Sequential)
                    
                    # Get the conv block
                    self.assertTrue(hasattr(block, 'conv'), f"Block {i} missing 'conv' field")  
                    conv_block = block.conv
                    self.assertIsInstance(conv_block, BasicConvBlock)
                    
                    
                    for module in conv_block.children():
                        if isinstance(module, nn.Conv2d):

                            size_h, size_w = module.kernel_size 

                            if size_h != 1: # the kernel 1 is added out of necessity.
                                current_size_h = size_h
                            else:
                                current_size_h = prev_size_h

                            if size_w != 1: # the kernel 1 is added out of necessity.
                                current_size_w = size_w
                            else:
                                current_size_w = prev_size_w
                            

                            # Kernel sizes should be decreasing or equal
                            self.assertLessEqual(current_size_h, prev_size_h, 
                                                f"Kernel height increased from {prev_size_h} to {current_size_h} in block {i}")
                            self.assertLessEqual(current_size_w, prev_size_w, 
                                                f"Kernel width increased from {prev_size_w} to {current_size_w} in block {i}")
                            
                            prev_size_h, prev_size_w = current_size_h, current_size_w

    def test_merged_blocks_structure(self):
        """Test that merged blocks have the correct structure"""
        for input_shape, output_shape in self.test_cases:
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                designer = ContractingCbDesigner(input_shape, output_shape)
                merged_blocks = designer.get_contracting_block()
                
                # Check that we have at least one block
                self.assertGreater(len(merged_blocks), 0)
                
                # Check each block's structure
                for i, block in enumerate(merged_blocks):
                    # Each block should be a Sequential
                    self.assertIsInstance(block, nn.Sequential)
                    
                    # Should have 'conv' and 'pool' fields
                    self.assertTrue(hasattr(block, 'conv'), f"Block {i} missing 'conv' field")
                    self.assertTrue(hasattr(block, 'pool'), f"Block {i} missing 'pool' field")
                    
                    # Check types
                    self.assertIsInstance(block.conv, BasicConvBlock)
                    self.assertIsInstance(block.pool, nn.AvgPool2d)

    def test_dimensions_reduction(self):
        """Test that the generated blocks correctly reduce dimensions"""
        for input_shape, output_shape in self.test_cases:
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                designer = ContractingCbDesigner(input_shape, output_shape)
                merged_blocks = designer.get_contracting_block()
                
                # Create a tensor with the input shape (add batch dimension)
                x = torch.randn(1, *input_shape)
                
                # Apply each block sequentially
                for block in merged_blocks:
                    x = block.forward(x)
                
                # Final output should have the expected shape
                expected_shape = (1, output_shape[0], output_shape[1], output_shape[2])
                self.assertEqual(tuple(x.shape), expected_shape, 
                               f"Expected shape {expected_shape}, got {tuple(x.shape)}")

    # @unittest.skip("skip for now")
    def test_channels_progression(self):
        """Test that channels follow a logical progression"""
        for input_shape, output_shape in self.test_cases:
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                designer = ContractingCbDesigner(input_shape, output_shape)
                merged_blocks = designer.get_contracting_block()
                
                # Get the channels from each block
                channels = []
                for block in merged_blocks:
                    # First Conv2d layer in the block.conv gives us the input channels
                    for module in block.conv.children():
                        if isinstance(module, nn.Conv2d):
                            channels.append(module.in_channels)
                            break
                
                # Add the final output channels
                last_block = merged_blocks[-1]
                for module in reversed(list(last_block.conv.children())):
                    if isinstance(module, nn.Conv2d):
                        channels.append(module.out_channels)
                        break
                
                # Check progression
                self.assertEqual(channels[0], input_shape[0], "First channel count should match input")
                self.assertEqual(channels[-1], output_shape[0], "Last channel count should match output")
                
                # Should progress monotonically
                if input_shape[0] < output_shape[0]:
                    for i in range(1, len(channels)):
                        self.assertGreaterEqual(channels[i], channels[i-1], 
                                             f"Channels should increase: {channels[i-1]} -> {channels[i]}")
                else:
                    for i in range(1, len(channels)):
                        self.assertLessEqual(channels[i], channels[i-1], 
                                           f"Channels should decrease: {channels[i-1]} -> {channels[i]}")


if __name__ == '__main__':
    unittest.main() 