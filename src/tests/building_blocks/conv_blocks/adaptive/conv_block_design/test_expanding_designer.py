import torch
import unittest

from torch import nn

from mypt.building_blocks.conv_blocks.basic.transpose_conv_block import TransposeConvBlock
from mypt.building_blocks.conv_blocks.adaptive.conv_block_design.expanding_designer import ExpandingCbDesigner

@unittest.skip("skip for now")
class TestExpandingDesigner(unittest.TestCase):
    """
    Test class for the ExpandingCbDesigner which designs optimal transpose convolutional blocks
    that expand dimensions from input_shape to output_shape.
    """
    
    def setUp(self):
        # Define test cases: (input_shape, output_shape) tuples
        self.test_cases = [
            ((3, 8, 8), (16, 32, 32)),

            ((3, 8, 8), (16, 64, 64)),

            ((3, 4, 4), (32, 128, 128)),

            ((3, 8, 8), (16, 32, 32)),

            ((3, 8, 8), (16, 64, 32)),

            ((3, 8, 4), (16, 64, 32)),

            ((3, 4, 4), (32, 128, 64)),

            ((3, 4, 4), (32, 100, 64)),

            ((3, 4, 4), (32, 100, 200)),

            ((1, 4, 4), (4, 16, 32)),
        ]
    
    def test_designer_initialization(self):
        """Test that the designer initializes correctly with valid inputs"""
        for input_shape, output_shape in self.test_cases:
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                designer = ExpandingCbDesigner(input_shape, output_shape)
                self.assertIsInstance(designer, ExpandingCbDesigner)
    
    def test_designer_invalid_input(self):
        """Test that the designer raises ValueError with invalid inputs"""
        invalid_cases = [
            # Output height < input height
            ((3, 32, 32), (16, 16, 64)),
            # Output width < input width
            ((3, 32, 32), (16, 64, 16)),
            # Both dimensions smaller
            ((3, 32, 32), (16, 16, 16)),
            # Equal dimensions
            ((3, 32, 32), (16, 32, 32)),
        ]
        
        for input_shape, output_shape in invalid_cases:
            with self.subTest(f"Invalid - Input: {input_shape}, Output: {output_shape}"):
                with self.assertRaises(ValueError):
                    ExpandingCbDesigner(input_shape, output_shape)
    
    def test_increasing_kernel_sizes(self):
        """Test that kernel sizes are appropriate for transpose convolution blocks"""
        for input_shape, output_shape in self.test_cases:  # Test a few cases
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                designer = ExpandingCbDesigner(input_shape, output_shape, max_conv_layers_per_block=5, min_conv_layers_per_block=2)
                merged_blocks = designer.get_expanding_block()

                prev_w, prev_h = float('-inf'), float('-inf')
                current_w, current_h = float('-inf'), float('-inf')

                # Check kernel sizes
                for i, block in enumerate(merged_blocks):
                    # Check that block is a TransposeConvBlock
                    self.assertIsInstance(block, TransposeConvBlock)
                    
                    # Kernel sizes should be appropriate for expansion
                    for module in block.children():
                        if not isinstance(module, nn.ConvTranspose2d):
                            continue

                        if isinstance(module, nn.ConvTranspose2d) and module.stride != (1, 1):
                            # ignore strided tranpose convolutional layers since they do not follow the increasing kernel size trend
                            continue

                        h, w = module.kernel_size
                        current_h = h if h != 1 else prev_h
                        current_w = w if w != 1 else prev_w

                        self.assertGreaterEqual(current_h, prev_h, f"Kernel height increased from {prev_h} to {current_h} in block {i}")
                        self.assertGreaterEqual(current_w, prev_w, f"Kernel width increased from {prev_w} to {current_w} in block {i}")

                        prev_h, prev_w = current_h, current_w



    # @unittest.skip("skip for now")
    def test_merged_blocks_structure(self):
        """Test that merged blocks have the correct structure"""
        for input_shape, output_shape in self.test_cases:
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                designer = ExpandingCbDesigner(input_shape, output_shape)
                merged_blocks = designer.get_expanding_block()
                
                # Check that we have at least one block
                self.assertGreater(len(merged_blocks), 0)
                
                # Check each block's structure
                for i, block in enumerate(merged_blocks):
                    # Each block should be a TransposeConvBlock
                    self.assertIsInstance(block, TransposeConvBlock)
                    
                    # Check that the block has layers
                    self.assertGreater(len(list(block.children())), 0)
                    
                    # Check for transpose conv and batch norm layers
                    has_tconv = False
                    has_bn = False
                    
                    for module in block.modules():
                        if isinstance(module, nn.ConvTranspose2d):
                            has_tconv = True
                        elif isinstance(module, nn.BatchNorm2d):
                            has_bn = True
                    
                    self.assertTrue(has_tconv, f"Block {i} missing ConvTranspose2d layers")
                    self.assertTrue(has_bn, f"Block {i} missing BatchNorm2d layers")

    # @unittest.skip("skip for now")
    def test_dimensions_expansion(self):
        """Test that the generated blocks correctly expand dimensions"""
        for input_shape, output_shape in self.test_cases:
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                designer = ExpandingCbDesigner(input_shape, output_shape)
                merged_blocks = designer.get_expanding_block()
                
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
                designer = ExpandingCbDesigner(input_shape, output_shape)
                merged_blocks = designer.get_expanding_block()
                
                # Get the channels from each block
                channels = []
                for block in merged_blocks:
                    # First ConvTranspose2d layer gives us the input channels
                    for module in block.children():
                        if isinstance(module, nn.ConvTranspose2d):
                            channels.append(module.in_channels)
                            break
                
                # Add the final output channels from the last block
                last_block = merged_blocks[-1]
                for module in reversed(list(last_block.children())):
                    if isinstance(module, nn.ConvTranspose2d):
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