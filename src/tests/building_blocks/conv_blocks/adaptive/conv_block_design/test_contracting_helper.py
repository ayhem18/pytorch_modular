import torch
import random
import unittest

import torch.nn as nn

from tqdm import tqdm

from mypt.building_blocks.conv_blocks.adaptive.conv_block_design.contracting_helper import (
    best_conv_block, 
    get_output_dim,
    compute_all_possible_inputs
)

@unittest.skip("skip for now")
class TestConvBlockDesignHelper(unittest.TestCase):
    """
    Test class for the helper functions in the convolutional block design module.
    """

    def _block_module(self, block: list, in_channels: int = 3) -> nn.Sequential:
        """
        Convert a block representation (list of dicts) into an nn.Sequential module.
        
        Args:
            block: List of layer dictionaries from best_conv_block
            in_channels: Number of input channels (default: 3)
            
        Returns:
            nn.Sequential module that implements the block
        """
        layers = []
        
        for layer in block:
            if layer["type"] == "conv":
                ks = layer["kernel_size"]
                stride = layer.get("stride", 1)
                padding = 0  # No padding to match the dimension reduction
                layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=ks, 
                                        stride=stride, padding=padding))
            elif layer["type"] == "pool":
                ks = layer["kernel_size"]
                stride = layer["stride"]
                layers.append(nn.AvgPool2d(kernel_size=ks, stride=stride))
            else:
                raise ValueError(f"Unknown layer type: {layer['type']}")
                
        return nn.Sequential(*layers)

    @unittest.skip("skip for now")
    def test_validation_output_dim_too_small(self):
        """
        Test that the main function validates whether the output_dim is too small
        relative to the input_dim.
        """
        min_n = 2
        max_n = 4
                
        # Test a wide range of dimensions
        for i in range(20, 1000, 20):  # Step by 20 to reduce test time
            # Choose a value that is at least half of i
            # This should trigger the ValueError for being too large relative to input_dim
            output_dim = random.randint(i // 2, i - 1)
            
            with self.subTest(input_dim=i, output_dim=output_dim):
                with self.assertRaises(ValueError):
                    best_conv_block(i, output_dim, min_n, max_n)

    @unittest.skip("skip for now")
    def test_best_conv_block_initial(self):
        """
        Test that the dp_function produces valid blocks that:
        1. When applied to input_dim result in output_dim
        2. Have decreasing kernel sizes
        3. Have groups of consecutive convolutional blocks between min_n and max_n
        """
        min_n = 2
        max_n = 6        
        # We'll test a subset of combinations to keep the test time reasonable
        output_dims = [1, 2, 4, 8, 16, 32, 64]
        
        for output_dim in output_dims:
            # Test a range of input dimensions for each output dimension
            input_dims = [
                2 * output_dim + 10,
                2 * output_dim + 50,
                2 * output_dim + 100
            ]
            
            for input_dim in input_dims:
                with self.subTest(input_dim=input_dim, output_dim=output_dim):
                    try:
                        memo_block = {}
                        block, cost = best_conv_block(
                            input_dim=input_dim, 
                            output_dim=output_dim, 
                            min_n=min_n, 
                            max_n=max_n,
                            memo_block=memo_block,
                            pool_layer_params=(2, 2)
                        )
                        
                        if block is None:
                            continue
                        
                        # Check 1: Applying the block to input_dim should result in output_dim
                        result_dim = get_output_dim(input_dim, block)
                        self.assertEqual(
                            result_dim, 
                            output_dim, 
                            f"Block should reduce input_dim={input_dim} to output_dim={output_dim}, "
                            f"but got {result_dim}"
                        )
                        
                        # Check 2: The kernel sizes should be in decreasing order
                        conv_blocks = [b for b in block if b["type"] == "conv"]
                        kernel_sizes = [b["kernel_size"] for b in conv_blocks]
                        
                        # The kernel sizes should be in decreasing order
                        sorted_kernel_sizes = sorted(kernel_sizes, reverse=True)
                        self.assertEqual(
                            kernel_sizes,
                            sorted_kernel_sizes,
                            f"Kernel sizes are not in decreasing order: {kernel_sizes}"
                        )
                        
                        # Check 3: Each group of consecutive conv blocks should have 
                        # between min_n and max_n conv layers
                        consecutive_groups = []
                        current_group = []
                        
                        for b in block:
                            if b["type"] == "conv":
                                current_group.append(b)
                            else:
                                # Pool block encountered - end of consecutive conv group
                                if current_group:
                                    consecutive_groups.append(current_group)
                                    current_group = []
                        
                        # Don't forget to add the last group if it exists
                        if current_group:
                            consecutive_groups.append(current_group)
                        
                        # Verify each group size is within bounds
                        for group_idx, group in enumerate(consecutive_groups):
                            num_conv_in_group = len(group)
                            self.assertGreaterEqual(
                                num_conv_in_group, 
                                min_n,
                                f"Group {group_idx}: Number of consecutive conv blocks ({num_conv_in_group}) "
                                f"should be >= min_n ({min_n})"
                            )
                            self.assertLessEqual(
                                num_conv_in_group, 
                                max_n,
                                f"Group {group_idx}: Number of consecutive conv blocks ({num_conv_in_group}) "
                                f"should be <= max_n ({max_n})"
                            )
                            
                        # Additional check: Make sure every block entry has expected keys
                        for b in block:
                            self.assertIn("type", b, "Block should have 'type' key")
                            self.assertIn("kernel_size", b, "Block should have 'kernel_size' key")
                            self.assertIn("stride", b, "Block should have 'stride' key")
                    except Exception as e:
                        print(e)
                        self.fail(f"Exception for input_dim={input_dim}, output_dim={output_dim}: {str(e)}")

    @unittest.skip("Very comprehensive test - runs extremely slowly")
    def test_best_conv_block(self):
        """
        Test that compute_all_possible_inputs generates valid input dimensions
        that can be processed by main_function to produce the expected output dimension.
        
        Also verifies that:
        1. When applied to input_dim result in output_dim
        2. Have decreasing kernel sizes
        3. Have groups of consecutive convolutional blocks between min_n and max_n
        """
        min_n = 2
        max_n = 5
        
        possible_output_dims = list(range(1, 200, 5))


        for output_dim in tqdm(possible_output_dims, desc="Testing output dimensions"):
            # Compute possible input dimensions for 5 blocks         
            memo = compute_all_possible_inputs(min_n, max_n, output_dim, 3)

            for input_dim in memo:
                with self.subTest(input_dim=input_dim, output_dim=output_dim):
                    # Verify that main_function can find a solution
                    block, cost = best_conv_block(input_dim, output_dim, min_n, max_n)
                    
                    # The block should not be None since we computed a valid input
                    self.assertIsNotNone(block, f"Failed to find valid block for input={input_dim}, output={output_dim}")
                    
                    # Check 1: Verify the resulting output dimension is correct
                    result_dim = get_output_dim(input_dim, block)
                    self.assertEqual(
                        result_dim, 
                        output_dim, 
                        f"Block should reduce input_dim={input_dim} to output_dim={output_dim}, "
                        f"but got {result_dim}"
                    )
                    
                    # Check 2: The kernel sizes should be in decreasing order
                    conv_blocks = [b for b in block if b["type"] == "conv"]
                    kernel_sizes = [b["kernel_size"] for b in conv_blocks]
                    
                    self.assertEqual(
                        kernel_sizes, 
                        sorted(kernel_sizes, reverse=True),
                        f"Kernel sizes should be in decreasing order, but got {kernel_sizes}"
                    )
                    
                    # Check 3: Each group of consecutive convolutional blocks should be between min_n and max_n
                    consecutive_groups = []
                    current_group = []
                    
                    for b in block:
                        if b["type"] == "conv":
                            current_group.append(b)
                        else:
                            # Pool block encountered - end of consecutive conv group
                            if current_group:
                                consecutive_groups.append(current_group)
                                current_group = []
                    
                    # Don't forget to add the last group if it exists
                    if current_group:
                        consecutive_groups.append(current_group)
                    
                    # Verify each group size is within bounds
                    for group_idx, group in enumerate(consecutive_groups):
                        num_conv_in_group = len(group)
                        self.assertGreaterEqual(
                            num_conv_in_group, 
                            min_n,
                            f"Group {group_idx}: Number of consecutive conv blocks ({num_conv_in_group}) "
                            f"should be >= min_n ({min_n})"
                        )
                        self.assertLessEqual(
                            num_conv_in_group, 
                            max_n,
                            f"Group {group_idx}: Number of consecutive conv blocks ({num_conv_in_group}) "
                            f"should be <= max_n ({max_n})"
                        )
                        
                    for b in block:
                        self.assertIn("type", b, "Block should have 'type' key")
                        self.assertIn("kernel_size", b, "Block should have 'kernel_size' key")
                        self.assertIn("stride", b, "Block should have 'stride' key")

                    # Create the sequential module
                    seq_module = self._block_module(block)
                    
                    # Create input tensor and pass through module
                    x = torch.randn(1, 3, input_dim, input_dim)
                    out = seq_module(x)
                    
                    # Check output shape
                    self.assertEqual(
                        tuple(out.shape), 
                        (1, 3, output_dim, output_dim),
                        f"Module output shape mismatch: expected {(1, 3, output_dim, output_dim)}, got {tuple(out.shape)}"
                    )
                            

    @unittest.skip("Skip negative test to save test runtime")
    def test_compute_impossible_inputs(self):
        """
        Complement to test_compute_all_possible_inputs.
        
        For each output dimension, compute all possible input dimensions,
        then verify that any dimensions between the output_dim and the maximum
        possible input (that aren't in the set of possible inputs) result
        in main_function returning None.
        
        This ensures main_function correctly returns None when no valid convolutional
        block solution exists.
        """
        min_n = 2
        max_n = 5
        
        # Test with a small set of output dimensions for speed
        possible_output_dims = list(range(1, 200, 5))
        
        for output_dim in tqdm(possible_output_dims, desc="Testing impossible inputs"):
            # Compute possible input dimensions
            possible_inputs = compute_all_possible_inputs(min_n, max_n, output_dim, 3)  
                
            # Find the maximum possible input dimension
            if possible_inputs:
                max_possible_input = max(possible_inputs)
                
                # Test a range of impossible inputs
                # Start from output_dim + 1 to avoid testing the output_dim itself
                for input_dim in range(2 * output_dim + 3, max_possible_input + 1):
                    # Skip if this is actually a possible input
                    if input_dim in possible_inputs:
                        continue
                        
                    with self.subTest(input_dim=input_dim, output_dim=output_dim):
                        # Verify main_function returns None for impossible dimensions
                        block, cost = best_conv_block(input_dim, output_dim, min_n, max_n)
                        
                        # main_function should return None for both block and cost
                        # when no solution exists
                        self.assertIsNone(block, 
                            f"Expected None block for impossible input={input_dim}, output={output_dim}")
                        self.assertIsNone(cost, 
                            f"Expected None cost for impossible input={input_dim}, output={output_dim}")
                        

if __name__ == "__main__":
    unittest.main() 