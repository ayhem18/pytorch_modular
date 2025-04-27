import unittest
import random
import numpy as np
from typing import Dict, List, Any

from mypt.building_blocks.conv_blocks.conv_block_design.helper import (
    main_function, 
    get_output_dim
)


class TestConvBlockDesignHelper(unittest.TestCase):
    """
    Test class for the helper functions in the convolutional block design module.
    """

    @unittest.skip("skip for now")
    def test_validation_output_dim_too_small(self):
        """
        Test that the main function validates whether the output_dim is too small
        relative to the input_dim.
        """
        min_n = 2
        max_n = 4
        
        # Test basic valid case first
        input_dim = 32
        output_dim = 1
        block, _ = main_function(input_dim, output_dim, min_n, max_n)
        self.assertIsNotNone(block, "Block should not be None for valid dimensions")
        
        # Test a wide range of dimensions
        for i in range(20, 1000, 20):  # Step by 20 to reduce test time
            # Choose a value that is at least half of i
            # This should trigger the ValueError for being too large relative to input_dim
            output_dim = random.randint(i // 2, i - 1)
            
            with self.subTest(input_dim=i, output_dim=output_dim):
                with self.assertRaises(ValueError):
                    main_function(i, output_dim, min_n, max_n)

    @unittest.skip("skip for now")
    def test_dp_function_results(self):
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
                        block, cost = main_function(
                            input_dim=input_dim, 
                            output_dim=output_dim, 
                            min_n=min_n, 
                            max_n=max_n,
                            memo_block=memo_block
                        )
                        
                        if block is None:
                            print(f"Failed to find a valid block for input_dim={input_dim}, "
                                    f"output_dim={output_dim}")
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
                        
                        self.assertEqual(
                            kernel_sizes, 
                            sorted(kernel_sizes, reverse=True),
                            f"Kernel sizes should be in decreasing order, but got {kernel_sizes}"
                        )
                        
                        # Check 3: Each group of consecutive convolutional blocks should be between min_n and max_n
                        # Count consecutive conv blocks
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
                        self.fail(f"Exception for input_dim={input_dim}, output_dim={output_dim}: {str(e)}")
    
    # @unittest.skip("Very comprehensive test - runs extremely slowly")
    def test_dp_function_results_comprehensive(self):
        """
        Comprehensive test that tests a much wider range of dimensions.
        This test is skipped by default because it would take a very long time to run.
        
        Tests that the dp_function produces valid blocks that:
        1. When applied to input_dim result in output_dim
        2. Have decreasing kernel sizes
        3. Have groups of consecutive convolutional blocks between min_n and max_n
        """
        min_n = 2
        max_n = 6
        
        total_counter = 0
        success_counter = 0
        # Test a wide range of output dimensions
        for output_dim in range(1, 1000, 10):  # Step by 10 to make test more manageable
            # For each output_dim, test a range of input dimensions
            input_dims = []
            for offset in range(10, 101, 10):  # Offsets from 10 to 100
                input_dims.append(2 * output_dim + offset)
            
            for input_dim in input_dims:
                with self.subTest(input_dim=input_dim, output_dim=output_dim):
                    try:
                        total_counter += 1
                        memo_block = {}
                        block, cost = main_function(
                            input_dim=input_dim, 
                            output_dim=output_dim, 
                            min_n=min_n, 
                            max_n=max_n,
                            memo_block=memo_block
                        )
                        

                        if block is None:
                            # print(f"Failed to find a valid block for input_dim={input_dim}, "
                            #      f"output_dim={output_dim}")

                            continue
                        
                        success_counter += 1 
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
                            
                    except Exception as e:
                        self.fail(f"Exception for input_dim={input_dim}, output_dim={output_dim}: {str(e)}")

        print(f"\nTotal counter: {total_counter}")
        print(f"\nSuccess counter: {success_counter}")

if __name__ == "__main__":
    unittest.main() 