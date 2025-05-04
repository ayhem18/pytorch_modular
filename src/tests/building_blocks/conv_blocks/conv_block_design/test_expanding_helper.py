import unittest
import random
from tqdm import tqdm

from mypt.building_blocks.conv_blocks.conv_block_design.expanding_helper import (
    best_transpose_conv_block,
    compute_outputs_exhaustive,
    get_output_dim,
    compute_all_possible_inputs 
)


class TestExpandingHelper(unittest.TestCase):
    """
    Test class for the transpose convolution block design helper functions.
    """
    
    @unittest.skip("skip for now")
    def test_basic_expansion(self):
        """Test that we can find transpose conv blocks that expand dimensions"""
        min_n = 2
        max_n = 5
        
        # Test some basic expansion cases
        test_cases = [
            (16, 40),  # Simple doubling
            (8, 36),   # 4x expansion
            (12, 36),  # 4x expansion with non-power-of-2 input
        ]
        
        for input_dim, output_dim in test_cases:
            with self.subTest(input_dim=input_dim, output_dim=output_dim):
                block, cost = best_transpose_conv_block(input_dim, output_dim, min_n, max_n)
                
                # Should find a valid block
                self.assertIsNotNone(block, f"Failed to find valid block for input={input_dim}, output={output_dim}")
                
                # Applying the block should give the expected output dimension
                result_dim = get_output_dim(input_dim, block)
                self.assertEqual(
                    result_dim, 
                    output_dim, 
                    f"Block should expand input_dim={input_dim} to output_dim={output_dim}, "
                    f"but got {result_dim}"
                )
    
    @unittest.skip("skip for now")
    def test_input_validation(self):
        """Test that we properly handle all invalid inputs"""
        min_n = 2
        max_n = 4
        
        # Test cases for invalid dimensions
        dimension_test_cases = [
            # Input dimension >= output dimension should raise ValueError
            (32, 16, ValueError, "input_dim must be smaller than output_dim"),
            (10, 10, ValueError, "input_dim must be smaller than output_dim"),
            # Negative dimensions should raise ValueError
            (-5, 10, ValueError, "dimensions must be positive"),
            (5, -10, ValueError, "dimensions must be positive"),
            (-5, -10, ValueError, "dimensions must be positive"),
            # Zero dimensions should raise ValueError
            (0, 10, ValueError, "dimensions must be positive"),
            (5, 0, ValueError, "dimensions must be positive"),
            (0, 0, ValueError, "dimensions must be positive"),
        ]
        
        # Test cases for invalid layer constraints
        layer_test_cases = [
            # min_n and max_n validation
            (5, 20, 0, 4, ValueError, "min_n must be positive"),
            (5, 20, -2, 4, ValueError, "min_n must be positive"),
            (5, 20, 2, 0, ValueError, "max_n must be positive"),
            (5, 20, 2, -4, ValueError, "max_n must be positive"),
            (5, 20, 6, 4, ValueError, "min_n cannot be greater than max_n"),
        ]
        
        # Test all dimension test cases
        for input_dim, output_dim, expected_error, error_msg in dimension_test_cases:
            with self.subTest(input_dim=input_dim, output_dim=output_dim, error=expected_error.__name__):
                with self.assertRaises(expected_error, msg=error_msg):
                    best_transpose_conv_block(input_dim, output_dim, min_n, max_n)
        
        # Test all layer constraint test cases
        for input_dim, output_dim, test_min_n, test_max_n, expected_error, error_msg in layer_test_cases:
            with self.subTest(input_dim=input_dim, output_dim=output_dim, 
                              min_n=test_min_n, max_n=test_max_n, 
                              error=expected_error.__name__):
                with self.assertRaises(expected_error, msg=error_msg):
                    best_transpose_conv_block(input_dim, output_dim, test_min_n, test_max_n)

    @unittest.skip("skip for now")
    def test_large_input_dim(self):
        """Test that large input dimensions cause ValueError when they would result in outputs larger than target"""
        min_n = 2
        max_n = 4
                
        for i in range(10, 1000, 10):
            # Choose output_dim between i and 2*i-5
            # This ensures input_dim < output_dim but the gap is small enough that
            # applying even a minimal transpose conv block would exceed the target
            input_dim = i
            output_dim = random.randint(i + 1, 2 * i - 5)
            
            with self.subTest(input_dim=input_dim, output_dim=output_dim):
                # The function should raise ValueError because the minimal transpose
                # convolution block would result in an output dimension larger than target
                with self.assertRaises(
                    ValueError, 
                    msg=f"Expected ValueError for input_dim={input_dim}, output_dim={output_dim}"
                ):
                    best_transpose_conv_block(input_dim, output_dim, min_n, max_n)
        

    @unittest.skip("Skip comprehensive test to save test runtime")
    def test_compute_all_possible_inputs(self):
        """
        Test that compute_all_possible_inputs generates valid output dimensions
        that can be produced from the given input dimension by transpose conv blocks.
        """
        min_n = 2
        max_n = 5
        
        # Test with a small set of input dimensions
        # possible_output_dims = [120, 150, 180, 200, 250]

        possible_output_dims = list(range(10, 1000, 10))

        for output_dim in tqdm(possible_output_dims, desc="iterating over output dims"):            
        # for output_dim in possible_output_dims:            
            possible_inputs = compute_all_possible_inputs(min_n, max_n, output_dim, 3)
            
            for input_dim in possible_inputs:
                # with self.subTest(input_dim=input_dim, output_dim=output_dim):
                    # Verify we can find a block that produces this output
                block, cost = best_transpose_conv_block(input_dim, output_dim, min_n, max_n)
                
                self.assertIsNotNone(block, 
                    f"Failed to find valid block for input={input_dim}, output={output_dim}")
                
                # Check the output is correct
                result_dim = get_output_dim(input_dim, block)
                self.assertEqual(result_dim, output_dim,
                    f"Block should expand input_dim={input_dim} to output_dim={output_dim}, "
                    f"but got {result_dim}")
                
                # Check kernel sizes are in increasing order
                kernel_sizes = [b["kernel_size"] for b in block if b["stride"] == 1]
                
                self.assertEqual(
                    kernel_sizes, 
                    sorted(kernel_sizes),
                    f"Kernel sizes should be in increasing order, but got {kernel_sizes}"
                )
                
                # Check 3: Each group of consecutive convolutional blocks should be between min_n and max_n
                # Count consecutive conv blocks
                consecutive_groups = []
                current_group = []
                
                for b in block:
                    if b["stride"] == 1:
                        current_group.append(b)
                    else:
                        # tconv with stride > 1 encountered - end of consecutive conv group
                        if current_group:
                            consecutive_groups.append(current_group)
                            current_group = []
                
                # Don't forget to add the last group if it exists
                if current_group:
                    consecutive_groups.append(current_group)

                for group in consecutive_groups:
                    self.assertGreaterEqual(
                        len(group), 
                        min_n,
                        f"Number of consecutive tconv blocks ({len(group)}) "
                        f"should be >= min_n ({min_n})"
                    )
                    
                    self.assertLessEqual(
                        len(group), 
                        max_n,
                        f"Number of consecutive tconv blocks ({len(group)}) "
                        f"should be <= max_n ({max_n})"
                    )
    
    @unittest.skip("Skip impossible outputs test to save test runtime")
    def test_compute_impossible_outputs(self):
        """
        Test that the main_function correctly returns None for dimensions that
        cannot be achieved through transpose convolution blocks.
        """
        min_n = 2
        max_n = 4
        
        # output_dims = [120, 150, 180, 200, 250]
        
        output_dims = list(range(10, 1000, 10))

        for output_dim in tqdm(output_dims, desc="iterating over output dims"):
            # Compute possible output dimensions
            possible_inputs = compute_outputs_exhaustive(min_n, max_n, output_dim, 3)

            max_possible_input_dim = max(possible_inputs)            
            min_possible_input_dim = min(possible_inputs)

            test_range = range(min_possible_input_dim + 1, max_possible_input_dim)
            impossible_inputs = [d for d in test_range if d not in possible_inputs]

            for input_dim in impossible_inputs:
                # with self.subTest(input_dim=input_dim, output_dim=output_dim):
                block, cost = best_transpose_conv_block(input_dim, output_dim, min_n, max_n)
                
                # Should not find a valid block
                self.assertIsNone(block, 
                    f"Expected None for impossible input={input_dim}, output={output_dim}")
                self.assertIsNone(cost, 
                    f"Expected None cost for impossible input={input_dim}, output={output_dim}")


if __name__ == "__main__":
    unittest.main() 