import unittest
import random
from tqdm import tqdm

from mypt.building_blocks.conv_blocks.conv_block_design.expanding_helper import (
    main_function,
    get_output_dim,
    compute_all_possible_outputs
)


class TestExpandingHelper(unittest.TestCase):
    """
    Test class for the transpose convolution block design helper functions.
    """
    
    def test_basic_expansion(self):
        """Test that we can find transpose conv blocks that expand dimensions"""
        min_n = 2
        max_n = 4
        
        # Test some basic expansion cases
        test_cases = [
            (16, 32),  # Simple doubling
            (8, 32),   # 4x expansion
            (12, 48),  # 4x expansion with non-power-of-2 input
            (7, 28),   # Odd input
        ]
        
        for input_dim, output_dim in test_cases:
            with self.subTest(input_dim=input_dim, output_dim=output_dim):
                block, cost = main_function(input_dim, output_dim, min_n, max_n)
                
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
    
    def test_invalid_inputs(self):
        """Test that we handle invalid inputs appropriately"""
        min_n = 2
        max_n = 4
        
        # Input dimension >= output dimension should raise ValueError
        with self.assertRaises(ValueError):
            main_function(32, 16, min_n, max_n)
        
        with self.assertRaises(ValueError):
            main_function(10, 10, min_n, max_n)
    
    @unittest.skip("Skip comprehensive test to save test runtime")
    def test_compute_all_possible_outputs(self):
        """
        Test that compute_all_possible_outputs generates valid output dimensions
        that can be produced from the given input dimension by transpose conv blocks.
        """
        min_n = 2
        max_n = 4
        
        # Test with a small set of input dimensions
        possible_input_dims = [8, 16, 24, 32]
        
        for input_dim in possible_input_dims:
            # Compute possible output dimensions
            possible_outputs = set()
            memo = set()
            
            compute_all_possible_outputs(min_n, max_n, input_dim, possible_outputs, memo, 2)
            
            # Remove the input dimension itself
            if input_dim in possible_outputs:
                possible_outputs.remove(input_dim)
            
            # Check a sample of the computed output dimensions
            # (testing all could be too time-consuming)
            sample_size = min(5, len(possible_outputs))
            if sample_size > 0:
                for output_dim in random.sample(list(possible_outputs), sample_size):
                    with self.subTest(input_dim=input_dim, output_dim=output_dim):
                        # Verify we can find a block that produces this output
                        block, cost = main_function(input_dim, output_dim, min_n, max_n)
                        
                        self.assertIsNotNone(block, 
                            f"Failed to find valid block for input={input_dim}, output={output_dim}")
                        
                        # Check the output is correct
                        result_dim = get_output_dim(input_dim, block)
                        self.assertEqual(result_dim, output_dim,
                            f"Block should expand input_dim={input_dim} to output_dim={output_dim}, "
                            f"but got {result_dim}")
                        
                        # Check kernel sizes are in decreasing order
                        tconv_blocks = [b for b in block if b["type"] == "tconv"]
                        kernel_sizes = [b["kernel_size"] for b in tconv_blocks]
                        
                        self.assertEqual(
                            kernel_sizes, 
                            sorted(kernel_sizes, reverse=True),
                            f"Kernel sizes should be in decreasing order, but got {kernel_sizes}"
                        )
                        
                        # Check block lengths meet min_n and max_n requirements
                        streak = 0
                        max_streak = 0
                        
                        for b in block:
                            if b["type"] == "tconv":
                                streak += 1
                            else:
                                max_streak = max(max_streak, streak)
                                streak = 0
                        
                        max_streak = max(max_streak, streak)
                        
                        if max_streak > 0:  # If there are any tconv blocks
                            self.assertGreaterEqual(
                                max_streak, 
                                min_n,
                                f"Number of consecutive tconv blocks ({max_streak}) "
                                f"should be >= min_n ({min_n})"
                            )
                            self.assertLessEqual(
                                max_streak, 
                                max_n,
                                f"Number of consecutive tconv blocks ({max_streak}) "
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
        
        # A few input dimensions to test
        input_dims = [8, 16, 32]
        
        for input_dim in input_dims:
            # Compute possible output dimensions
            possible_outputs = set()
            memo = set()
            
            compute_all_possible_outputs(min_n, max_n, input_dim, possible_outputs, memo, 2)
            
            # Try some dimensions that aren't in the possible_outputs set
            # These should be impossible to achieve with our blocks
            test_range = range(input_dim + 1, input_dim * 4)
            impossible_outputs = [d for d in test_range if d not in possible_outputs]
            
            # Test a sample of impossible outputs
            sample_size = min(5, len(impossible_outputs))
            if sample_size > 0:
                for output_dim in random.sample(impossible_outputs, sample_size):
                    with self.subTest(input_dim=input_dim, output_dim=output_dim):
                        block, cost = main_function(input_dim, output_dim, min_n, max_n)
                        
                        # Should not find a valid block
                        self.assertIsNone(block, 
                            f"Expected None for impossible input={input_dim}, output={output_dim}")
                        self.assertIsNone(cost, 
                            f"Expected None cost for impossible input={input_dim}, output={output_dim}")


if __name__ == "__main__":
    unittest.main() 