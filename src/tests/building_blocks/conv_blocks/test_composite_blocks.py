import torch
import unittest
import random

from torch import nn
from typing import Tuple

import mypt.code_utils.pytorch_utils as pu
from tests.custom_base_test import CustomModuleBaseTest
from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser
from mypt.building_blocks.conv_blocks.composite_blocks import ContractingBlock, ExpandingBlock


class TestContractingBlock(CustomModuleBaseTest):
    """
    Test class for ContractingBlock from composite_blocks.py
    """
    
    def setUp(self):
        """Initialize test case parameters"""
        self.dim_analyser = DimensionsAnalyser()
        
        # Define test cases: (input_shape, output_shape) tuples
        self.test_cases = [
            ((3, 32, 32), (16, 8, 8)),
            ((3, 64, 64), (16, 8, 8)),
            ((3, 128, 128), (32, 4, 4)),
            ((3, 64, 32), (16, 8, 8)),
            ((3, 128, 64), (16, 8, 4)),
            ((3, 100, 64), (32, 4, 4)),
        ]
    
    def _get_valid_input(self, batch_size=2, channels=3, height=32, width=32) -> torch.Tensor:
        """Generate a random input tensor with the correct shape"""
        return torch.randn(batch_size, channels, height, width)
    
    def test_initialization(self):
        """Test that ContractingBlock initializes correctly"""
        for input_shape, output_shape in self.test_cases:
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                block = ContractingBlock(input_shape, output_shape)
                
                # Check that blocks were created successfully
                self.assertIsInstance(block, ContractingBlock)
                self.assertGreater(len(block.block), 0)
                
    ########################## general custom module tests ##########################
    def test_eval_mode(self):
        for input_shape, output_shape in self.test_cases:
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                block = ContractingBlock(input_shape, output_shape)
                super()._test_eval_mode(block)

    def test_train_mode(self):
        for input_shape, output_shape in self.test_cases:
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                block = ContractingBlock(input_shape, output_shape)
                super()._test_train_mode(block)

    def test_consistent_output_without_dropout_bn(self):
        for input_shape, output_shape in self.test_cases:
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                block = ContractingBlock(input_shape, output_shape)
                # to avoid issues with input, create a random input tensor with the same shape as the test
                input_tensor = self._get_valid_input(batch_size=2, channels=input_shape[0], height=input_shape[1], width=input_shape[2])
                super()._test_consistent_output_without_dropout_bn(block, input_tensor)

    def test_named_parameters_length(self):
        for input_shape, output_shape in self.test_cases:
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                block = ContractingBlock(input_shape, output_shape)
                super()._test_named_parameters_length(block) 

    def test_to_device(self):
        for input_shape, output_shape in self.test_cases:
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                block = ContractingBlock(input_shape, output_shape)
                super()._test_to_device(block)

    
    def test_batch_size_one_in_eval_mode(self):
        for input_shape, output_shape in self.test_cases:
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                block = ContractingBlock(input_shape, output_shape)
                super()._test_batch_size_one_in_eval_mode(block, self._get_valid_input(batch_size=1, channels=input_shape[0], height=input_shape[1], width=input_shape[2]))

    ########################## ContractingBlock specific tests ##########################
        
    def test_forward_pass(self):
        """Test that the forward pass produces the expected output shape"""
        for input_shape, output_shape in self.test_cases:
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                block = ContractingBlock(input_shape, output_shape)
                
                # Create input tensor (with batch dimension)
                x = torch.randn(2, *input_shape)
                
                # Standard forward pass
                output = block(x)
                
                # Check output shape
                expected_shape = (2, *output_shape)
                self.assertEqual(tuple(output.shape), expected_shape,
                               f"Expected shape {expected_shape}, got {tuple(output.shape)}")
    
    def test_full_forward_pass(self):
        """Test the full forward pass that returns intermediate outputs"""
        for input_shape, output_shape in self.test_cases:
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                block = ContractingBlock(input_shape, output_shape)
                
                # Create input tensor (with batch dimension)
                x = self._get_valid_input(batch_size=2, channels=input_shape[0], height=input_shape[1], width=input_shape[2])
                
                # Full forward pass
                output, intermediate_outputs = block(x, full=True)
                
                # Check main output shape
                expected_shape = (2, *output_shape)
                self.assertEqual(tuple(output.shape), expected_shape)
                
                # Check intermediate outputs
                self.assertEqual(len(intermediate_outputs), len(block.blocks))
                
                # Each intermediate output should have valid shape
                for i, interm_out in enumerate(intermediate_outputs):
                    self.assertIsInstance(interm_out, torch.Tensor)
                    # Check that each dimension has batch_size as first dimension
                    self.assertEqual(interm_out.shape[0], 2)
                    # Check that the last intermediate output matches the final channel count
                    if i == len(intermediate_outputs) - 1:
                        self.assertEqual(interm_out.shape[1], output_shape[0])
    


class TestExpandingBlock(CustomModuleBaseTest):
    """
    Test class for ExpandingBlock from composite_blocks.py
    """
    
    def setUp(self):
        """Initialize test case parameters"""
        self.dim_analyser = DimensionsAnalyser()
        
        # Define test cases: (input_shape, output_shape) tuples - reversed compared to ContractingBlock
        self.test_cases = [
            ((16, 8, 8), (3, 32, 32)),
            ((16, 8, 8), (3, 64, 64)),
            ((32, 4, 4), (3, 128, 128)),
            ((16, 8, 8), (3, 64, 32)),
            ((16, 8, 4), (3, 128, 64)),
            ((32, 4, 4), (3, 100, 64)),
        ]
    
    def _get_valid_input(self, batch_size=2, channels=16, height=8, width=8) -> torch.Tensor:
        """Generate a random input tensor with the correct shape"""
        return torch.randn(batch_size, channels, height, width)
    
    def test_initialization(self):
        """Test that ExpandingBlock initializes correctly"""
        for input_shape, output_shape in self.test_cases:
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                block = ExpandingBlock(input_shape, output_shape)
                
                # Check that blocks were created successfully
                self.assertIsInstance(block, ExpandingBlock)
                self.assertGreater(len(block.block), 0)
                
    ########################## general custom module tests ##########################
    def test_eval_mode(self):
        for input_shape, output_shape in self.test_cases:
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                block = ExpandingBlock(input_shape, output_shape)
                super()._test_eval_mode(block)

    def test_train_mode(self):
        for input_shape, output_shape in self.test_cases:
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                block = ExpandingBlock(input_shape, output_shape)
                super()._test_train_mode(block)

    def test_consistent_output_without_dropout_bn(self):
        for input_shape, output_shape in self.test_cases:
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                block = ExpandingBlock(input_shape, output_shape)
                # to avoid issues with input, create a random input tensor with the same shape as the test
                input_tensor = self._get_valid_input(batch_size=2, channels=input_shape[0], height=input_shape[1], width=input_shape[2])
                super()._test_consistent_output_without_dropout_bn(block, input_tensor)

    def test_named_parameters_length(self):
        for input_shape, output_shape in self.test_cases:
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                block = ExpandingBlock(input_shape, output_shape)
                super()._test_named_parameters_length(block) 

    def test_to_device(self):
        for input_shape, output_shape in self.test_cases:
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                block = ExpandingBlock(input_shape, output_shape)
                super()._test_to_device(block)

    
    def test_batch_size_one_in_eval_mode(self):
        for input_shape, output_shape in self.test_cases:
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                block = ExpandingBlock(input_shape, output_shape)
                super()._test_batch_size_one_in_eval_mode(block, self._get_valid_input(batch_size=1, channels=input_shape[0], height=input_shape[1], width=input_shape[2]))

    ########################## ExpandingBlock specific tests ##########################

    def test_forward_pass(self):
        """Test that the forward pass produces the expected output shape"""
        for input_shape, output_shape in self.test_cases:
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                block = ExpandingBlock(input_shape, output_shape)
                
                # Create input tensor (with batch dimension)
                x = torch.randn(2, *input_shape)
                
                # Standard forward pass
                output = block(x)
                
                # Check output shape
                expected_shape = (2, *output_shape)
                self.assertEqual(tuple(output.shape), expected_shape,
                               f"Expected shape {expected_shape}, got {tuple(output.shape)}")
    
    def test_full_forward_pass(self):
        """Test the full forward pass that returns intermediate outputs"""
        for input_shape, output_shape in self.test_cases:
            with self.subTest(f"Input: {input_shape}, Output: {output_shape}"):
                block = ExpandingBlock(input_shape, output_shape)
                
                # Create input tensor (with batch dimension)
                x = self._get_valid_input(batch_size=2, channels=input_shape[0], height=input_shape[1], width=input_shape[2])
                
                # Full forward pass
                output, intermediate_outputs = block(x, full=True)
                
                # Check main output shape
                expected_shape = (2, *output_shape)
                self.assertEqual(tuple(output.shape), expected_shape)
                
                # Check intermediate outputs
                self.assertEqual(len(intermediate_outputs), len(block.blocks))
                
                # Each intermediate output should have valid shape
                for i, interm_out in enumerate(intermediate_outputs):
                    self.assertIsInstance(interm_out, torch.Tensor)
                    # Check that each dimension has batch_size as first dimension
                    self.assertEqual(interm_out.shape[0], 2)
                    # Check that the last intermediate output matches the final channel count
                    if i == len(intermediate_outputs) - 1:
                        self.assertEqual(interm_out.shape[1], output_shape[0])
    


if __name__ == '__main__':
    pu.seed_everything(42)
    unittest.main() 