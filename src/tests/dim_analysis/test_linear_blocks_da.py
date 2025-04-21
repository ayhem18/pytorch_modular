import unittest
import torch
import random
import numpy as np
from torch import nn
from random import randint as ri

import mypt.code_utils.pytorch_utils as pu
from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser
from mypt.building_blocks.linear_blocks.components import BasicLinearBlock
from mypt.building_blocks.linear_blocks.fc_blocks import GenericFCBlock, ExponentialFCBlock
from mypt.building_blocks.linear_blocks.residual_fc_blocks import (
    GenericResidualFCBlock, 
    ExponentialResidualFCBlock
)


class TestDimAnalyzerLinearBlock(unittest.TestCase):
    """
    Test the dimension analyzer's ability to correctly predict the output shapes
    of various linear block implementations.
    """
    
    def setUp(self):
        self.analyzer = DimensionsAnalyser(method='static')
        self.activation_names = BasicLinearBlock._ACTIVATIONS.copy()
    
    def _generate_random_basic_linear_block(self, is_final=None, add_activation=None, activation=None, dropout=None):
        """
        Generate a random BasicLinearBlock with configurable parameters
        """
        in_features = ri(10, 100)
        out_features = ri(10, 100)
        
        if activation is None:
            activation = random.choice(self.activation_names)
        
        if is_final is None:
            is_final = random.choice([True, False])
            
        if add_activation is None:
            add_activation = random.choice([True, False])
            
        if dropout is None:
            dropout = random.choice([None, 0.1, 0.5])

        if dropout is False:
            dropout = None

        return BasicLinearBlock(
            in_features=in_features,
            out_features=out_features,
            activation=activation,
            dropout=dropout,
            is_final=is_final,
            add_activation=add_activation
        )
    
    def _generate_random_units(self, in_features, output, num_layers):
        """Generate a random list of units for hidden layers"""
        units = [in_features]
        
        min_dim,max_dim = sorted([output, in_features])

        # Generate hidden layer units
        for _ in range(num_layers - 1):
            units.append(ri(max(min_dim, 10), max(max_dim, 64)))
            
        units.append(output)
        return units
    
    def _generate_random_generic_fc_block(self, num_layers=None, activation=None, dropout=None):
        """
        Generate a random GenericFCBlock with configurable parameters
        """
        if num_layers is None:
            num_layers = ri(2, 5)
            
        in_features = ri(10, 100)
        output = ri(2, 20)
        
        if activation is None:
            activation = random.choice(self.activation_names)
            
        units = self._generate_random_units(in_features, output, num_layers)
        
        if dropout is None:
            if random.choice([True, False]):
                # Use a single dropout value
                dropout_value = random.uniform(0.1, 0.5)
            else:
                # Use a list of dropout values
                dropout_value = [random.uniform(0.1, 0.5) for _ in range(num_layers - 1)]
        elif dropout is False:
            dropout_value = None
        else:
            dropout_value = dropout
            
        return GenericFCBlock(
            output=output,
            in_features=in_features,
            num_layers=num_layers,
            units=units,
            activation=activation,
            dropout=dropout_value
        )
    
    def _generate_random_exponential_fc_block(self, num_layers=None, activation=None, dropout=None):
        """
        Generate a random ExponentialFCBlock with configurable parameters
        """
        if num_layers is None:
            num_layers = ri(2, 5)
            
        # Ensure that we have a wide enough range between in_features and output
        # for the exponential scaling to work well
        in_features = 2 ** ri(5, 10)  # Between 32 and 1024
        output = 2 ** ri(1, 4)      # Between 2 and 16
        
        if activation is None:
            activation = random.choice(self.activation_names)
            
        if dropout is None:
            if random.choice([True, False]):
                # Use a single dropout value
                dropout_value = random.uniform(0.1, 0.5)
            else:
                # Use a list of dropout values
                dropout_value = [random.uniform(0.1, 0.5) for _ in range(num_layers - 1)]
        elif dropout is False:
            dropout_value = None
        else:
            dropout_value = dropout
            
        return ExponentialFCBlock(
            output=output,
            in_features=in_features,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout_value
        )
    
    def _generate_random_generic_residual_fc_block(self, num_layers=None, 
                                                  activation=None, 
                                                  dropout=None,
                                                  force_residual=None,
                                                  matching_dimensions=None):
        """
        Generate a random GenericResidualFCBlock with configurable parameters
        """
        if num_layers is None:
            num_layers = ri(2, 5)
            
        in_features = ri(10, 100)
        output = ri(2, 20)
        
        # Handle matching_dimensions parameter
        if matching_dimensions is not None:
            if matching_dimensions:
                output = in_features
            else:
                while output == in_features:
                    output = ri(2, 20)
        
        if activation is None:
            activation = random.choice(self.activation_names)
            
        units = self._generate_random_units(in_features, output, num_layers)
        
        if dropout is None:
            if random.choice([True, False]):
                # Use a single dropout value
                dropout_value = random.uniform(0.1, 0.5)
            else:
                # Use a list of dropout values
                dropout_value = [random.uniform(0.1, 0.5) for _ in range(num_layers - 1)]
        elif dropout is False:
            dropout_value = None
        else:
            dropout_value = dropout
            
        if force_residual is None:
            force_residual = random.choice([True, False])
            
        return GenericResidualFCBlock(
            output=output,
            in_features=in_features,
            num_layers=num_layers,
            units=units,
            activation=activation,
            dropout=dropout_value,
            force_residual=force_residual
        )
    
    def _generate_random_exponential_residual_fc_block(self, num_layers=None, 
                                                      activation=None, 
                                                      dropout=None,
                                                      force_residual=None):
        """
        Generate a random ExponentialResidualFCBlock with configurable parameters
        """
        if num_layers is None:
            num_layers = ri(2, 5)
            
        # Ensure that we have a wide enough range between in_features and output
        # for the exponential scaling to work well
        in_features = 2 ** ri(5, 10)  # Between 32 and 1024
        output = 2 ** ri(1, 4)      # Between 2 and 16
        
        if activation is None:
            activation = random.choice(self.activation_names)
            
        if dropout is None:
            if random.choice([True, False]):
                # Use a single dropout value
                dropout_value = random.uniform(0.1, 0.5)
            else:
                # Use a list of dropout values
                dropout_value = [random.uniform(0.1, 0.5) for _ in range(num_layers - 1)]
        elif dropout is False:
            dropout_value = None
        else:
            dropout_value = dropout
            
        if force_residual is None:
            force_residual = random.choice([True, False])
            
        return ExponentialResidualFCBlock(
            output=output,
            in_features=in_features,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout_value,
            force_residual=force_residual
        )
    
    def test_basic_linear_block_dimension_analysis(self):
        """Test that the dimension analyzer correctly predicts BasicLinearBlock output shapes"""
        for _ in range(5000):
            # Create a random BasicLinearBlock
            block = self._generate_random_basic_linear_block()
            
            # Create input with random batch size
            batch_size = ri(2, 16)  # Avoid batch size 1 which can cause BatchNorm issues in train mode
            input_shape = (batch_size, block.in_features)
            input_tensor = torch.randn(input_shape)
            
            # Set to eval mode to handle batch norm with any batch size
            block.eval()
            
            # Get actual output shape
            actual_output = block(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            # Get predicted shape using the analyzer
            predicted_shape = self.analyzer.analyse_dimensions(input_shape, block)
            
            # Assert that the predicted shape matches the actual shape
            self.assertEqual(predicted_shape, actual_shape, 
                            f"Shape mismatch for BasicLinearBlock: predicted {predicted_shape}, actual {actual_shape}")
            
            # Additional test with batch size 1 (in eval mode)
            input_shape_single = (1, block.in_features)
            input_tensor_single = torch.randn(input_shape_single)
            
            actual_output_single = block(input_tensor_single)
            actual_shape_single = tuple(actual_output_single.shape)
            
            predicted_shape_single = self.analyzer.analyse_dimensions(input_shape_single, block)
            
            self.assertEqual(predicted_shape_single, actual_shape_single, 
                            f"Shape mismatch for BasicLinearBlock with batch size 1: " +
                            f"predicted {predicted_shape_single}, actual {actual_shape_single}")
    
    def test_generic_fc_block_dimension_analysis(self):
        """Test that the dimension analyzer correctly predicts GenericFCBlock output shapes"""
        for _ in range(5000):
            # Create a random GenericFCBlock
            block = self._generate_random_generic_fc_block()
            
            # Create input with random batch size
            batch_size = ri(2, 16)  # Avoid batch size 1 which can cause BatchNorm issues in train mode
            input_shape = (batch_size, block.in_features)
            input_tensor = torch.randn(input_shape)
            
            # Set to eval mode to handle batch norm with any batch size
            block.eval()
            
            # Get actual output shape
            actual_output = block(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            # Get predicted shape using the analyzer
            predicted_shape = self.analyzer.analyse_dimensions(input_shape, block)
            
            # Assert that the predicted shape matches the actual shape
            self.assertEqual(predicted_shape, actual_shape, 
                            f"Shape mismatch for GenericFCBlock: predicted {predicted_shape}, actual {actual_shape}")
            
            # Additional test with batch size 1 (in eval mode)
            input_shape_single = (1, block.in_features)
            input_tensor_single = torch.randn(input_shape_single)
            
            actual_output_single = block(input_tensor_single)
            actual_shape_single = tuple(actual_output_single.shape)
            
            predicted_shape_single = self.analyzer.analyse_dimensions(input_shape_single, block)
            
            self.assertEqual(predicted_shape_single, actual_shape_single, 
                            f"Shape mismatch for GenericFCBlock with batch size 1: " +
                            f"predicted {predicted_shape_single}, actual {actual_shape_single}")
    
    def test_exponential_fc_block_dimension_analysis(self):
        """Test that the dimension analyzer correctly predicts ExponentialFCBlock output shapes"""
        for _ in range(5000):
            # Create a random ExponentialFCBlock
            block = self._generate_random_exponential_fc_block()
            
            # Create input with random batch size
            batch_size = ri(2, 16)  # Avoid batch size 1 which can cause BatchNorm issues in train mode
            input_shape = (batch_size, block.in_features)
            input_tensor = torch.randn(input_shape)
            
            # Set to eval mode to handle batch norm with any batch size
            block.eval()
            
            # Get actual output shape
            actual_output = block(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            # Get predicted shape using the analyzer
            predicted_shape = self.analyzer.analyse_dimensions(input_shape, block)
            
            # Assert that the predicted shape matches the actual shape
            self.assertEqual(predicted_shape, actual_shape, 
                            f"Shape mismatch for ExponentialFCBlock: predicted {predicted_shape}, actual {actual_shape}")
            
            # Additional test with batch size 1 (in eval mode)
            input_shape_single = (1, block.in_features)
            input_tensor_single = torch.randn(input_shape_single)
            
            actual_output_single = block(input_tensor_single)
            actual_shape_single = tuple(actual_output_single.shape)
            
            predicted_shape_single = self.analyzer.analyse_dimensions(input_shape_single, block)
            
            self.assertEqual(predicted_shape_single, actual_shape_single, 
                            f"Shape mismatch for ExponentialFCBlock with batch size 1: " +
                            f"predicted {predicted_shape_single}, actual {actual_shape_single}")
    
    # def test_generic_residual_fc_block_dimension_analysis(self):
    #     """Test that the dimension analyzer correctly predicts GenericResidualFCBlock output shapes"""
    #     for _ in range(50):
    #         # Test both with matching dimensions and different dimensions
    #         for matching in [True, False]:
    #             try:
    #                 # Create a random GenericResidualFCBlock
    #                 block = self._generate_random_generic_residual_fc_block(matching_dimensions=matching)
                    
    #                 # Create input with random batch size
    #                 batch_size = ri(2, 16)  # Avoid batch size 1 which can cause BatchNorm issues in train mode
    #                 input_shape = (batch_size, block.in_features)
    #                 input_tensor = torch.randn(input_shape)
                    
    #                 # Set to eval mode to handle batch norm with any batch size
    #                 block.eval()
                    
    #                 # Get actual output shape
    #                 actual_output = block(input_tensor)
    #                 actual_shape = tuple(actual_output.shape)
                    
    #                 # Get predicted shape using the analyzer
    #                 predicted_shape = self.analyzer.analyse_dimensions(input_shape, block)
                    
    #                 # Assert that the predicted shape matches the actual shape
    #                 self.assertEqual(predicted_shape, actual_shape, 
    #                                 f"Shape mismatch for GenericResidualFCBlock: predicted {predicted_shape}, actual {actual_shape}")
                    
    #                 # Additional test with batch size 1 (in eval mode)
    #                 input_shape_single = (1, block.in_features)
    #                 input_tensor_single = torch.randn(input_shape_single)
                    
    #                 actual_output_single = block(input_tensor_single)
    #                 actual_shape_single = tuple(actual_output_single.shape)
                    
    #                 predicted_shape_single = self.analyzer.analyse_dimensions(input_shape_single, block)
                    
    #                 self.assertEqual(predicted_shape_single, actual_shape_single, 
    #                                 f"Shape mismatch for GenericResidualFCBlock with batch size 1: " +
    #                                 f"predicted {predicted_shape_single}, actual {actual_shape_single}")
    #             except ValueError:
    #                 # Skip if we can't create a valid block with the requested constraints
    #                 continue
    
    # def test_exponential_residual_fc_block_dimension_analysis(self):
    #     """Test that the dimension analyzer correctly predicts ExponentialResidualFCBlock output shapes"""
    #     for _ in range(50):
    #         # Create a random ExponentialResidualFCBlock
    #         block = self._generate_random_exponential_residual_fc_block()
            
    #         # Create input with random batch size
    #         batch_size = ri(2, 16)  # Avoid batch size 1 which can cause BatchNorm issues in train mode
    #         input_shape = (batch_size, block.in_features)
    #         input_tensor = torch.randn(input_shape)
            
    #         # Set to eval mode to handle batch norm with any batch size
    #         block.eval()
            
    #         # Get actual output shape
    #         actual_output = block(input_tensor)
    #         actual_shape = tuple(actual_output.shape)
            
    #         # Get predicted shape using the analyzer
    #         predicted_shape = self.analyzer.analyse_dimensions(input_shape, block)
            
    #         # Assert that the predicted shape matches the actual shape
    #         self.assertEqual(predicted_shape, actual_shape, 
    #                         f"Shape mismatch for ExponentialResidualFCBlock: predicted {predicted_shape}, actual {actual_shape}")
            
    #         # Additional test with batch size 1 (in eval mode)
    #         input_shape_single = (1, block.in_features)
    #         input_tensor_single = torch.randn(input_shape_single)
            
    #         actual_output_single = block(input_tensor_single)
    #         actual_shape_single = tuple(actual_output_single.shape)
            
    #         predicted_shape_single = self.analyzer.analyse_dimensions(input_shape_single, block)
            
    #         self.assertEqual(predicted_shape_single, actual_shape_single, 
    #                         f"Shape mismatch for ExponentialResidualFCBlock with batch size 1: " +
    #                         f"predicted {predicted_shape_single}, actual {actual_shape_single}")


if __name__ == '__main__':
    pu.seed_everything(42)
    unittest.main()
