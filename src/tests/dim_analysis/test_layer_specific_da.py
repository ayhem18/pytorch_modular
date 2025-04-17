"""
This script contains functionalities to test the code written in both
'layer specific' and 'dimension_analyser' scripts
"""

import torch
import unittest

from random import randint as ri


from mypt.code_utils import pytorch_utils as pu
from mypt.dimensions_analysis import layer_specific as ls
from mypt.backbones import resnetFeatureExtractor as res_fe
from mypt.randomGeneration.randomLayerGenerator import RandomLayerGenerator
from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser, _STATIC


class TestLayerSpecific(unittest.TestCase):
    def setUp(self):
        self.analyser = DimensionsAnalyser(method=_STATIC) 
        self.random_layer_generator = RandomLayerGenerator()
    def _generate_random_input_tensor(self, batch_size, channels, height, width=None):
        """Generate a random input tensor with the specified dimensions"""
        if width is None:
            width = height
        return torch.randn(batch_size, channels, height, width)


    # @unittest.skip("already tested")
    def test_conv2d_output(self):
        """Test that Conv2d output dimensions are correctly computed"""

        test_batch_one = False
        for _ in range(1000):
            # generate the random parameters
            batch_size = ri(1, 50)
            if batch_size == 1:
                test_batch_one = True 

            conv_layer = self.random_layer_generator.generate_random_conv_layer()

            input_tensor = torch.randn(batch_size, conv_layer.in_channels, 
                                       ri(conv_layer.kernel_size[0] + 10, conv_layer.kernel_size[0] + 50), 
                                       ri(conv_layer.kernel_size[1] + 10, conv_layer.kernel_size[1] + 50), 
                                       requires_grad=False)

            input_shape = tuple(input_tensor.shape)
            
            # Compute expected output shape using dimension analyzer
            expected_shape = self.analyser.analyse_dimensions(input_shape, conv_layer)
            
            # Get actual output shape
            actual_output = conv_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            # Assert shapes match
            self.assertEqual(actual_shape, expected_shape, 
                            f"Conv2d shape mismatch: got {actual_shape}, expected {expected_shape}")

        # make sure to test the batch size one case
        if test_batch_one:
            conv_layer = self.random_layer_generator.generate_random_conv_layer()

            input_tensor = torch.randn(1, conv_layer.in_channels, 
                                       ri(conv_layer.kernel_size[0] + 10, conv_layer.kernel_size[0] + 50), 
                                       ri(conv_layer.kernel_size[1] + 10, conv_layer.kernel_size[1] + 50), 
                                       requires_grad=False)
            input_shape = tuple(input_tensor.shape)

            expected_shape = self.analyser.analyse_dimensions(input_shape, conv_layer)
            actual_output = conv_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            self.assertEqual(actual_shape, expected_shape, 
                            f"Conv2d shape mismatch: got {actual_shape}, expected {expected_shape}")


    # @unittest.skip("already tested")
    def test_average_pool2d_output(self):
        """Test that AvgPool2d output dimensions are correctly computed"""
        test_batch_one = False
        for _ in range(1000):
            # Generate the random parameters
            batch_size = ri(1, 50)
            if batch_size == 1:
                test_batch_one = True 
                
            channels = ri(1, 64)
            pool_layer = self.random_layer_generator.generate_random_avg_pool_layer()
            
            # Create a random input tensor
            input_tensor = torch.randn(batch_size, channels, 
                                       ri(pool_layer.kernel_size + 10, pool_layer.kernel_size + 50), 
                                       ri(pool_layer.kernel_size + 10, pool_layer.kernel_size + 50), 
                                       requires_grad=False)
            
            input_shape = tuple(input_tensor.shape)
            
            # Compute expected output shape using dimension analyzer
            expected_shape = self.analyser.analyse_dimensions(input_shape, pool_layer)
            
            # Get actual output shape
            actual_output = pool_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            # Assert shapes match
            self.assertEqual(actual_shape, expected_shape, 
                            f"AvgPool2d shape mismatch: got {actual_shape}, expected {expected_shape}")
        
        # Make sure to test the batch size one case
        if not test_batch_one:
            channels = ri(1, 64)
            pool_layer = self.generate_random_avg_pool_layer()
            
            height = ri(pool_layer.kernel_size + 10, pool_layer.kernel_size + 50)
            width = ri(pool_layer.kernel_size + 10, pool_layer.kernel_size + 50)
            input_tensor = torch.randn(1, channels, height, width, requires_grad=False)
            input_shape = tuple(input_tensor.shape)
            
            expected_shape = self.analyser.analyse_dimensions(input_shape, pool_layer)
            actual_output = pool_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            self.assertEqual(actual_shape, expected_shape, 
                            f"AvgPool2d shape mismatch: got {actual_shape}, expected {expected_shape}")


    # @unittest.skip("already tested")
    def test_max_pool2d_output(self):
        """Test that MaxPool2d output dimensions are correctly computed"""
        test_batch_one = False
        for _ in range(1000):
            # Generate the random parameters
            batch_size = ri(1, 50)
            if batch_size == 1:
                test_batch_one = True 
                
            channels = ri(1, 64)
            pool_layer = self.random_layer_generator.generate_random_max_pool_layer()
            
            # Create a random input tensor
            input_tensor = torch.randn(batch_size, channels, 
                                       ri(pool_layer.kernel_size + 10, pool_layer.kernel_size + 50), 
                                       ri(pool_layer.kernel_size + 10, pool_layer.kernel_size + 50), 
                                       requires_grad=False)

            input_shape = tuple(input_tensor.shape)
            
            # Compute expected output shape using dimension analyzer
            expected_shape = self.analyser.analyse_dimensions(input_shape, pool_layer)
            
            # Get actual output shape
            actual_output = pool_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            # Assert shapes match
            self.assertEqual(actual_shape, expected_shape, 
                            f"MaxPool2d shape mismatch: got {actual_shape}, expected {expected_shape}")
        
        # Make sure to test the batch size one case
        if not test_batch_one:
            channels = ri(1, 64)
            pool_layer = self.generate_random_max_pool_layer()
            
            input_tensor = torch.randn(1, channels, 
                                       ri(pool_layer.kernel_size + 10, pool_layer.kernel_size + 50), 
                                       ri(pool_layer.kernel_size + 10, pool_layer.kernel_size + 50), 
                                       requires_grad=False)
            input_shape = tuple(input_tensor.shape)
            
            expected_shape = self.analyser.analyse_dimensions(input_shape, pool_layer)
            actual_output = pool_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            self.assertEqual(actual_shape, expected_shape, 
                            f"MaxPool2d shape mismatch: got {actual_shape}, expected {expected_shape}")


    # @unittest.skip("already tested")
    def test_adaptive_average_pool2d_output(self):
        """Test that AdaptiveAvgPool2d output dimensions are correctly computed"""
        test_batch_one = False
        for _ in range(1000):
            # Generate the random parameters
            batch_size = ri(1, 50)
            if batch_size == 1:
                test_batch_one = True 
                
            channels = ri(1, 64)
            pool_layer = self.random_layer_generator.generate_random_adaptive_pool_layer()
            
            # Create a random input tensor
            input_tensor = torch.randn(batch_size, channels, 
                                       ri(pool_layer.output_size[0] + 10, pool_layer.output_size[0] + 50), 
                                       ri(pool_layer.output_size[1] + 10, pool_layer.output_size[1] + 50), 
                                       requires_grad=False)
            
            input_shape = tuple(input_tensor.shape)
            
            # Compute expected output shape using dimension analyzer
            expected_shape = self.analyser.analyse_dimensions(input_shape, pool_layer)
            
            # Get actual output shape
            actual_output = pool_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            # Assert shapes match
            self.assertEqual(actual_shape, expected_shape, 
                            f"AdaptiveAvgPool2d shape mismatch: got {actual_shape}, expected {expected_shape}")
        
        # Make sure to test the batch size one case
        if not test_batch_one:
            channels = ri(1, 64)
            pool_layer = self.generate_random_adaptive_pool_layer()
            
            input_tensor = self._generate_random_input_tensor(1, channels, pool_layer.output_size[0])
            input_shape = tuple(input_tensor.shape)
            
            expected_shape = self.analyser.analyse_dimensions(input_shape, pool_layer)
            actual_output = pool_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            self.assertEqual(actual_shape, expected_shape, 
                            f"AdaptiveAvgPool2d shape mismatch: got {actual_shape}, expected {expected_shape}")


    # @unittest.skip("already tested")
    def test_linear_output(self):
        """Test that Linear output dimensions are correctly computed"""
        test_batch_one = False
        for _ in range(1000):
            # Generate the random parameters
            batch_size = ri(1, 50)
            if batch_size == 1:
                test_batch_one = True 
                
            linear_layer = self.random_layer_generator.generate_random_linear_layer()
            
            # Create a random input tensor
            input_tensor = torch.randn(batch_size, linear_layer.in_features, requires_grad=False)
            input_shape = tuple(input_tensor.shape)
            
            # Compute expected output shape using dimension analyzer
            expected_shape = self.analyser.analyse_dimensions(input_shape, linear_layer)
            
            # Get actual output shape
            actual_output = linear_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            # Assert shapes match
            self.assertEqual(actual_shape, expected_shape, 
                            f"Linear shape mismatch: got {actual_shape}, expected {expected_shape}")
        
        # Make sure to test the batch size one case
        if not test_batch_one:
            linear_layer = self.random_layer_generator.generate_random_linear_layer()
            
            input_tensor = torch.randn(1, linear_layer.in_features, requires_grad=False)
            input_shape = tuple(input_tensor.shape)
            
            expected_shape = self.analyser.analyse_dimensions(input_shape, linear_layer)
            actual_output = linear_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            self.assertEqual(actual_shape, expected_shape, 
                            f"Linear shape mismatch: got {actual_shape}, expected {expected_shape}")


    # @unittest.skip("already tested")
    def test_flatten_output(self):
        """Test that Flatten output dimensions are correctly computed"""
        test_batch_one = False
        for _ in range(1000):
            # Generate the random parameters
            batch_size = ri(1, 50)
            if batch_size == 1:
                test_batch_one = True 
                
            flatten_layer = self.random_layer_generator.generate_random_flatten_layer()
            
            # Create a random input tensor with 4D shape (batch, channels, height, width)
            channels = ri(1, 16)
            height = ri(1, 10)
            width = ri(1, 10)
            
            input_tensor = torch.randn(batch_size, channels, height, width, requires_grad=False)
            input_shape = tuple(input_tensor.shape)
            
            # Compute expected output shape using dimension analyzer
            expected_shape = self.analyser.analyse_dimensions(input_shape, flatten_layer)
            
            # Get actual output shape
            actual_output = flatten_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            # Assert shapes match
            self.assertEqual(actual_shape, expected_shape, 
                            f"Flatten shape mismatch: got {actual_shape}, expected {expected_shape}")
        
        # Make sure to test the batch size one case
        if not test_batch_one:
            flatten_layer = self.random_layer_generator.generate_random_flatten_layer()
            
            channels = ri(1, 16)
            height = ri(1, 10)
            width = ri(1, 10)
            input_tensor = self._generate_random_input_tensor(1, channels, height)
            input_shape = tuple(input_tensor.shape)
            
            expected_shape = self.analyser.analyse_dimensions(input_shape, flatten_layer)
            actual_output = flatten_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            self.assertEqual(actual_shape, expected_shape, 
                            f"Flatten shape mismatch: got {actual_shape}, expected {expected_shape}")

    
    def test_batchnorm1d_output(self):
        """Test that BatchNorm1d output dimensions are correctly computed"""
        test_batch_one = False
        for _ in range(1000):
            batch_size = ri(1, 50)
            if batch_size == 1:
                test_batch_one = True
                
            bn_layer = self.random_layer_generator.generate_random_batchnorm1d_layer()
            
            # Create a random input tensor with correct shape
            input_tensor = torch.randn(batch_size, bn_layer.num_features, requires_grad=False)
            input_shape = tuple(input_tensor.shape)
            
            # Compute expected output shape
            expected_shape = self.analyser.analyse_dimensions(input_shape, bn_layer)
            
            # Set to eval mode to avoid batch size issues
            bn_layer.eval()

            # after going through the code, it seems that the batchnorm layer would still be considered in training mode 
            # if both running_mean and running_var fields are None
            if bn_layer.running_mean is None and bn_layer.running_var is None:
                bn_layer.running_mean = torch.zeros(bn_layer.num_features)
                bn_layer.running_var = torch.ones(bn_layer.num_features) 

            # at this point, the batchnorm layer should fully in eval mode
            actual_output = bn_layer.forward(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            # Assert shapes match
            self.assertEqual(actual_shape, expected_shape, 
                            f"BatchNorm1d shape mismatch: got {actual_shape}, expected {expected_shape}")
        
        # Make sure to test batch size one case
        if not test_batch_one:
            bn_layer = self.random_layer_generator.generate_random_batchnorm1d_layer()
            input_tensor = torch.randn(1, bn_layer.num_features, requires_grad=False)
            input_shape = tuple(input_tensor.shape)
            
            # Set to eval mode to avoid batch size issues
            bn_layer.eval()
            # after going through the code, it seems that the batchnorm layer would still be considered in training mode 
            # if both running_mean and running_var fields are None
            if bn_layer.running_mean is None and bn_layer.running_var is None:
                bn_layer.running_mean = torch.zeros(bn_layer.num_features)
                bn_layer.running_var = torch.ones(bn_layer.num_features) 
            
            expected_shape = self.analyser.analyse_dimensions(input_shape, bn_layer)
            actual_output = bn_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            self.assertEqual(actual_shape, expected_shape, 
                            f"BatchNorm1d shape mismatch: got {actual_shape}, expected {expected_shape}")

    
    def test_batchnorm2d_output(self):
        """Test that BatchNorm2d output dimensions are correctly computed"""
        test_batch_one = False

        for _ in range(1000):
            batch_size = ri(1, 50)
            if batch_size == 1:
                test_batch_one = True
                
            bn_layer = self.random_layer_generator.generate_random_batchnorm2d_layer()
            
            # Create a random input tensor with correct shape
            height = ri(8, 32)
            width = ri(8, 32)
            input_tensor = torch.randn(batch_size, bn_layer.num_features, height, width, requires_grad=False)
            input_shape = tuple(input_tensor.shape)
            
            # Compute expected output shape
            expected_shape = self.analyser.analyse_dimensions(input_shape, bn_layer)
            
            # Set to eval mode to avoid batch size issues
            bn_layer.eval()
            
            # Get actual output shape
            actual_output = bn_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            # Assert shapes match
            self.assertEqual(actual_shape, expected_shape, 
                            f"BatchNorm2d shape mismatch: got {actual_shape}, expected {expected_shape}")
        
        # Make sure to test batch size one case
        if not test_batch_one:
            bn_layer = self.random_layer_generator.generate_random_batchnorm2d_layer()
            height = ri(8, 32)
            width = ri(8, 32)
            input_tensor = torch.randn(1, bn_layer.num_features, height, width, requires_grad=False)
            input_shape = tuple(input_tensor.shape)
            
            bn_layer.eval()  # BatchNorm2d needs eval mode for batch size 1
            
            expected_shape = self.analyser.analyse_dimensions(input_shape, bn_layer)
            actual_output = bn_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            self.assertEqual(actual_shape, expected_shape, 
                            f"BatchNorm2d shape mismatch: got {actual_shape}, expected {expected_shape}")
    
    
    def test_layernorm_output(self):
        """Test that LayerNorm output dimensions are correctly computed"""
        test_batch_one = False
        for _ in range(1000):
            batch_size = ri(1, 50)
            if batch_size == 1:
                test_batch_one = True
                
            # First create input shape, then create matching LayerNorm
            feature_dim = ri(8, 32)
            seq_len = ri(8, 32)
            
            # 3D input: (batch_size, feature_dim, seq_len)
            input_tensor = torch.randn(batch_size, feature_dim, seq_len, requires_grad=False)
            
            # LayerNorm typically normalizes over the last dimension
            ln_layer = self.random_layer_generator.generate_random_layernorm_layer(seq_len)
            
            input_shape = tuple(input_tensor.shape)
            
            # Compute expected output shape
            expected_shape = self.analyser.analyse_dimensions(input_shape, ln_layer)
            
            # Get actual output shape
            actual_output = ln_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            # Assert shapes match
            self.assertEqual(actual_shape, expected_shape, 
                            f"LayerNorm shape mismatch: got {actual_shape}, expected {expected_shape}")
        
        # Make sure to test batch size one case
        if not test_batch_one:
            feature_dim = ri(8, 32)
            seq_len = ri(8, 32)
            input_tensor = torch.randn(1, feature_dim, seq_len, requires_grad=False)
            
            ln_layer = self.random_layer_generator.generate_random_layernorm_layer(seq_len)
            input_shape = tuple(input_tensor.shape)
            
            expected_shape = self.analyser.analyse_dimensions(input_shape, ln_layer)
            actual_output = ln_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            self.assertEqual(actual_shape, expected_shape, 
                            f"LayerNorm shape mismatch: got {actual_shape}, expected {expected_shape}")
    
    
    def test_groupnorm_output(self):
        """Test that GroupNorm output dimensions are correctly computed"""
        test_batch_one = False
        for _ in range(1000):
            batch_size = ri(1, 50)
            if batch_size == 1:
                test_batch_one = True
                
            gn_layer = self.random_layer_generator.generate_random_groupnorm_layer()
            
            # Create a random input tensor with correct shape
            height = ri(8, 32)
            width = ri(8, 32)
            input_tensor = torch.randn(batch_size, gn_layer.num_channels, height, width, requires_grad=False)
            input_shape = tuple(input_tensor.shape)
            
            # Compute expected output shape
            expected_shape = self.analyser.analyse_dimensions(input_shape, gn_layer)
            
            # Get actual output shape
            actual_output = gn_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            # Assert shapes match
            self.assertEqual(actual_shape, expected_shape, 
                            f"GroupNorm shape mismatch: got {actual_shape}, expected {expected_shape}")
        
        # Make sure to test batch size one case
        if not test_batch_one:
            gn_layer = self.random_layer_generator.generate_random_groupnorm_layer()
            height = ri(8, 32)
            width = ri(8, 32)
            input_tensor = torch.randn(1, gn_layer.num_channels, height, width, requires_grad=False)
            input_shape = tuple(input_tensor.shape)
            
            expected_shape = self.analyser.analyse_dimensions(input_shape, gn_layer)
            actual_output = gn_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            self.assertEqual(actual_shape, expected_shape, 
                            f"GroupNorm shape mismatch: got {actual_shape}, expected {expected_shape}")
    

    def test_instancenorm2d_output(self):
        """Test that InstanceNorm2d output dimensions are correctly computed"""
        test_batch_one = False
        for _ in range(1000):
            batch_size = ri(1, 50)
            if batch_size == 1:
                test_batch_one = True
                
            in_layer = self.random_layer_generator.generate_random_instancenorm2d_layer()
            
            # Create a random input tensor with correct shape
            height = ri(8, 32)
            width = ri(8, 32)
            input_tensor = torch.randn(batch_size, in_layer.num_features, height, width, requires_grad=False)
            input_shape = tuple(input_tensor.shape)
            
            # Compute expected output shape
            expected_shape = self.analyser.analyse_dimensions(input_shape, in_layer)
            
            # Set to eval mode to avoid batch size issues
            in_layer.eval()
            
            # Get actual output shape
            actual_output = in_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            # Assert shapes match
            self.assertEqual(actual_shape, expected_shape, 
                            f"InstanceNorm2d shape mismatch: got {actual_shape}, expected {expected_shape}")
        
        # Make sure to test batch size one case
        if not test_batch_one:
            in_layer = self.random_layer_generator.generate_random_instancenorm2d_layer()
            height = ri(8, 32)
            width = ri(8, 32)
            input_tensor = torch.randn(1, in_layer.num_features, height, width, requires_grad=False)
            input_shape = tuple(input_tensor.shape)
            
            in_layer.eval()
            
            expected_shape = self.analyser.analyse_dimensions(input_shape, in_layer)
            actual_output = in_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            self.assertEqual(actual_shape, expected_shape, 
                            f"InstanceNorm2d shape mismatch: got {actual_shape}, expected {expected_shape}")


    def test_dropout_output(self):
        """Test that Dropout output dimensions are correctly computed"""
        test_batch_one = False
        for _ in range(1000):
            batch_size = ri(1, 50)
            if batch_size == 1:
                test_batch_one = True
                
            dropout_layer = self.random_layer_generator.generate_random_dropout_layer()
            
            # Create a random input tensor (can be any shape)
            features = ri(8, 32)
            input_tensor = torch.randn(batch_size, features, requires_grad=False)
            input_shape = tuple(input_tensor.shape)
            
            # Compute expected output shape
            expected_shape = self.analyser.analyse_dimensions(input_shape, dropout_layer)
            
            # Set to eval mode to make output deterministic (no dropout)
            dropout_layer.eval()
            
            # Get actual output shape
            actual_output = dropout_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            # Assert shapes match
            self.assertEqual(actual_shape, expected_shape, 
                            f"Dropout shape mismatch: got {actual_shape}, expected {expected_shape}")
        
        # Make sure to test batch size one case
        if not test_batch_one:
            dropout_layer = self.random_layer_generator.generate_random_dropout_layer()
            features = ri(8, 32)
            input_tensor = torch.randn(1, features, requires_grad=False)
            input_shape = tuple(input_tensor.shape)
            
            dropout_layer.eval()
            
            expected_shape = self.analyser.analyse_dimensions(input_shape, dropout_layer)
            actual_output = dropout_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            self.assertEqual(actual_shape, expected_shape, 
                            f"Dropout shape mismatch: got {actual_shape}, expected {expected_shape}")
    

    def test_dropout2d_output(self):
        """Test that Dropout2d output dimensions are correctly computed"""
        test_batch_one = False
        for _ in range(1000):
            batch_size = ri(1, 50)
            if batch_size == 1:
                test_batch_one = True
                
            dropout_layer = self.random_layer_generator.generate_random_dropout2d_layer()
            
            # Create a random input tensor (must be 4D for Dropout2d)
            channels = ri(1, 32)
            height = ri(8, 32)
            width = ri(8, 32)
            input_tensor = torch.randn(batch_size, channels, height, width, requires_grad=False)
            input_shape = tuple(input_tensor.shape)
            
            # Compute expected output shape
            expected_shape = self.analyser.analyse_dimensions(input_shape, dropout_layer)
            
            # Set to eval mode to make output deterministic (no dropout)
            dropout_layer.eval()
            
            # Get actual output shape
            actual_output = dropout_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            # Assert shapes match
            self.assertEqual(actual_shape, expected_shape, 
                            f"Dropout2d shape mismatch: got {actual_shape}, expected {expected_shape}")
        
        # Make sure to test batch size one case
        if not test_batch_one:
            dropout_layer = self.random_layer_generator.generate_random_dropout2d_layer()
            channels = ri(1, 32)
            height = ri(8, 32)
            width = ri(8, 32)
            input_tensor = torch.randn(1, channels, height, width, requires_grad=False)
            input_shape = tuple(input_tensor.shape)
            
            dropout_layer.eval()
            
            expected_shape = self.analyser.analyse_dimensions(input_shape, dropout_layer)
            actual_output = dropout_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            self.assertEqual(actual_shape, expected_shape, 
                            f"Dropout2d shape mismatch: got {actual_shape}, expected {expected_shape}")
    

    def test_activation_output(self):
        """Test that activation functions output dimensions are correctly computed"""
        test_batch_one = False
        for _ in range(1000):
            batch_size = ri(1, 50)
            if batch_size == 1:
                test_batch_one = True
                
            activation_layer = self.random_layer_generator.generate_random_activation_layer()
            
            # Create a random input tensor (can be any shape)
            features = ri(8, 32)
            input_tensor = torch.randn(batch_size, features, requires_grad=False)
            input_shape = tuple(input_tensor.shape)
            
            # Compute expected output shape
            expected_shape = self.analyser.analyse_dimensions(input_shape, activation_layer)
            
            # Get actual output shape
            actual_output = activation_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            # Assert shapes match
            self.assertEqual(actual_shape, expected_shape, 
                            f"Activation shape mismatch: got {actual_shape}, expected {expected_shape}")
        
        # Make sure to test batch size one case
        if not test_batch_one:
            activation_layer = self.random_layer_generator.generate_random_activation_layer()
            features = ri(8, 32)
            input_tensor = torch.randn(1, features, requires_grad=False)
            input_shape = tuple(input_tensor.shape)
            
            expected_shape = self.analyser.analyse_dimensions(input_shape, activation_layer)
            actual_output = activation_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            self.assertEqual(actual_shape, expected_shape, 
                            f"Activation shape mismatch: got {actual_shape}, expected {expected_shape}")
    

    def test_upsample_output(self):
        """Test that Upsample output dimensions are correctly computed"""
        test_batch_one = False
        for _ in range(1000):
            batch_size = ri(1, 50)
            if batch_size == 1:
                test_batch_one = True
                
            upsample_layer = self.random_layer_generator.generate_random_upsample_layer()
            
            # Create a random input tensor
            channels = ri(1, 16)
            height = ri(8, 32)
            width = ri(8, 32)
            input_tensor = torch.randn(batch_size, channels, height, width, requires_grad=False)
            
            # if upsample_layer.mode == 'linear':
            #     input_tensor = input_tensor[0,:,:,:][0]
                
            input_shape = tuple(input_tensor.shape)
            
            # Compute expected output shape
            expected_shape = self.analyser.analyse_dimensions(input_shape, upsample_layer)
            
            # Get actual output shape
            actual_output = upsample_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            # Assert shapes match
            self.assertEqual(actual_shape, expected_shape, 
                            f"Upsample shape mismatch: got {actual_shape}, expected {expected_shape}")
        
        # Make sure to test batch size one case
        if not test_batch_one:
            upsample_layer = self.random_layer_generator.generate_random_upsample_layer()
            channels = ri(1, 16)
            height = ri(8, 32)
            width = ri(8, 32)
            input_tensor = torch.randn(1, channels, height, width, requires_grad=False)
            input_shape = tuple(input_tensor.shape)
            
            expected_shape = self.analyser.analyse_dimensions(input_shape, upsample_layer)
            actual_output = upsample_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            self.assertEqual(actual_shape, expected_shape, 
                            f"Upsample shape mismatch: got {actual_shape}, expected {expected_shape}")
    

    def test_convtranspose2d_output(self):
        """Test that ConvTranspose2d output dimensions are correctly computed"""
        test_batch_one = False
        for _ in range(1000):
            batch_size = ri(1, 50)
            if batch_size == 1:
                test_batch_one = True
                
            conv_layer = self.random_layer_generator.generate_random_convtranspose2d_layer()
            
            # Create a random input tensor with correct channels
            height = ri(8, 32)
            width = ri(8, 32)
            input_tensor = torch.randn(batch_size, conv_layer.in_channels, height, width, requires_grad=False)
            input_shape = tuple(input_tensor.shape)
            
            # Compute expected output shape
            expected_shape = self.analyser.analyse_dimensions(input_shape, conv_layer)
            
            # Get actual output shape
            actual_output = conv_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            # Assert shapes match
            self.assertEqual(actual_shape, expected_shape, 
                            f"ConvTranspose2d shape mismatch: got {actual_shape}, expected {expected_shape}")
        
        # Make sure to test batch size one case
        if not test_batch_one:
            conv_layer = self.random_layer_generator.generate_random_convtranspose2d_layer()
            height = ri(8, 32)
            width = ri(8, 32)
            input_tensor = torch.randn(1, conv_layer.in_channels, height, width, requires_grad=False)
            input_shape = tuple(input_tensor.shape)
            
            expected_shape = self.analyser.analyse_dimensions(input_shape, conv_layer)
            actual_output = conv_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            self.assertEqual(actual_shape, expected_shape, 
                            f"ConvTranspose2d shape mismatch: got {actual_shape}, expected {expected_shape}")
    

    def test_embedding_output(self):
        """Test that Embedding output dimensions are correctly computed"""
        test_batch_one = False
        for _ in range(1000):
            batch_size = ri(1, 50)
            if batch_size == 1:
                test_batch_one = True
                
            embedding_layer = self.random_layer_generator.generate_random_embedding_layer()
            
            # Create a random input tensor with integer indices
            seq_len = ri(8, 32)
            input_tensor = torch.randint(0, embedding_layer.num_embeddings, (batch_size, seq_len), requires_grad=False)
            input_shape = tuple(input_tensor.shape)
            
            # Compute expected output shape
            expected_shape = self.analyser.analyse_dimensions(input_shape, embedding_layer)
            
            # Get actual output shape
            actual_output = embedding_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            # Assert shapes match
            self.assertEqual(actual_shape, expected_shape, 
                            f"Embedding shape mismatch: got {actual_shape}, expected {expected_shape}")
        
        # Make sure to test batch size one case
        if not test_batch_one:
            embedding_layer = self.random_layer_generator.generate_random_embedding_layer()
            seq_len = ri(8, 32)
            input_tensor = torch.randint(0, embedding_layer.num_embeddings, (1, seq_len), requires_grad=False)
            input_shape = tuple(input_tensor.shape)
            
            expected_shape = self.analyser.analyse_dimensions(input_shape, embedding_layer)
            actual_output = embedding_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            self.assertEqual(actual_shape, expected_shape, 
                            f"Embedding shape mismatch: got {actual_shape}, expected {expected_shape}") 



# class TestOutputAnalysis(unittest.TestCase):

#     # def test_output_analyser_components(self):
#     #     # device = get_default_device()
#     #     analyser = da.DimensionsAnalyser(method=da._STATIC)
#     #     cpu = 'cpu'
#     #     for _ in range(50):
#     #         in_channels, conv_block = rc.random_conv_block(return_in_c=True)
#     #         # conv_block.to()
#     #         conv_block.eval()
#     #         # create the input shape
#     #
#     #         # settings the height and width to significantly large numbers
#     #         # as the random settings of the convolutional block makes it likely for the dimensions to decrease
#     #         # extremely fast
#     #         random_height = ri(2000, 4000)
#     #         random_width = ri(2000, 4000)
#     #         random_batch = ri(1, 3)  # keeping the random_batch small for memory concerns (mainly with GPU)
#     #
#     #         input_shape = (random_batch, in_channels, random_height, random_width)
#     #         input_tensor = torch.ones(size=input_shape)
#     #         for m in conv_block:
#     #             shape_computed = analyser.analyse_dimensions(input_shape, m)
#     #             input_tensor = m.forward(input_tensor)
#     #             input_shape = tuple(input_tensor.size())
#     #
#     #             self.assertEqual(input_shape, shape_computed), "THE OUTPUT SHAPES ARE DIFFERENT"
#     #         # clear the GPU is needed
#     #         torch.cuda.empty_cache()
#     #         # deleting the unnecessary variable
#     #         del conv_block
#     #         del input_tensor

#     # def test_output_with_random_network(self):
#     #     # create a module analyser
#     #     # make sure to use static
#     #     analyser = da.DimensionsAnalyser(method=da._STATIC)
#     #     # 50 test cases should be good enough
#     #     for _ in range(50 ):
#     #         in_channels, conv_block = rc.random_conv_block(return_in_c=True)
#     #         conv_block.eval()
#     #         # create the input shape
#     #         random_batch = ri(1, 2)  # keeping the random_batch small for memory concerns (mainly with GPU)
#     #         random_height = ri(1000, 2000)
#     #         random_width = ri(1000, 2000)
#     #
#     #         input_shape = (random_batch, in_channels, random_height, random_width)
#     #         computed_output_shape = analyser.analyse_dimensions(input_shape, conv_block)
#     #
#     #         input_tensor = torch.ones(size=input_shape)
#     #         output_tensor = conv_block.forward(input_tensor)
#     #
#     #         self.assertEqual(tuple(output_tensor.size()), computed_output_shape), "THE OUTPUT SHAPES ARE DIFFERENT"
#     #         # make sure to clear the gpu
#     #         torch.cuda.empty_cache()
#     #         # delete the variables
#     #         del conv_block
#     #         del input_tensor
#
#     # def test_output_pretrained_network(self):
#     #     device = pu.get_default_device()
#     #
#     #     if device != 'cuda':
#     #         raise ValueError("This test must run on a GPU")
#     #
#     #     layers = list(range(1, 5))
#     #
#     #     for v in layers:
#     #         feature_extractor = res_fe.ResNetFeatureExtractor(num_layers=v).to(device)
#     #         feature_extractor.eval()
#     #
#     #         for _ in range(50):
#     #             # create the input shape
#     #             random_batch = ri(1, 3)  # keeping the random_batch small for memory concerns (mainly with GPU)
#     #             random_height = ri(1000, 2000)
#     #             random_width = ri(1000, 2000)
#     #
#     #             # resnet expected a usual image with 3 channels
#     #             input_shape = (random_batch, 3, random_height, random_width)
#     #             computed_output_shape = self.analyser.analyse_dimensions(input_shape, feature_extractor)
#     #
#     #             input_tensor = torch.ones(size=input_shape, requires_grad=False).to(device) # move to 
#     #             output_tensor = feature_extractor.forward(input_tensor)
#     #
#     #             self.assertEqual(tuple(output_tensor.size()), computed_output_shape), "THE OUTPUT SHAPES ARE DIFFERENT"
#     #             # make sure to clear the gpu
#     #             torch.cuda.empty_cache()
#     #             # delete the output and input tensors
#     #             del input_tensor
#     #             del output_tensor
#     #         
#     #         # the next loop will define another feature extractor, make sure to free the memory
#     #         del feature_extractor


if __name__ == '__main__':
    pu.seed_everything()
    unittest.main()
