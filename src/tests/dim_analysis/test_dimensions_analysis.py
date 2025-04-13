"""
This script contains functionalities to test the code written in both
'layer specific' and 'dimension_analyser' scripts
"""

import torch
import unittest
import random
from random import randint as ri

from torch import nn
from mypt.dimensions_analysis import layer_specific as ls
from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser, _STATIC, _FORWARD
from mypt.backbones import resnetFeatureExtractor as res_fe
from mypt.code_utilities import pytorch_utilities as pu


class TestLayerSpecific(unittest.TestCase):
    def setUp(self):
        self.analyser = DimensionsAnalyser(method=_STATIC) 


    def _generate_random_conv_layer(self) -> nn.Conv2d:
        """
        This function generates a random Conv2d layer with random parameters
        """
        # Create random parameters for Conv2d
        in_channels = ri(1, 64)
        out_channels = ri(1, 64)
        kernel_size = ri(1, 7) 
        stride = ri(1, 3)
        padding = ri(0, 3)
        dilation = ri(1, 2)
        
        # Create a Conv2d layer with these parameters
        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )

        return conv_layer        

    def _generate_random_avg_pool_layer(self) -> nn.AvgPool2d:
        """
        This function generates a random AvgPool2d layer with random parameters
        """
        kernel_size = ri(3, 11)
        stride = ri(1, 3)
        padding = ri(0, 2)
        padding = min(padding, kernel_size // 2)

        # Create an AvgPool2d layer with these parameters
        pool_layer = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        return pool_layer
        
    def _generate_random_max_pool_layer(self) -> nn.MaxPool2d:
        """
        This function generates a random MaxPool2d layer with random parameters
        """
        kernel_size = ri(3, 11)
        stride = ri(1, 3)
        padding = random.choice([0, 1, 2])
        padding = min(padding, kernel_size // 2)

        dilation = ri(1, 2)
        
        # Create a MaxPool2d layer with these parameters
        pool_layer = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )

        return pool_layer
        
    def _generate_random_adaptive_pool_layer(self) -> nn.AdaptiveAvgPool2d:
        """
        This function generates a random AdaptiveAvgPool2d layer with random parameters
        """
        output_height = ri(1, 10)
        output_width = ri(1, 10)
        
        # Create an AdaptiveAvgPool2d layer with these parameters
        pool_layer = nn.AdaptiveAvgPool2d((output_height, output_width))
        
        return pool_layer
        
    def _generate_random_linear_layer(self) -> nn.Linear:
        """
        This function generates a random Linear layer with random parameters
        """
        in_features = ri(1, 100)
        out_features = ri(1, 100)
        
        # Create a Linear layer with these parameters
        linear_layer = nn.Linear(in_features, out_features)
        
        return linear_layer
        
    def _generate_random_flatten_layer(self) -> nn.Flatten:
        """
        This function generates a random Flatten layer with random parameters
        """
        start_dim = random.choice([1, 2])
        end_dim = -1
        
        # Create a Flatten layer with these parameters
        flatten_layer = nn.Flatten(start_dim=start_dim, end_dim=end_dim)
        
        return flatten_layer


    def test_conv2d_output(self):
        """Test that Conv2d output dimensions are correctly computed"""

        test_batch_one = False
        for _ in range(1000):
            # generate the random parameters
            batch_size = ri(1, 50)
            if batch_size == 1:
                test_batch_one = True 

            conv_layer = self._generate_random_conv_layer()

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
            conv_layer = self._generate_random_conv_layer()

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

    def test_average_pool2d_output(self):
        """Test that AvgPool2d output dimensions are correctly computed"""
        test_batch_one = False
        for _ in range(1000):
            # Generate the random parameters
            batch_size = ri(1, 50)
            if batch_size == 1:
                test_batch_one = True 
                
            channels = ri(1, 64)
            pool_layer = self._generate_random_avg_pool_layer()
            
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
            pool_layer = self._generate_random_avg_pool_layer()
            
            height = ri(pool_layer.kernel_size + 10, pool_layer.kernel_size + 50)
            width = ri(pool_layer.kernel_size + 10, pool_layer.kernel_size + 50)
            input_tensor = torch.randn(1, channels, height, width, requires_grad=False)
            input_shape = tuple(input_tensor.shape)
            
            expected_shape = self.analyser.analyse_dimensions(input_shape, pool_layer)
            actual_output = pool_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            self.assertEqual(actual_shape, expected_shape, 
                            f"AvgPool2d shape mismatch: got {actual_shape}, expected {expected_shape}")

    def test_max_pool2d_output(self):
        """Test that MaxPool2d output dimensions are correctly computed"""
        test_batch_one = False
        for _ in range(1000):
            # Generate the random parameters
            batch_size = ri(1, 50)
            if batch_size == 1:
                test_batch_one = True 
                
            channels = ri(1, 64)
            pool_layer = self._generate_random_max_pool_layer()
            
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
            pool_layer = self._generate_random_max_pool_layer()
            
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

    def test_adaptive_average_pool2d_output(self):
        """Test that AdaptiveAvgPool2d output dimensions are correctly computed"""
        test_batch_one = False
        for _ in range(1000):
            # Generate the random parameters
            batch_size = ri(1, 50)
            if batch_size == 1:
                test_batch_one = True 
                
            channels = ri(1, 64)
            pool_layer = self._generate_random_adaptive_pool_layer()
            
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
            pool_layer = self._generate_random_adaptive_pool_layer()
            
            input_tensor = self._generate_random_input_tensor(1, channels, pool_layer.output_size[0])
            input_shape = tuple(input_tensor.shape)
            
            expected_shape = self.analyser.analyse_dimensions(input_shape, pool_layer)
            actual_output = pool_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            self.assertEqual(actual_shape, expected_shape, 
                            f"AdaptiveAvgPool2d shape mismatch: got {actual_shape}, expected {expected_shape}")

    def test_linear_output(self):
        """Test that Linear output dimensions are correctly computed"""
        test_batch_one = False
        for _ in range(1000):
            # Generate the random parameters
            batch_size = ri(1, 50)
            if batch_size == 1:
                test_batch_one = True 
                
            linear_layer = self._generate_random_linear_layer()
            
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
            linear_layer = self._generate_random_linear_layer()
            
            input_tensor = torch.randn(1, linear_layer.in_features, requires_grad=False)
            input_shape = tuple(input_tensor.shape)
            
            expected_shape = self.analyser.analyse_dimensions(input_shape, linear_layer)
            actual_output = linear_layer(input_tensor)
            actual_shape = tuple(actual_output.shape)
            
            self.assertEqual(actual_shape, expected_shape, 
                            f"Linear shape mismatch: got {actual_shape}, expected {expected_shape}")

    def test_flatten_output(self):
        """Test that Flatten output dimensions are correctly computed"""
        test_batch_one = False
        for _ in range(1000):
            # Generate the random parameters
            batch_size = ri(1, 50)
            if batch_size == 1:
                test_batch_one = True 
                
            flatten_layer = self._generate_random_flatten_layer()
            
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
            flatten_layer = self._generate_random_flatten_layer()
            
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
