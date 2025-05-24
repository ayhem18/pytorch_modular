import torch
import random
import unittest

import torch.nn as nn

from typing import Optional

from tests.custom_base_test import CustomModuleBaseTest
from mypt.building_blocks.auxiliary.film_block import OneDimFiLMBlock, ThreeDimFiLMBlock


class TestOneDimFiLMBlock(CustomModuleBaseTest):
    """Test class for OneDimFiLMBlock implementation"""
    
    def setUp(self):
        """Set up test parameters"""
        # Default parameters for testing
        self.batch_size_range = (1, 4)
        self.channels_range = (1, 32)
        self.height_range = (8, 64)
        self.width_range = (8, 64)
        self.condition_dim_range = (4, 64)
        self.activations = [nn.ReLU, nn.LeakyReLU, nn.Tanh]
        
    def _get_random_film_block(self, with_activation=True):
        """Generate a random FiLM block for testing"""
        batch_size = random.randint(*self.batch_size_range)
            
        channels = random.randint(*self.channels_range)

        height = random.randint(*self.height_range)
        width = random.randint(*self.width_range)
        condition_dim = random.randint(*self.condition_dim_range)
        
        activation = random.choice(self.activations) if with_activation else 'relu'
        activation_params = {'inplace': True} if activation in [nn.ReLU, nn.LeakyReLU] else {}
        
        film_block = OneDimFiLMBlock(
            out_channels=channels,
            cond_dimension=condition_dim,

            normalization='batchnorm2d',
            normalization_params={'num_features': channels},
            
            activation=activation,
            activation_params=activation_params,
        )
        
        return film_block, batch_size, channels, height, width, condition_dim
    
    def _get_valid_input(self, 
                         film_block: OneDimFiLMBlock, 
                         batch_size: int,
                         height: Optional[int] = None,
                         width: Optional[int] = None    ):
        """Generate valid input tensors for testing"""

        channels = film_block.out_channels
        condition_dim = film_block._film_layer[0].in_features

        height = height or random.randint(*self.height_range)
        width = width or random.randint(*self.width_range)

        x = torch.randn(batch_size, channels, height, width)
        condition = torch.randn(batch_size, condition_dim)
        
        return x, condition
    
    def test_initialization(self):
        """Test that FiLM block initializes correctly"""
        for _ in range(100):
            channels = random.randint(*self.channels_range)
            condition_dim = random.randint(*self.condition_dim_range)
            activation = random.choice(self.activations)
            
            film_block = OneDimFiLMBlock(
                normalization='batchnorm2d',
                normalization_params={'num_features': channels},
                activation=activation,
                activation_params={},
                out_channels=channels,
                cond_dimension=condition_dim
            )
            
            self.assertEqual(film_block.out_channels, channels)
            self.assertEqual(film_block.cond_dimension, condition_dim)
            self.assertIsInstance(film_block._activation, activation)
            self.assertIsInstance(film_block._normalization, nn.BatchNorm2d) 

            # Check that the film layer is correctly initialized
            self.assertIsInstance(film_block._film_layer, nn.Sequential)
            self.assertEqual(film_block._film_layer[0].in_features, condition_dim)
            self.assertEqual(film_block._film_layer[-1].out_features, channels * 2) 
            
            # Check activation
            if activation:
                self.assertIsInstance(film_block._activation, activation)
            else:
                self.assertIsNotNone(film_block._activation)


    def test_forward_pass_shape(self):
        """Test that forward pass produces output with expected shape"""
        for _ in range(100):
            film_block, batch_size, channels, height, width, condition_dim = self._get_random_film_block()
            x, condition = self._get_valid_input(film_block, batch_size, height, width)
            
            film_block.eval()

            with torch.no_grad():    
                output = film_block(x, condition)
                self.assertEqual(output.shape, (batch_size, channels, height, width))


    def test_film_zero_weights_with_bias(self):
        """Test the case where FiLM weights are zero and bias is set to a value 'a'"""

        for a in torch.linspace(1, 100, 1000):
            with self.subTest(f"Testing with bias value a={a}"):
                film_block, batch_size, _, height, width, _= self._get_random_film_block(with_activation=False) # to use the ReLU activation
                x, condition = self._get_valid_input(film_block, batch_size, height, width)
                

                # make sure all the values in 'x' are positive 
                # so that passing it through the ReLu activation wouldn't change the values 
                x = torch.abs(x) 

                # Set weights to zero and bias to 'a'
                with torch.no_grad():
                    # set the weights of the first linear layer to zero
                    film_block._film_layer[0].weight.fill_(0.0)
                    # set the bias of the first linear layer to 'a'
                    film_block._film_layer[0].bias.fill_(a)

                    # set the weights of the second linear layer to zero
                    film_block._film_layer[2].weight.fill_(0.0)
                    # set the bias of the second linear layer to 'a'
                    film_block._film_layer[2].bias.fill_(a)
                
                self.assertTrue(torch.allclose(film_block._film_layer.forward(condition), torch.ones(batch_size, 2 * film_block.out_channels) * a))

                film_block._normalization = nn.Identity()

                output = film_block(x, condition)
                
                # calculate the expected output
                expected_output = x * (a) + a
                
                # Check that output matches expected
                self.assertTrue(torch.allclose(output, expected_output))

    # Tests inherited from CustomModuleBaseTest

    def test_film_conditioning(self):
        """Test that different conditioning values produce different outputs"""
        film_block, batch_size, channels, height, width, condition_dim = self._get_random_film_block()
        x = torch.randn(batch_size, channels, height, width)
        
        # Different condition tensors
        condition1 = torch.randn(batch_size, condition_dim)
        condition2 = torch.randn(batch_size, condition_dim)
        
        output1 = film_block(x, condition1)
        output2 = film_block(x, condition2)
        
        # The outputs should be different due to different conditioning
        self.assertFalse(torch.allclose(output1, output2))

    def test_eval_mode(self):
        """Test that eval mode is correctly set"""
        for _ in range(10):
            film_block, batch_size, _, height, width, _ = self._get_random_film_block()
            super()._test_eval_mode(film_block)
    
    def test_train_mode(self):
        """Test that train mode is correctly set"""
        for _ in range(100):     
            film_block, batch_size, _, height, width, _ = self._get_random_film_block()            
            super()._test_train_mode(film_block)
    

    def test_consistent_output_in_eval_mode(self):
        """Test that the FiLM block produces consistent output in eval mode"""
        for _ in range(100):
            film_block, batch_size, _, height, width, _ = self._get_random_film_block()
            x, condition = self._get_valid_input(film_block, batch_size, height, width)
            super()._test_consistent_output_in_eval_mode(film_block)

    
    def test_to_device(self):
        """Test device transfer functionality"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping GPU tests")
            
        for _ in range(100):
            film_block, batch_size, _, height, width, _ = self._get_random_film_block()
            x, condition = self._get_valid_input(film_block, batch_size, height, width)
            super()._test_to_device(film_block, x, condition)


    def _test_eval_mode(self, module, *args, **kwargs):
        """Custom implementation for FiLM modules"""
        super()._test_eval_mode(module, *args, **kwargs)


    def _test_train_mode(self, module, *args, **kwargs):
        """Custom implementation for FiLM modules"""
        super()._test_train_mode(module, *args, **kwargs)
    

    def test_consistent_output_in_eval_mode(self):
        """Test that the FiLM block produces consistent output in eval mode"""
        for _ in range(100):
            film_block, batch_size, _, height, width, _ = self._get_random_film_block()
            x, condition = self._get_valid_input(film_block, batch_size, height, width)
            super()._test_consistent_output_in_eval_mode(film_block, x, condition)


    def test_batch_size_one_in_train_mode(self):
        """Test that the FiLM block produces consistent output in train mode"""
        for _ in range(100):
            film_block, _, _, height, width, _ = self._get_random_film_block()
            x, condition = self._get_valid_input(film_block, 1, height, width)
            super()._test_batch_size_one_in_train_mode(film_block, x, condition)   


    def test_batch_size_one_in_eval_mode(self):
        """Test that the FiLM block produces consistent output in eval mode"""
        for _ in range(100):
            film_block, _, _, height, width, _ = self._get_random_film_block()
            x, condition = self._get_valid_input(film_block, 1, height, width)
            super()._test_batch_size_one_in_eval_mode(film_block, x, condition)


class TestThreeDimFiLMBlock(CustomModuleBaseTest):
    """Test class for ThreeDimFiLMBlock implementation"""
    
    def setUp(self):
        """Set up test parameters"""
        # Default parameters for testing
        self.batch_size_range = (1, 4)
        self.channels_range = (1, 32)
        self.height_range = (8, 32)
        self.width_range = (8, 32)
        self.condition_dim_range = (4, 64)
        self.activations = [nn.ReLU, nn.LeakyReLU, nn.Tanh]
        
    def _get_random_film_block(self, with_activation=True):
        """Generate a random FiLM block for testing"""
        batch_size = random.randint(*self.batch_size_range)
        channels = random.randint(*self.channels_range)
        height = random.randint(*self.height_range)
        width = random.randint(*self.width_range)
        condition_dim = random.randint(*self.condition_dim_range)
        
        activation = random.choice(self.activations) if with_activation else 'relu'
        activation_params = {'inplace': True} if activation in [nn.ReLU, nn.LeakyReLU] else {}
        
        film_block = ThreeDimFiLMBlock(
            out_channels=channels,
            cond_dimension=condition_dim,
            
            normalization='batchnorm2d',
            normalization_params={'num_features': channels},
            
            activation=activation,
            activation_params=activation_params,
        )
        
        return film_block, batch_size, channels, height, width, condition_dim
    
    def _get_valid_input(self, 
                         film_block: ThreeDimFiLMBlock, 
                         batch_size: int,
                         height: Optional[int] = None,
                         width: Optional[int] = None):
        """Generate valid input tensors for testing"""
        
        channels = film_block.out_channels
        condition_dim = film_block.cond_dimension
        
        height = height or random.randint(*self.height_range)
        width = width or random.randint(*self.width_range)
        
        x = torch.randn(batch_size, channels, height, width)
        condition = torch.randn(batch_size, condition_dim, height, width)
        
        return x, condition
    
    def test_initialization(self):
        """Test that FiLM block initializes correctly"""
        for _ in range(100):
            channels = random.randint(*self.channels_range)
            condition_dim = random.randint(*self.condition_dim_range)
            activation = random.choice(self.activations)
            
            film_block = ThreeDimFiLMBlock(
                normalization='batchnorm2d',
                normalization_params={'num_features': channels},
                activation=activation,
                activation_params={},
                out_channels=channels,
                cond_dimension=condition_dim
            )
            
            self.assertEqual(film_block.out_channels, channels)
            self.assertEqual(film_block.cond_dimension, condition_dim)
            self.assertIsInstance(film_block._activation, activation)
            self.assertIsInstance(film_block._normalization, nn.BatchNorm2d)
            
            # Check that the film layer is correctly initialized
            self.assertIsInstance(film_block._film_layer, nn.Sequential)
            # In the 3D case, it's a sequential with Conv2d layers
            self.assertIsInstance(film_block._film_layer[0], nn.Conv2d)
            self.assertEqual(film_block._film_layer[0].in_channels, condition_dim)
            self.assertEqual(film_block._film_layer[-1].out_channels, channels * 2)
            
            # Check activation
            if activation:
                self.assertIsInstance(film_block._activation, activation)
            else:
                self.assertIsNotNone(film_block._activation)
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces output with expected shape"""
        for _ in range(100):
            film_block, batch_size, channels, height, width, _ = self._get_random_film_block()
            x, condition = self._get_valid_input(film_block, batch_size, height, width)
            
            film_block.eval()
            
            with torch.no_grad():
                output = film_block(x, condition)
                self.assertEqual(output.shape, (batch_size, channels, height, width))
    
    def test_film_zero_weights_with_bias(self):
        """Test the case where FiLM weights are zero and bias is set to a value 'a'"""
        
        for a in torch.linspace(1, 100, 1000):
            with self.subTest(f"Testing with bias value a={a}"):
                film_block, batch_size, _, height, width, _ = self._get_random_film_block(with_activation=False) # to use the ReLU activation
                x, condition = self._get_valid_input(film_block, batch_size, height, width)
                
                # make sure all the values in 'x' are positive 
                # so that passing it through the ReLu activation wouldn't change the values
                x = torch.abs(x)
                
                # Set weights to zero and bias to 'a' for all Conv2d layers
                with torch.no_grad():
                    # For ThreeDimFiLMBlock, we need to set weights and biases for Conv2d layers
                    for layer in film_block._film_layer:
                        if isinstance(layer, nn.Conv2d):
                            layer.weight.fill_(0.0)
                            layer.bias.fill_(a)
                
                film_block._normalization = nn.Identity()
                
                output = film_block(x, condition)
                
                # Calculate expected output
                expected_output = x * (a) + a
                
                # Check that output matches expected
                self.assertTrue(torch.allclose(output, expected_output))
    
    def test_film_conditioning(self):
        """Test that different conditioning values produce different outputs"""
        film_block, batch_size, channels, height, width, condition_dim = self._get_random_film_block()
        x = torch.randn(batch_size, channels, height, width)
        
        # Different condition tensors
        condition1 = torch.randn(batch_size, condition_dim, height, width)
        condition2 = torch.randn(batch_size, condition_dim, height, width)
        
        output1 = film_block(x, condition1)
        output2 = film_block(x, condition2)
        
        # The outputs should be different due to different conditioning
        self.assertFalse(torch.allclose(output1, output2))
    
    # tests inherited from CustomModuleBaseTest
    def test_eval_mode(self):
        """Test that eval mode is correctly set"""
        for _ in range(100):
            film_block, batch_size, _, height, width, _ = self._get_random_film_block()
            super()._test_eval_mode(film_block)
    
    def test_train_mode(self):
        """Test that train mode is correctly set"""
        for _ in range(100):
            film_block, batch_size, _, height, width, _ = self._get_random_film_block()
            super()._test_train_mode(film_block)
    
    def test_consistent_output_in_eval_mode(self):
        """Test that the FiLM block produces consistent output in eval mode"""
        for _ in range(100):
            film_block, batch_size, _, height, width, _ = self._get_random_film_block()
            x, condition = self._get_valid_input(film_block, batch_size, height, width)
            super()._test_consistent_output_in_eval_mode(film_block, x, condition)
    
    def test_batch_size_one_in_train_mode(self):
        """Test that the FiLM block produces consistent output in train mode"""
        for _ in range(100):
            film_block, _, _, height, width, _ = self._get_random_film_block()
            x, condition = self._get_valid_input(film_block, 1, height, width)
            super()._test_batch_size_one_in_train_mode(film_block, x, condition)
            
    def test_batch_size_one_in_eval_mode(self):
        """Test that the FiLM block produces consistent output in eval mode"""
        for _ in range(100):
            film_block, _, _, height, width, _ = self._get_random_film_block()
            x, condition = self._get_valid_input(film_block, 1, height, width)
            super()._test_batch_size_one_in_eval_mode(film_block, x, condition)

    def test_to_device(self):
        """Test device transfer functionality"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping GPU tests")
            
        for _ in range(100):
            film_block, batch_size, _, height, width, _ = self._get_random_film_block()
            x, condition = self._get_valid_input(film_block, batch_size, height, width)
            
            # Test with super method
            super()._test_to_device(film_block, x, condition)
    
    def _test_eval_mode(self, module, *args, **kwargs):
        """Custom implementation for FiLM modules"""
        super()._test_eval_mode(module, *args, **kwargs)

    def _test_train_mode(self, module, *args, **kwargs):
        """Custom implementation for FiLM modules"""
        super()._test_train_mode(module, *args, **kwargs)

    
if __name__ == '__main__':
    unittest.main()
