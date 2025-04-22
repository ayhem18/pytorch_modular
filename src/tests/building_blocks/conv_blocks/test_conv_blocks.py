from typing import List, Tuple
import unittest
import torch
import random
from torch import nn

import mypt.code_utils.pytorch_utils as pu
from mypt.building_blocks.conv_blocks.conv_block import BasicConvBlock
from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser
from tests.building_blocks.custom_base_test import CustomModuleBaseTest


class TestConvBlocks(CustomModuleBaseTest):
    def setUp(self):
        self.dim_analyser = DimensionsAnalyser()

    def _generate_random_conv_block(self, 
                                    use_bn=True, 
                                    activation_after_each_layer=False, 
                                    activation=nn.ReLU) -> Tuple[BasicConvBlock, int, List[int]]:
        num_layers = random.randint(1, 4)
        channels = [random.randint(1, 64) for _ in range(num_layers + 1)]
        kernel_sizes = random.randint(1, 5)
        
        return BasicConvBlock(num_layers, channels, kernel_sizes, use_bn, activation_after_each_layer, activation), num_layers, channels


    def test_conv_block_single_activation_no_bn(self):
        """Test that conv block with single activation at the end and no batch norm works correctly"""
        for _ in range(100):  # Test multiple random configurations
            block, num_layers, channels = self._generate_random_conv_block(use_bn=False, activation_after_each_layer=False, activation=nn.ReLU)
            
            # Check that block was created correctly
            children = list(block.children())
            # We expect num_layers conv layers followed by 1 activation layer
            self.assertEqual(len(children), num_layers + 1)
            
            # Check that all except the last layer are Conv2d
            for i in range(num_layers):
                self.assertIsInstance(children[i], nn.Conv2d)
                self.assertEqual(children[i].in_channels, channels[i])
                self.assertEqual(children[i].out_channels, channels[i+1])
            
            # Check the last layer is an activation
            self.assertIsInstance(children[-1], nn.ReLU)

    def test_conv_block_single_activation_bn(self):
        """Test that conv block with single activation at the end and batch norm works correctly"""
        for _ in range(100):  # Test multiple random configurations            
            block, num_layers, channels = self._generate_random_conv_block(use_bn=True, activation_after_each_layer=False, activation=nn.ReLU)

            # Check that block was created correctly
            children = list(block.children())
            # We expect (conv + bn) * num_layers + 1 activation layer
            self.assertEqual(len(children), num_layers * 2 + 1)
            
            # Check the layers
            for i in range(num_layers):
                # Every even layer should be Conv2d
                self.assertIsInstance(children[i*2], nn.Conv2d)
                self.assertEqual(children[i*2].in_channels, channels[i])
                self.assertEqual(children[i*2].out_channels, channels[i+1])
                
                # Every odd layer should be BatchNorm2d
                self.assertIsInstance(children[i*2+1], nn.BatchNorm2d)
                self.assertEqual(children[i*2+1].num_features, channels[i+1])
            
            # Check the last layer is an activation
            self.assertIsInstance(children[-1], nn.ReLU)

    def test_conv_block_activation_after_each_layer_no_bn(self):
        """Test that conv block with activation after each layer and no batch norm works correctly"""
        for _ in range(100):  # Test multiple random configurations            
            block, num_layers, channels = self._generate_random_conv_block(use_bn=False, activation_after_each_layer=True, activation=nn.ReLU)

            # Check that block was created correctly
            children = list(block.children())
            # We expect (conv + activation) * num_layers layers
            self.assertEqual(len(children), num_layers * 2)
            
            # Check the layers
            for i in range(num_layers):
                # Every even layer should be Conv2d
                self.assertIsInstance(children[i*2], nn.Conv2d)
                self.assertEqual(children[i*2].in_channels, channels[i])
                self.assertEqual(children[i*2].out_channels, channels[i+1])
                
                # Every odd layer should be activation
                self.assertIsInstance(children[i*2+1], nn.ReLU)

    def test_conv_block_activation_after_each_layer_bn(self):
        """Test that conv block with activation after each layer and batch norm works correctly"""
        for _ in range(100):  # Test multiple random configurations            
            block, num_layers, channels = self._generate_random_conv_block(use_bn=True, activation_after_each_layer=True, activation=nn.ReLU)

            # Check that block was created correctly
            children = list(block.children())
            # We expect (conv + bn + activation) * num_layers layers
            self.assertEqual(len(children), num_layers * 3)
            
            # Check the layers
            for i in range(num_layers):
                # Check Conv2d layer
                self.assertIsInstance(children[i*3], nn.Conv2d)
                self.assertEqual(children[i*3].in_channels, channels[i])
                self.assertEqual(children[i*3].out_channels, channels[i+1])
                
                # Check BatchNorm2d layer
                self.assertIsInstance(children[i*3+1], nn.BatchNorm2d)
                self.assertEqual(children[i*3+1].num_features, channels[i+1])
                
                # Check activation layer
                self.assertIsInstance(children[i*3+2], nn.ReLU)


    def test_children_parameters_method_single_activation_layer_no_bn(self):
        """Test that children, parameters, named_children, named_parameters methods work correctly for a single activation layer and no batch norm"""
        for _ in range(100):  # Test multiple random configurations            
            block, num_layers, channels = self._generate_random_conv_block(use_bn=False, activation_after_each_layer=False, activation=nn.ReLU)

            named_children = list(block.named_children())            
            # We expect num_layers conv layers followed by 1 activation layer
            self.assertEqual(len(named_children), num_layers + 1)
            
            # Check that all except the last layer are Conv2d
            for i in range(num_layers):
                self.assertEqual(named_children[i][0], f"conv_{i+1}")
                self.assertIsInstance(named_children[i][1], nn.Conv2d)
            
            # Check the last layer is an activation
            self.assertEqual(named_children[-1][0], f"activation_layer")
            self.assertIsInstance(named_children[-1][1], nn.ReLU)

    def test_children_parameters_method_single_activation_layer_bn(self):
        """Test that children, parameters, named_children, named_parameters methods work correctly for a single activation layer and batch norm"""
        for _ in range(100):  # Test multiple random configurations
            block, num_layers, _ = self._generate_random_conv_block(use_bn=True, activation_after_each_layer=False, activation=nn.ReLU)

            named_children = list(block.named_children())            
            # We expect (conv + bn) * num_layers + 1 activation layer
            self.assertEqual(len(named_children), num_layers * 2 + 1)
            
            # Check the layers
            for i in range(num_layers):
                # Every even layer should be Conv2d
                self.assertEqual(named_children[i*2][0], f"conv_{i+1}")
                self.assertIsInstance(named_children[i*2][1], nn.Conv2d)
                
                # Every odd layer should be BatchNorm2d
                self.assertEqual(named_children[i*2+1][0], f"bn_{i+1}")
                self.assertIsInstance(named_children[i*2+1][1], nn.BatchNorm2d)

            # Check the last layer is an activation
            self.assertEqual(named_children[-1][0], f"activation_layer")
            self.assertIsInstance(named_children[-1][1], nn.ReLU)

    def test_children_parameters_method_activation_after_each_layer_no_bn(self):
        """Test that children, parameters, named_children, named_parameters methods work correctly for an activation after each layer and no batch norm"""
        for _ in range(100):  # Test multiple random configurations
            block, num_layers, _ = self._generate_random_conv_block(use_bn=False, activation_after_each_layer=True, activation=nn.ReLU)
            
            # Check that block was created correctly
            named_children = list(block.named_children())
            # We expect (conv + activation) * num_layers layers
            self.assertEqual(len(named_children), num_layers * 2)
            
            # Check the layers
            for i in range(num_layers):
                # Every even layer should be Conv2d
                self.assertEqual(named_children[i*2][0], f"conv_{i+1}")
                self.assertIsInstance(named_children[i*2][1], nn.Conv2d)
                
                # Every odd layer should be activation
                self.assertEqual(named_children[i*2+1][0], f"activation_{i+1}")
                self.assertIsInstance(named_children[i*2+1][1], nn.ReLU)

    def test_children_parameters_method_activation_after_each_layer_bn(self):
        """Test that children, parameters, named_children, named_parameters methods work correctly for an activation after each layer and batch norm"""
        for _ in range(100):  # Test multiple random configurations
            block, num_layers, channels = self._generate_random_conv_block(use_bn=True, activation_after_each_layer=True, activation=nn.ReLU)
            
            # Check that block was created correctly
            named_children = list(block.named_children())
            # We expect (conv + bn + activation) * num_layers layers
            self.assertEqual(len(named_children), num_layers * 3)
            
            # Check the layers
            for i in range(num_layers):
                # Check Conv2d layer
                self.assertEqual(named_children[i*3][0], f"conv_{i+1}")
                self.assertIsInstance(named_children[i*3][1], nn.Conv2d)
                
                # Check BatchNorm2d layer
                self.assertEqual(named_children[i*3+1][0], f"bn_{i+1}")
                self.assertIsInstance(named_children[i*3+1][1], nn.BatchNorm2d)
                
                # Check activation layer
                self.assertEqual(named_children[i*3+2][0], f"activation_{i+1}")
                self.assertIsInstance(named_children[i*3+2][1], nn.ReLU)


    @unittest.skip("Skipping forward pass shape test as it takes a long time to run")
    def test_forward_pass_shape(self):
        """Test that forward pass produces output with expected shape"""
        for _ in range(1000):  # Test multiple random configurations
            # Generate random but valid parameters
            num_layers = random.randint(1, 4)
            channels = [random.randint(1, 64) for _ in range(num_layers + 1)]
            kernel_sizes = [2 * random.randint(1, 5) + 1 for _ in range(num_layers)] # have odd kernel sizes in general
            batch_size = random.randint(1, 8)
            img_size = random.randint(16, 64)  # Small-ish image for speed
            
            # Create random input tensor
            input_tensor = torch.randn(batch_size, channels[0], img_size, img_size)
            
            # Create block
            block = ConvBlock(
                num_conv_layers=num_layers,
                channels=channels,
                kernel_sizes=kernel_sizes,
                use_bn=True,
                activation_after_each_layer=True,
                activation=nn.ReLU
            )
            
            # Get actual output shape
            output = block(input_tensor)
            actual_shape = tuple(output.shape)
            
            # Get predicted output shape using DimensionsAnalyser
            expected_shape = self.dim_analyser.analyse_dimensions(
                input_shape=tuple(input_tensor.shape),
                net=block
            )
            
            # Compare shapes
            self.assertEqual(actual_shape, expected_shape, 
                            f"Mismatch in output shape: got {actual_shape}, expected {expected_shape}")

    
    def test_eval_train_methods(self):        
        """Test that eval and train methods work correctly"""
        # Create a block with batch normalization
        for _ in range(100):  # Test multiple random configurations
            num_layers = random.randint(2, 4)
            channels = [random.randint(1, 32) for _ in range(num_layers + 1)]
            kernel_sizes = [2 * random.randint(1, 5) + 1 for _ in range(num_layers)]
            
            block = ConvBlock(
                num_conv_layers=num_layers,
                channels=channels,
                kernel_sizes=kernel_sizes,
                use_bn=True,  # Must use batch norm for this test
                activation_after_each_layer=True,
                activation=nn.ReLU
            )
            
            # Test eval mode
            block.eval()
            
            # Verify all children are in eval mode
            for child in block.children():
                self.assertFalse(child.training, "Child module should be in eval mode")
            
            # Create a small batch (size 1)
            single_item_batch = torch.randn(1, channels[0], 24, 24)
            
            # This should not raise an error in eval mode
            try:
                _ = block.forward(single_item_batch)
            except Exception as e:
                self.fail(f"Block in eval mode raised an error with batch size 1: {e}")
            
            # Test train mode
            block.train()
            
            # Verify all children are in train mode
            for child in block.children():
                self.assertTrue(child.training, "Child module should be in train mode")
            
            # Note: The BatchNorm2d doesn't actually throw an error with batch size 1 in train mode,
            # it just computes running stats from a single sample (which is statistically invalid).
            # PyTorch allows this but warns about it in documentation. So we can't test for an error here.
            
            # This is the ideal test, but it doesn't throw an error in PyTorch:
            _ = block.forward(single_item_batch)  # No error should be raised, but results are statistically invalid
            
            # Instead, we'll verify that train() and eval() states are different by checking statistics
            block.train()
            _ = block.forward(single_item_batch)  # Run once to update running stats
            
            # Find BatchNorm2d modules
            bn_modules = [m for m in block.modules() if isinstance(m, nn.BatchNorm2d)]
            self.assertGreater(len(bn_modules), 0, "No BatchNorm2d modules found")
            
            # Store running stats in train mode
            train_running_means = [bn.running_mean.clone() for bn in bn_modules]
            
            # Set to eval mode
            block.eval()
            
            # Run with different data
            different_data = torch.randn(1, channels[0], 24, 24)
            _ = block.forward(different_data)
            
            # In eval mode, running stats should not change
            for i, bn in enumerate(bn_modules):
                self.assertTrue(torch.allclose(train_running_means[i], bn.running_mean), 
                            "BatchNorm running mean should not change in eval mode")

    def _get_valid_input(self, *args, **kwargs) -> torch.Tensor:
        """
        Generate a random input tensor with the correct shape.
        """
        if 'in_features' not in kwargs:
            raise ValueError("in_features must be provided")
        
        in_features = kwargs['in_features']
        return torch.randn(10, in_features, 224, 224)

    # Custom module base tests
    def test_eval_mode(self):
        """Test that eval mode is correctly set across the conv block"""
        for _ in range(1000):
            block, _, _ = self._generate_random_conv_block(use_bn=True, activation_after_each_layer=True, activation=nn.ReLU)
            super()._test_eval_mode(block)

        for _ in range(1000):
            block, _, _ = self._generate_random_conv_block(use_bn=True, activation_after_each_layer=False, activation=nn.ReLU)
            super()._test_eval_mode(block)

        for _ in range(1000):
            block, _, _ = self._generate_random_conv_block(use_bn=False, activation_after_each_layer=True, activation=nn.ReLU)
            super()._test_eval_mode(block)

        for _ in range(1000):
            block, _, _ = self._generate_random_conv_block(use_bn=False, activation_after_each_layer=False, activation=nn.ReLU)
            super()._test_eval_mode(block)
    
    def test_train_mode(self):
        """Test that train mode is correctly set across the conv block"""
        for _ in range(1000):
            block, _, _ = self._generate_random_conv_block(use_bn=True, activation_after_each_layer=True, activation=nn.ReLU)
            super()._test_train_mode(block)

        for _ in range(1000):
            block, _, _ = self._generate_random_conv_block(use_bn=True, activation_after_each_layer=False, activation=nn.ReLU)
            super()._test_train_mode(block)    

        for _ in range(1000):
            block, _, _ = self._generate_random_conv_block(use_bn=False, activation_after_each_layer=True, activation=nn.ReLU)
            super()._test_train_mode(block)

        for _ in range(1000):
            block, _, _ = self._generate_random_conv_block(use_bn=False, activation_after_each_layer=False, activation=nn.ReLU)
            super()._test_train_mode(block)
    
    def test_consistent_output_in_eval_mode(self):
        """Test that the conv block produces consistent output in eval mode"""
        for _ in range(1000):
            block, _, _ = self._generate_random_conv_block(use_bn=True, activation_after_each_layer=True, activation=nn.ReLU)   
            super()._test_consistent_output_in_eval_mode(block)

        for _ in range(1000):
            block, _, _ = self._generate_random_conv_block(use_bn=True, activation_after_each_layer=False, activation=nn.ReLU)   
            super()._test_consistent_output_in_eval_mode(block)
            

        for _ in range(1000):   
            block, _, _ = self._generate_random_conv_block(use_bn=False, activation_after_each_layer=True, activation=nn.ReLU)   
            super()._test_consistent_output_in_eval_mode(block)

        for _ in range(1000):   
            block, _, _ = self._generate_random_conv_block(use_bn=False, activation_after_each_layer=False, activation=nn.ReLU)   
            super()._test_consistent_output_in_eval_mode(block)
        
        for _ in range(5):
            # Create a conv block with batch norm
            num_layers = random.randint(1, 3)
            channels = [random.randint(1, 16) for _ in range(num_layers + 1)]
            kernel_sizes = [2 * random.randint(1, 3) + 1 for _ in range(num_layers)]
            
            block = ConvBlock(
                num_conv_layers=num_layers,
                channels=channels,
                kernel_sizes=kernel_sizes,
                use_bn=True,
                activation_after_each_layer=True,
                activation=nn.ReLU
            )
            
            # Override _get_valid_input_shape for ConvBlock
            def _get_valid_input_shape(block, batch_size=2):
                return (batch_size, block.channels[0], 32, 32)
            
            self._get_valid_input_shape = _get_valid_input_shape
            self._test_consistent_output_in_eval_mode(block)
    
    def test_batch_size_one_in_train_mode(self):
        """Test that the conv block handles batch size 1 in train mode"""
        for _ in range(5):
            # Create a conv block with batch norm
            num_layers = random.randint(1, 3)
            channels = [random.randint(1, 16) for _ in range(num_layers + 1)]
            kernel_sizes = [2 * random.randint(1, 3) + 1 for _ in range(num_layers)]
            
            block = ConvBlock(
                num_conv_layers=num_layers,
                channels=channels,
                kernel_sizes=kernel_sizes,
                use_bn=True,
                activation_after_each_layer=True,
                activation=nn.ReLU
            )
            
            # Override _get_valid_input_shape for ConvBlock
            def _get_valid_input_shape(block, batch_size=1):
                return (batch_size, block.channels[0], 32, 32)
            
            self._get_valid_input_shape = _get_valid_input_shape
            self._test_batch_size_one_in_train_mode(block)
    
    def test_batch_size_one_in_eval_mode(self):
        """Test that the conv block handles batch size 1 in eval mode"""
        for _ in range(5):
            # Create a conv block with batch norm
            num_layers = random.randint(1, 3)
            channels = [random.randint(1, 16) for _ in range(num_layers + 1)]
            kernel_sizes = [2 * random.randint(1, 3) + 1 for _ in range(num_layers)]
            
            block = ConvBlock(
                num_conv_layers=num_layers,
                channels=channels,
                kernel_sizes=kernel_sizes,
                use_bn=True,
                activation_after_each_layer=True,
                activation=nn.ReLU
            )
            
            # Override _get_valid_input_shape for ConvBlock
            def _get_valid_input_shape(block, batch_size=1):
                return (batch_size, block.channels[0], 32, 32)
            
            self._get_valid_input_shape = _get_valid_input_shape
            self._test_batch_size_one_in_eval_mode(block)
    
    def test_named_parameters_length(self):
        """Test that named_parameters and parameters have the same length"""
        for _ in range(5):
            # Create a conv block
            num_layers = random.randint(1, 3)
            channels = [random.randint(1, 16) for _ in range(num_layers + 1)]
            kernel_sizes = [2 * random.randint(1, 3) + 1 for _ in range(num_layers)]
            
            block = ConvBlock(
                num_conv_layers=num_layers,
                channels=channels,
                kernel_sizes=kernel_sizes,
                use_bn=True,
                activation_after_each_layer=True,
                activation=nn.ReLU
            )
            self._test_named_parameters_length(block)



if __name__ == '__main__':
    pu.seed_everything(69)
    unittest.main()