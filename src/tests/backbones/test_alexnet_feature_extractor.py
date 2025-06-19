import torch
import random
import unittest
from torch import nn
from typing import List, Union
from collections import OrderedDict

from mypt.backbones.alexnetFE import AlexNetFE
from torchvision.models import alexnet, AlexNet_Weights

from tests.custom_base_test import CustomModuleBaseTest


@unittest.skip("skipping the alexnet Feature extractor tests: too time consuming")
class TestAlexNetFE(CustomModuleBaseTest):
    """
    Test class for AlexNetFE feature extractor implementation.
    Tests construction with different block specifications and freezing strategies.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Initialize tests for AlexNet architecture.
        This method is run ONCE before any tests are executed.
        Sets up the original AlexNet model for comparison.
        """
        # Load the original AlexNet model for comparison
        cls.original_model = alexnet(weights=AlexNet_Weights.DEFAULT)
        
        # Extract the individual blocks for testing
        features, avgpool, _ = list(cls.original_model.children())
        
        # Store the blocks for later use
        cls.conv_blocks = {}
        cls._extract_conv_blocks(features)
        cls.avgpool = avgpool

    @classmethod
    def _extract_conv_blocks(cls, features: nn.Sequential):
        """
        Extract the convolutional blocks from the features module of AlexNet.
        Each block starts with a Conv2d and ends before the next Conv2d.
        
        Args:
            features: The features module from AlexNet
        """
        block_index = 0
        current_block = []
        
        # Extract blocks following the same logic as in AlexNetFE
        for name, module in features.named_children():
            if isinstance(module, nn.Conv2d):
                if len(current_block) == 0:
                    current_block.append((name, module))
                else:
                    cls.conv_blocks[block_index] = nn.Sequential(OrderedDict(current_block))
                    current_block = [(name, module)]
                    block_index += 1
            else:
                current_block.append((name, module))
        
        # Add the last block
        cls.conv_blocks[block_index] = nn.Sequential(OrderedDict(current_block))

    def _get_expected_blocks(self, model_blocks: Union[str, List[str], int, List[int]]) -> List[int]:
        """
        Helper method to determine which blocks should be in the feature extractor.
        Follows the same logic as the __verify_blocks method in AlexNetFE.
        
        Args:
            model_blocks: Block specification as accepted by AlexNetFE
            
        Returns:
            List of block indices that should be included
        """
        if isinstance(model_blocks, str):
            if model_blocks in AlexNetFE._AlexNetFE__str_arguments:
                if model_blocks == 'conv_block':
                    return list(range(5))
                elif model_blocks == 'conv_block_avgpool':
                    return list(range(6))
                else:  # 'convX'
                    return list(range(int(model_blocks[-1])))
            else:
                raise ValueError(f"Invalid string argument: {model_blocks}")
                
        elif isinstance(model_blocks, int):
            if 0 <= model_blocks <= 5:
                return list(range(model_blocks + 1))
            else:
                raise ValueError(f"Invalid integer argument: {model_blocks}")
                
        elif isinstance(model_blocks, list):
            if len(model_blocks) == 0:
                return []
                
            if isinstance(model_blocks[0], str):
                # Convert strings to indices
                indices = []
                for block in model_blocks:
                    if block == 'avgpool':
                        indices.append(5)
                    else:  # 'convX'
                        indices.append(int(block[-1]) - 1)
                return sorted(indices)
                
            elif isinstance(model_blocks[0], int):
                return sorted(model_blocks)
                
        return []
    
    def _count_frozen_parameters(self, model: nn.Module) -> int:
        """
        Count the number of frozen parameters in a model.
        
        Args:
            model: The model to check
            
        Returns:
            Number of frozen parameters
        """
        frozen_count = 0
        for param in model.parameters():
            if not param.requires_grad:
                frozen_count += 1
        return frozen_count
    
    def _get_frozen_blocks(self, feature_extractor: AlexNetFE) -> List[int]:
        """
        Determine which blocks are frozen in a feature extractor.
        
        Args:
            feature_extractor: The AlexNetFE instance to check
            
        Returns:
            List of block indices that are frozen
        """
        frozen_blocks = []
        for i in range(6):  # Check all possible blocks (0-5)
            block_name = AlexNetFE._AlexNetFE__index2block_name.get(i)
            if not hasattr(feature_extractor._model, block_name):
                continue

            block = getattr(feature_extractor._model, block_name)
            # Check if all parameters in the block are frozen

            # if the block has no parameters, all_frozen should be set to False
            all_frozen = len(list(block.parameters())) > 0 
            for param in block.parameters():
                if param.requires_grad:
                    all_frozen = False
                    break
            
            if all_frozen:
                frozen_blocks.append(i)
        
        return frozen_blocks


    def test_valid_blocks(self):
        """
        Test that AlexNetFE accepts valid block specifications and rejects invalid ones.
        """
        # Test valid string arguments
        valid_str_args = ['conv_block', 'conv_block_avgpool', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5']
        for arg in valid_str_args:
            try:
                feature_extractor = AlexNetFE(model_blocks=arg, frozen_model_blocks=False)
                self.assertIsInstance(feature_extractor, AlexNetFE)
            except Exception as e:
                self.fail(f"AlexNetFE constructor raised an exception with valid string argument {arg}: {e}")
        
        # Test valid integer arguments
        for i in range(6):
            try:
                feature_extractor = AlexNetFE(model_blocks=i, frozen_model_blocks=False)
                self.assertIsInstance(feature_extractor, AlexNetFE)
            except Exception as e:
                self.fail(f"AlexNetFE constructor raised an exception with valid integer argument {i}: {e}")
        
    def test_valid_frozen_blocks(self):
        """
        Test that AlexNetFE accepts valid frozen block specifications and rejects invalid ones.
        """
        # Test with boolean frozen_model_blocks
        for bool_val in [True, False]:

            feature_extractor = AlexNetFE(model_blocks='conv_block_avgpool', frozen_model_blocks=bool_val)
            self.assertEqual(len(self._get_frozen_blocks(feature_extractor)), 5 * int(bool_val))

        # Test with string frozen_model_blocks
        model_blocks = 'conv_block_avgpool'  # All blocks
        valid_frozen = ['conv1', 'conv2', 'conv5', 'avgpool']
        feature_extractor = AlexNetFE(model_blocks=model_blocks, frozen_model_blocks=valid_frozen)
        frozen_indices = self._get_frozen_blocks(feature_extractor)
        expected_indices = [0, 1, 4]  # Indices corresponding to conv1, conv2, conv5, avgpool
        self.assertEqual(sorted(frozen_indices), sorted(expected_indices))
        
        # Test with integer frozen_model_blocks
        model_blocks = 5  # All blocks (0-5)
        for i in range(5):
            feature_extractor = AlexNetFE(model_blocks=model_blocks, frozen_model_blocks=i)
            frozen_indices = self._get_frozen_blocks(feature_extractor)
            expected_indices = list(range(i + 1)) 
            self.assertEqual(sorted(frozen_indices), sorted(expected_indices))
            self.assertEqual(len(frozen_indices), i + 1)
        
        # Test with list of integers frozen_model_blocks
        model_blocks = list(range(6))  # All blocks
        valid_frozen = [0, 2, 4]  # Blocks 0, 2, 4
        feature_extractor = AlexNetFE(model_blocks=model_blocks, frozen_model_blocks=valid_frozen)
        frozen_indices = self._get_frozen_blocks(feature_extractor)
        self.assertEqual(sorted(frozen_indices), sorted(valid_frozen))
        
        # Test with invalid frozen_model_blocks (not a subset of model_blocks)
        model_blocks = [0, 1, 2]  # Only first three blocks
        invalid_frozen = [0, 3]  # Block 3 is not in model_blocks
        with self.assertRaises(ValueError):
            AlexNetFE(model_blocks=model_blocks, frozen_model_blocks=invalid_frozen)
        
        # Test with invalid frozen_model_blocks (wrong type)
        for i in range(4):
            model_blocks = 'conv'+str(i + 1)  # Blocks 0-2
            invalid_frozen = i + 1 # the block i + 1 should not be in the model blocks
        
            with self.assertRaises(ValueError):
                AlexNetFE(model_blocks=model_blocks, frozen_model_blocks=invalid_frozen)
        
            invalid_frozen = [i + 1]
            with self.assertRaises(ValueError):
                AlexNetFE(model_blocks=model_blocks, frozen_model_blocks=invalid_frozen)


    def test_structure_with_string_args(self):
        """
        Test that the feature extractor has the correct structure when built with string arguments.
        Compare it to the original AlexNet model.
        """
        # Test with 'conv_block'
        feature_extractor = AlexNetFE(model_blocks='conv_block', frozen_model_blocks=False)
        self.assertEqual(len(feature_extractor._model), 5)
        
        # Check each block matches the original
        for i in range(5):
            block_name = f'conv{i+1}'
            self.assertTrue(hasattr(feature_extractor._model, block_name))
            
            # Extract corresponding blocks for comparison
            fe_block = getattr(feature_extractor._model, block_name)
            orig_block = self.conv_blocks[i]
            
            # Compare number of layers
            self.assertEqual(len(fe_block), len(orig_block))
            
            # Compare layer types
            for (fe_name, fe_module), (orig_name, orig_module) in zip(
                fe_block.named_children(), orig_block.named_children()):
                self.assertEqual(type(fe_module), type(orig_module))
            
        # Test with 'conv_block_avgpool'
        feature_extractor = AlexNetFE(model_blocks='conv_block_avgpool', frozen_model_blocks=False)
        self.assertEqual(len(feature_extractor._model), 6)
        
        # Check avgpool is included and matches original
        self.assertTrue(hasattr(feature_extractor._model, 'avgpool'))
        fe_avgpool = getattr(feature_extractor._model, 'avgpool')
        orig_avgpool = self.avgpool
        self.assertEqual(type(fe_avgpool), type(orig_avgpool))
        
    def test_structure_with_integer_args(self):
        """
        Test that the feature extractor has the correct structure when built with integer arguments.
        """
        # Test with integer 2 (should include blocks 0-2)
        feature_extractor = AlexNetFE(model_blocks=2, frozen_model_blocks=False)
        self.assertEqual(len(feature_extractor._model), 3)
        
        # Check blocks 0-2 are included
        expected_blocks = ['conv1', 'conv2', 'conv3']
        actual_blocks = [name for name, _ in list(feature_extractor.named_children())]
        self.assertEqual(actual_blocks, expected_blocks)
        
        # Test with integer 5 (should include all blocks)
        feature_extractor = AlexNetFE(model_blocks=5, frozen_model_blocks=False)
        self.assertEqual(len(feature_extractor._model), 6)
        
        # Check all blocks are included
        expected_blocks = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'avgpool']
        actual_blocks = [name for name, _ in list(feature_extractor.named_children())]
        self.assertEqual(actual_blocks, expected_blocks)
        
        # Test with list of non consecutive elements
        with self.assertRaises(ValueError):
            AlexNetFE(model_blocks=[0, 2, 4], frozen_model_blocks=False)
            
        # Check only blocks 0, 2, 4 are included
        with self.assertRaises(ValueError):
            AlexNetFE(model_blocks=['conv1', 'conv3', 'conv5'], frozen_model_blocks=False)


        # make sure that `avgpool` is a consecutive argument to any `conv(i)` argument
        for i in range(1, 5):
            convs = [f'conv{j}' for j in range(1, i + 1)]
            try:
                AlexNetFE(model_blocks=convs + ['avgpool'], frozen_model_blocks=False)
            except Exception as e:
                self.fail(f"AlexNetFE constructor raised an exception with valid list of non consecutive elements: {convs + ['avgpool']}: {e}")



    def test_freeze_all_blocks(self):
        """
        Test freezing a feature extractor built with all blocks.
        """
        # Test with frozen_model_blocks=True (all blocks frozen)
        feature_extractor = AlexNetFE(model_blocks='conv_block_avgpool', frozen_model_blocks=True)
        
        # Check all parameters are frozen
        for param in feature_extractor.parameters():
            self.assertFalse(param.requires_grad)
        
        # Test with frozen_model_blocks=False (no blocks frozen)
        feature_extractor = AlexNetFE(model_blocks='conv_block_avgpool', frozen_model_blocks=False)
        
        # Check all parameters are trainable
        for param in feature_extractor.parameters():
            self.assertTrue(param.requires_grad)
        
        # Test with frozen_model_blocks='conv(i + 1)': first i + 1 blocks are frozen 
        for i in range(5):
            feature_extractor = AlexNetFE(model_blocks='conv_block_avgpool', frozen_model_blocks='conv'+str(i + 1))
            
            # Check blocks 0-2 are frozen
            frozen_blocks = self._get_frozen_blocks(feature_extractor)
            expected_frozen = list(range(i + 1))  # conv1, conv2, conv3
            self.assertEqual(sorted(frozen_blocks), sorted(expected_frozen))
            
            # Check blocks 3-5 are trainable
            for block_idx in range(i + 1, 6):
                block_name = AlexNetFE._AlexNetFE__index2block_name.get(block_idx)
                for param in getattr(feature_extractor._model, block_name).parameters():
                    self.assertTrue(param.requires_grad)

        for i in range(5):
            feature_extractor = AlexNetFE(model_blocks='conv_block_avgpool', frozen_model_blocks=[i])
            frozen_blocks = self._get_frozen_blocks(feature_extractor)
            expected_frozen = [i]
            self.assertEqual(sorted(frozen_blocks), sorted(expected_frozen))
            
            # all blocks are trainable except the i-th one
            for block_idx in range(6):
                
                if block_idx == i:
                    continue
                
                block_name = AlexNetFE._AlexNetFE__index2block_name.get(block_idx)
                for param in getattr(feature_extractor._model, block_name).parameters():
                    self.assertTrue(param.requires_grad)


    def test_freeze_conv_blocks(self):
        """
        Test freezing a feature extractor built with all convolutional layers (without average pooling).
        """
        # Test with frozen_model_blocks=True (all blocks frozen)
        feature_extractor = AlexNetFE(model_blocks='conv_block', frozen_model_blocks=True)
        
        # Check all parameters are frozen
        for param in feature_extractor.parameters():
            self.assertFalse(param.requires_grad)
        
        # Test with frozen_model_blocks=[0, 2, 4] (specific blocks frozen)
        # get a random subset of the [0, 1, 2, 3, 4]
        for _ in range(10):
            sample_size = random.randint(1, 5)
            frozen_blocks = random.sample(range(5), sample_size)
            feature_extractor = AlexNetFE(model_blocks='conv_block', frozen_model_blocks=frozen_blocks)
            
            frozen_blocks = self._get_frozen_blocks(feature_extractor)
            expected_frozen = frozen_blocks
            self.assertEqual(sorted(frozen_blocks), sorted(expected_frozen))
            
            for block_idx in range(5):                    
                block_name = AlexNetFE._AlexNetFE__index2block_name.get(block_idx)
                for param in getattr(feature_extractor._model, block_name).parameters():
                    self.assertEqual(param.requires_grad, block_idx not in frozen_blocks)


    def test_freeze_with_indices(self):
        """
        Test freezing a feature extractor built by passing indices to the constructor.
        The freezing should be valid for any n less or equal to the "n" used to build the feature extractor.
        """
        # Build with blocks 0-3
        for i in range(1, 5):
            j = random.randint(0, i - 1)
            feature_extractor = AlexNetFE(model_blocks=i, frozen_model_blocks=j)
            
            # Check blocks 0-j are frozen
            frozen_blocks = self._get_frozen_blocks(feature_extractor)
            expected_frozen = list(range(j + 1))  # First j + 1 blocks
            self.assertEqual(sorted(frozen_blocks), sorted(expected_frozen))
            
            
            for k in range(i + 1):                
                block_name = AlexNetFE._AlexNetFE__index2block_name.get(k)
                for param in getattr(feature_extractor._model, block_name).parameters():
                    self.assertEqual(param.requires_grad, k > j) # blocks k > j are trainable so param.requires_grad should be True

        

    ########################################################## CustomModuleBaseTest tests ##########################################################

    def test_eval_mode(self):
        """Test that eval mode is correctly set across the feature extractor"""
        for i in range(1, 6):
            feature_extractor = AlexNetFE(model_blocks=i, frozen_model_blocks=False)
            super()._test_eval_mode(feature_extractor)
    
    def test_train_mode(self):
        """Test that train mode is correctly set across the feature extractor"""
        for i in range(1, 6):
            feature_extractor = AlexNetFE(model_blocks=i, frozen_model_blocks=False)
            super()._test_train_mode(feature_extractor)
    
    def test_consistent_output_in_eval_mode(self):  
        """Test that the feature extractor produces consistent output in eval mode"""
        for i in range(1, 6):
            feature_extractor = AlexNetFE(model_blocks=i, frozen_model_blocks=False)
            input_tensor = torch.randn(random.randint(1, 10), 3, 224, 224)
            super()._test_consistent_output_in_eval_mode(feature_extractor, input_tensor)
    
    def test_batch_size_one_in_eval_mode(self):
        """Test that the feature extractor handles batch size 1 in eval mode"""
        for i in range(1, 6):
            feature_extractor = AlexNetFE(model_blocks=i, frozen_model_blocks=False)
            input_tensor = torch.randn(1, 3, 224, 224)
            super()._test_batch_size_one_in_eval_mode(feature_extractor, input_tensor)
    
    def test_batch_size_one_in_train_mode(self):
        """Test that the feature extractor handles batch size 1 in train mode"""
        for i in range(1, 6):
            feature_extractor = AlexNetFE(model_blocks=i, frozen_model_blocks=False)
            input_tensor = torch.randn(1, 3, 224, 224)
            super()._test_batch_size_one_in_train_mode(feature_extractor, input_tensor)

    def test_named_parameters_length(self): 
        """Test that named_parameters and parameters have the same length"""
        for i in range(1, 6):
            feature_extractor = AlexNetFE(model_blocks=i, frozen_model_blocks=False)
            super()._test_named_parameters_length(feature_extractor)


    def test_consistent_output_without_dropout_bn(self):
        """Test that modules without dropout or batch normalization produce consistent output for the same input"""
        for i in range(1, 6):
            feature_extractor = AlexNetFE(model_blocks=i, frozen_model_blocks=False)
            input_tensor = torch.randn(random.randint(1, 10), 3, 224, 224)
            super()._test_consistent_output_without_dropout_bn(feature_extractor, input_tensor)

    def test_to_device(self):
        """Test that the feature extractor can be moved to different devices"""
        for i in range(1, 6):
            feature_extractor = AlexNetFE(model_blocks=i, frozen_model_blocks=False)
            super()._test_to_device(feature_extractor, torch.randn(random.randint(1, 10), 3, 224, 224))


if __name__ == '__main__':
    unittest.main()
