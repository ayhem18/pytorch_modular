import unittest
import torch
import random
import mypt.code_utils.pytorch_utils as pu

from torch import nn
from typing import Dict
from mypt.backbones.resnetFE import ResnetFE
from torchvision.models.resnet import Bottleneck

from tests.custom_base_test import CustomModuleBaseTest


@unittest.skip("the test is too time consuming")
class TestResnetFE(CustomModuleBaseTest):
    """
    Test class for ResnetFE feature extractor implementation.
    Tests construction with different architectures and extraction strategies.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Initialize tests for all ResNet architectures.
        This method is run ONCE before any tests are executed.
        Calculates the number of layer blocks and residual blocks in each architecture.
        """
        cls.architectures = ResnetFE.__archs__
        
        # Create dictionaries to store layer and bottleneck counts for each architecture
        cls.layer_counts = {}
        cls.bottleneck_counts = {}
        
        # Compute counts for each architecture
        for arch in cls.architectures:
            cls.layer_counts[arch] = cls._compute_layers(arch)
            cls.bottleneck_counts[arch] = cls._compute_bottlenecks(arch)
        
        # Print summary for debugging
        # print(f"Layer counts by architecture: {cls.layer_counts}")
        # print(f"Bottleneck counts by architecture: {cls.bottleneck_counts}")


    @classmethod
    def _compute_layers(cls, architecture: int) -> int:
        """
        Computes the number of layer blocks in a given ResNet architecture.
        
        Args:
            architecture: ResNet architecture (18, 34, 50, 101, or 152)
            
        Returns:
            Number of layer blocks in the architecture
        """
        # Load the model
        model_constructor, weights = ResnetFE.get_model(architecture)
        model = model_constructor(weights=weights.DEFAULT)
        
        # Count layer blocks
        layer_count = 0
        for name, _ in model.named_children():
            if ResnetFE.LAYER_BLOCK.lower() in name.lower():
                layer_count += 1
        
        return layer_count
    
    @classmethod
    def _compute_bottlenecks(cls, architecture: int) -> Dict[str, int]:
        """
        Computes the number of residual blocks in each layer of a given ResNet architecture.
        
        Args:
            architecture: ResNet architecture (18, 34, 50, 101, or 152)
            
        Returns:
            Dictionary mapping layer names to number of residual blocks
        """
        # Load the model
        model_constructor, weights = ResnetFE.get_model(architecture)
        model = model_constructor(weights=weights.DEFAULT)
        
        # Count residual blocks in each layer
        bottleneck_counts = {}
        
        for name, module in model.named_children():
            if ResnetFE.LAYER_BLOCK.lower() in name.lower():
                residual_count = 0
                for _, child in module.named_children():
                    if isinstance(child, Bottleneck):
                            residual_count += 1
                
                bottleneck_counts[name] = residual_count
        
        return bottleneck_counts

    def test_assumptions(self):
        """
        Verifies that all supported ResNet architectures follow the expected structure:
        1. A convolutional layer
        2. A batch normalization layer 
        3. A ReLU activation
        4. A maxpool layer
        5. A variable number of "Layer" blocks (each containing Bottleneck blocks)
        6. An adaptive pooling layer
        7. A linear layer
        
        These structural assumptions are crucial for the correctness of the ResnetFE class.
        """
        for arch in self.architectures:
            # Get the model
            model_constructor, weights = ResnetFE.get_model(architecture=arch)
            model = model_constructor(weights=weights.DEFAULT)
            
            # Check the overall structure by examining the named children
            children = list(model.named_children())
            
            # 1 - 4 verify the initial layers
            self.assertIsInstance(children[0][1], nn.Conv2d)
            self.assertIsInstance(children[1][1], nn.BatchNorm2d)
            self.assertIsInstance(children[2][1], nn.ReLU)
            self.assertIsInstance(children[3][1], nn.MaxPool2d)
            
            # 6 - 7 verify the final layers
            self.assertIsInstance(children[-2][1], nn.AdaptiveAvgPool2d)
            self.assertIsInstance(children[-1][1], nn.Linear)

            blocks = children[4:-2]

            for i, block in enumerate(blocks, start=1):
                self.assertIsInstance(block[1], nn.Sequential)
                self.assertEqual(block[0], f'{ResnetFE.LAYER_BLOCK}{i}')
                # make sure each child of the block is a bottleneck layer
                for child in block[1].children():
                    self.assertIsInstance(child, Bottleneck) 
        
    def _count_layers_in_feature_extractor(self, resnetFE: ResnetFE) -> int:
        """
        Counts the number of layer blocks in a feature extractor.
        
        Args:
            feature_extractor: The feature extractor Sequential module
            
        Returns:
            Number of layer blocks
        """
        layer_count = 0
        for name, _ in resnetFE.named_children():
            if ResnetFE.LAYER_BLOCK.lower() in name.lower():
                layer_count += 1
        
        return layer_count
    
    def _count_bottlenecks_in_feature_extractor(self, resnetFE: ResnetFE) -> int:
        """
        Counts the number of residual blocks in a feature extractor.
        
        Args:
            feature_extractor: The feature extractor Sequential module
            
        Returns:
            Number of residual blocks
        """
        bottleneck_count = 0
        for _, module in resnetFE.named_children():
            if isinstance(module, Bottleneck):
                bottleneck_count += 1
        
        return bottleneck_count

    ################## TESTS FOR THE LAYER BUILD    
    def _test_built_by_layer(self, feature_extractor: ResnetFE, expected_num_layers: int, has_global_avg: bool):
        """
        Helper method to test the structure of a ResnetFE built by layers.
        
        Args:
            feature_extractor: The ResnetFE instance to test
            expected_num_layers: The expected number of layer blocks
            has_global_avg: Whether the feature extractor has a global average pooling layer
        """
        # Get the original ResNet model for comparison
        arch = feature_extractor._architecture
        model_constructor, weights = ResnetFE.get_model(architecture=arch)
        original_model = model_constructor(weights=weights.DEFAULT)
        
        # Get all children from both models
        fe_children = list(feature_extractor.named_children())
        original_children = list(original_model.named_children())
        
        # Check the overall number of components
        expected_components = 4  # Conv, BN, ReLU, MaxPool
        expected_components += expected_num_layers  # Layer blocks
        if has_global_avg:
            expected_components += 1  # AvgPool
            
        self.assertEqual(len(fe_children), expected_components,
                        f"Expected {expected_components} components, got {len(fe_children)}")
        
        # Check initial layers (first 4 components)
        for i in range(4):
            fe_name, fe_module = fe_children[i]
            orig_name, orig_module = original_children[i]
            
            # Check names match
            self.assertEqual(fe_name, orig_name, f"Layer name mismatch at position {i}")
            
            # Check types match
            self.assertEqual(type(fe_module), type(orig_module), 
                           f"Layer type mismatch at position {i}: {type(fe_module)} vs {type(orig_module)}")
            
            # Compare string representations
            self.assertEqual(str(fe_module), str(orig_module),
                            f"Module structure mismatch at position {i}")
        
        # Check layer blocks
        for i in range(expected_num_layers):
            layer_idx = 4 + i
            orig_layer_idx = 4 + i
            
            fe_layer_name, fe_layer = fe_children[layer_idx]
            orig_layer_name, orig_layer = original_children[orig_layer_idx]
            
            # Check layer names match
            self.assertEqual(fe_layer_name, orig_layer_name, 
                           f"Layer name mismatch: {fe_layer_name} vs {orig_layer_name}")
            
            # Check that it's a layer block by name
            self.assertTrue(ResnetFE.LAYER_BLOCK.lower() in fe_layer_name.lower(),
                           f"Component at index {layer_idx} should be a layer block, got {fe_layer_name}")
            
            # Ensure the layer is a Sequential module
            self.assertIsInstance(fe_layer, nn.Sequential,
                                 f"Layer block {fe_layer_name} should be a Sequential module")
            
            # Check each bottleneck block within the layer
            fe_blocks = list(fe_layer.children())
            orig_blocks = list(orig_layer.children())
            
            # First, verify the number of blocks is the same
            self.assertEqual(len(fe_blocks), len(orig_blocks), 
                            f"Number of blocks in {fe_layer_name} differs: {len(fe_blocks)} vs {len(orig_blocks)}")
            
            # Check each bottleneck block
            for j, (fe_block, orig_block) in enumerate(zip(fe_blocks, orig_blocks)):
                # Check that both blocks are of the same type (Bottleneck)
                self.assertEqual(type(fe_block), type(orig_block),
                               f"Block type mismatch in {fe_layer_name} at position {j}")
                
                # Compare string representations of the blocks
                self.assertEqual(str(fe_block), str(orig_block),
                                f"Block structure mismatch in {fe_layer_name} at position {j}")
        
        # Check adaptive pooling layer if expected
        if has_global_avg:
            avg_pool_idx = 4 + expected_num_layers
            orig_avg_pool_idx = original_children.index(next(filter(lambda x: isinstance(x[1], nn.AdaptiveAvgPool2d), original_children)))
            
            fe_avg_name, fe_avg = fe_children[avg_pool_idx]
            orig_avg_name, orig_avg = original_children[orig_avg_pool_idx]
            
            # Check avgpool is the correct type
            self.assertIsInstance(fe_avg, nn.AdaptiveAvgPool2d,
                                 "Last layer should be AdaptiveAvgPool2d when add_global_average=True")
            
            # Compare string representation
            self.assertEqual(str(fe_avg), str(orig_avg),
                            "AdaptiveAvgPool2d structure mismatch")

    def test_build_by_layer_1_total_layers(self):
        for arch in self.architectures:
            total_layers = self.layer_counts[arch]
            
            # Test with different numbers of layers to extract
            for add_global_avg in [True, False]:
                for i in range(1, total_layers + 1):
                    # Create a feature extractor
                    feature_extractor = ResnetFE(
                        build_by_layer=True,
                        num_extracted_layers=i,
                        num_extracted_bottlenecks=i,  # This should be ignored when building by layer
                        freeze=False,
                        freeze_by_layer=True,
                        add_global_average=add_global_avg,
                        architecture=arch
                    )
                    
                    # Test the structure using the helper method
                    self._test_built_by_layer(feature_extractor, i, add_global_avg)
                    
                    # Count actual layers to ensure backward compatibility with existing tests
                    actual_layers = self._count_layers_in_feature_extractor(feature_extractor)
                    
                    self.assertEqual(
                        actual_layers, 
                        i, 
                        f"Architecture {arch}, num_extracted_layers={i}: Expected {i} layers, got {actual_layers}"
                    )

    def test_build_by_layer_negative_values(self):
        for arch in self.architectures:
            total_layers = self.layer_counts[arch]

            for global_avg in [True, False]:    
                # test random negative values 
                num_layers = -1

                # Create a feature extractor
                feature_extractor = ResnetFE(
                    build_by_layer=True,
                    num_extracted_layers=num_layers,
                    num_extracted_bottlenecks=total_layers,  # This should be ignored when building by layer
                    freeze=False,
                    freeze_by_layer=True,
                    add_global_average=global_avg,
                    architecture=arch
                )
                
                # Test the structure using the helper method
                self._test_built_by_layer(feature_extractor, total_layers, global_avg)
                
                # Count actual layers to ensure backward compatibility with existing tests
                actual_layers = self._count_layers_in_feature_extractor(feature_extractor)
                
                self.assertEqual(
                    actual_layers, 
                    total_layers, 
                    f"Architecture {arch}, num_extracted_layers={actual_layers}: Expected {total_layers} layers, got {actual_layers}"
                )
                
    # @unittest.skip("passed")
    def test_build_by_layer_beyond_total_layers(self):
        for arch in self.architectures:
            total_layers = self.layer_counts[arch]

            for global_avg in [True, False]:
                for _ in range(10):
                    # test random values beyond the total number of layers
                    i = random.randint(total_layers + 1, 1000)

                    # Create a feature extractor
                    feature_extractor = ResnetFE(
                        build_by_layer=True,
                        num_extracted_layers=i,
                        num_extracted_bottlenecks=total_layers,  # This should be ignored when building by layer
                        freeze=False,
                        freeze_by_layer=True,   
                        add_global_average=global_avg,
                        architecture=arch
                    )
                    
                    # Test the structure using the helper method
                    self._test_built_by_layer(feature_extractor, total_layers, global_avg)
                    
                    # Count actual layers to ensure backward compatibility with existing tests
                    actual_layers = self._count_layers_in_feature_extractor(feature_extractor)  
                    
                    self.assertEqual(
                        actual_layers, 
                        total_layers, 
                        f"Architecture {arch}, num_extracted_layers={i}: Expected {total_layers} layers, got {actual_layers}"
                    )

    ################## TESTS FOR THE BOTTLENECK BUILD   
    
    def _test_built_by_bottleneck(self, feature_extractor: ResnetFE, expected_num_bottlenecks: int, has_global_avg: bool):
        """
        Helper method to test the structure of a ResnetFE built by bottlenecks.
        
        Args:
            feature_extractor: The ResnetFE instance to test
            expected_num_bottlenecks: The expected number of bottleneck blocks
            has_global_avg: Whether the feature extractor has a global average pooling layer
        """
        # Get the original ResNet model for comparison
        arch = feature_extractor._architecture
        model_constructor, weights = ResnetFE.get_model(architecture=arch)
        original_model = model_constructor(weights=weights.DEFAULT)
        
        # Get all children from both models
        fe_children = list(feature_extractor.named_children())
        original_children = list(original_model.named_children())
        
        # Check the overall number of components
        expected_components = 4  # Conv, BN, ReLU, MaxPool
        expected_components += expected_num_bottlenecks  # Bottleneck blocks
        expected_components += int(has_global_avg)  # AvgPool
            
        self.assertEqual(len(fe_children), expected_components,
                        f"Expected {expected_components} components, got {len(fe_children)}")
        
        # Check initial layers (first 4 components)
        for i in range(4):
            fe_name, fe_module = fe_children[i]
            orig_name, orig_module = original_children[i]
            
            # Check names match
            self.assertEqual(fe_name, orig_name, f"Layer name mismatch at position {i}")
            
            # Check types match
            self.assertEqual(type(fe_module), type(orig_module), 
                           f"Layer type mismatch at position {i}: {type(fe_module)} vs {type(orig_module)}")
            
            # Compare string representations
            self.assertEqual(str(fe_module), str(orig_module),
                            f"Module structure mismatch at position {i}")
        
        # Build mapping of original bottlenecks for comparison
        original_bottlenecks = []
        for name, module in original_model.named_children():
            if ResnetFE.LAYER_BLOCK.lower() in name.lower():
                for _, child in module.named_children():
                    if isinstance(child, Bottleneck):
                        original_bottlenecks.append(child)
        
        # Check bottleneck blocks
        for i in range(expected_num_bottlenecks):
            block_idx = 4 + i
            
            if block_idx >= len(fe_children):
                break
                
            _, fe_block = fe_children[block_idx]
            
            # Ensure the block is a Bottleneck
            self.assertIsInstance(fe_block, Bottleneck,
                                 f"Component at index {block_idx} should be a Bottleneck block")
            
            # Check the bottleneck matches one from the original model
            if i < len(original_bottlenecks):
                orig_block = original_bottlenecks[i]
                
                # Compare string representations of the blocks
                self.assertEqual(str(fe_block), str(orig_block),
                                f"Block structure mismatch at position {i}")
        
        # Check adaptive pooling layer if expected
        if has_global_avg:
            avg_pool_idx = 4 + expected_num_bottlenecks
            
            if avg_pool_idx < len(fe_children):
                _, fe_avg = fe_children[avg_pool_idx]
                
                # Find original avgpool
                orig_avg = None
                for _, module in original_model.named_children():
                    if isinstance(module, nn.AdaptiveAvgPool2d):
                        orig_avg = module
                        break
                
                # Check avgpool is the correct type
                self.assertIsInstance(fe_avg, nn.AdaptiveAvgPool2d,
                                     "Last layer should be AdaptiveAvgPool2d when add_global_average=True")
                
                # Compare string representation
                self.assertEqual(str(fe_avg), str(orig_avg),
                                "AdaptiveAvgPool2d structure mismatch")
    
    # @unittest.skip("passed")
    def test_build_by_bottleneck_1_total_bottlenecks(self):
        for arch in self.architectures:
            # Get total bottlenecks by summing across all layers
            total_bottlenecks = sum(self.bottleneck_counts[arch].values())
            
            # Test with explicit values from 1 to total bottlenecks
            for add_global_avg in [True, False]:
                for i in range(1, total_bottlenecks + 1):
                    # Create a feature extractor
                    feature_extractor = ResnetFE(
                        build_by_layer=False,
                        num_extracted_layers=1,  # this argument should be ignored when building by bottleneck
                        num_extracted_bottlenecks=i,
                        freeze=False, 
                        freeze_by_layer=False,
                        add_global_average=add_global_avg,
                        architecture=arch
                    )
                    
                    # Test the structure using the helper method
                    self._test_built_by_bottleneck(feature_extractor, i, add_global_avg)

    # @unittest.skip("passed")
    def test_build_by_bottleneck_negative_values(self):
        for arch in self.architectures:
            total_bottlenecks = sum(self.bottleneck_counts[arch].values())

            # Test with random negative values (should extract all bottlenecks)
            for add_global_avg in [True, False]:
                num_bottlenecks =  -1
                
                feature_extractor = ResnetFE(
                    build_by_layer=False,
                    num_extracted_layers=1,  # This should be ignored when building by bottleneck
                    num_extracted_bottlenecks=num_bottlenecks,
                    freeze=False,
                    freeze_by_layer=False,
                    add_global_average=add_global_avg,
                    architecture=arch
                )
                
                # Test the structure using the helper method
                self._test_built_by_bottleneck(feature_extractor, total_bottlenecks, add_global_avg)
                    
    # @unittest.skip("passed")
    def test_build_by_bottleneck_beyond_total_bottlenecks(self):
        for arch in self.architectures:
            total_bottlenecks = sum(self.bottleneck_counts[arch].values())

            # Test with random values beyond the total (should extract all bottlenecks)
            for add_global_avg in [True, False]:
                for _ in range(20):
                    num_bottlenecks = random.randint(total_bottlenecks + 1, 1000)
                    
                    feature_extractor = ResnetFE(
                        build_by_layer=False,
                        num_extracted_layers=1,  # This should be ignored when building by bottleneck
                        num_extracted_bottlenecks=num_bottlenecks,
                        freeze=False,
                        freeze_by_layer=False,
                        add_global_average=add_global_avg,
                        architecture=arch
                    )
                    
                    # Test the structure using the helper method
                    self._test_built_by_bottleneck(feature_extractor, total_bottlenecks, add_global_avg)
                    
    # @unittest.skip("passed")
    def test_input_validation(self):
        """
        Tests that the ResnetFE class properly validates input parameters
        and raises appropriate errors for invalid inputs.
        """
        # Test case 1: Passing 0 for num_extracted_layers when build_by_layer=True should raise ValueError
        with self.assertRaises(ValueError) as context:
            ResnetFE(
                build_by_layer=True,
                num_extracted_layers=0,
                num_extracted_bottlenecks=1,
                freeze=False,
                freeze_by_layer=True,
                add_global_average=True,
                architecture=50
            )
        self.assertTrue("number of extracted layers cannot be zero" in str(context.exception))
        
        # Test case 2: Passing 0 for num_extracted_bottlenecks when build_by_layer=False should raise ValueError
        with self.assertRaises(ValueError) as context:
            ResnetFE(
                build_by_layer=False,
                num_extracted_layers=1,
                num_extracted_bottlenecks=0,
                freeze=False,
                freeze_by_layer=False,
                add_global_average=True,
                architecture=50
            )
        self.assertTrue("number of extracted blocks cannot be zero" in str(context.exception))
        
        # Test case 3: Passing 0 for num_extracted_layers when build_by_layer=False should NOT raise error
        try:
            ResnetFE(
                build_by_layer=False,
                num_extracted_layers=0,
                num_extracted_bottlenecks=1,
                freeze=False,
                freeze_by_layer=False,
                add_global_average=True,
                architecture=50
            )
        except ValueError as e:
            self.fail(f"ResnetFE raised ValueError unexpectedly when build_by_layer=False and num_extracted_layers=0: {e}")
        
        # Test case 4: Passing 0 for num_extracted_bottlenecks when build_by_layer=True should NOT raise error
        try:
            ResnetFE(
                build_by_layer=True,
                num_extracted_layers=1,
                num_extracted_bottlenecks=0,
                freeze=False,
                freeze_by_layer=True,
                add_global_average=True,
                architecture=50
            )
        except ValueError as e:
            self.fail(f"ResnetFE raised ValueError unexpectedly when build_by_layer=True and num_extracted_bottlenecks=0: {e}")

        # passing any negative value other than -1 should raise a ValueError
        for _ in range(1, 10):
            random_neg_value = random.randint(-100, -2)
            with self.assertRaises(ValueError) as context:
                ResnetFE(
                    build_by_layer=True,
                    num_extracted_layers=random_neg_value,
                    num_extracted_bottlenecks=2,
                    freeze=False,
                    freeze_by_layer=True,
                    add_global_average=True,
                    architecture=50
                )
            self.assertTrue("only negative value allowed for `num_extracted_layers` is -1" in str(context.exception))

        # passing any negative value other than -1 should raise a ValueError
        for _ in range(1, 10):
            random_neg_value = random.randint(-100, -2)
            with self.assertRaises(ValueError) as context:
                ResnetFE(
                    build_by_layer=False,
                    num_extracted_layers=2,
                    num_extracted_bottlenecks=random_neg_value,
                    freeze=False,
                    freeze_by_layer=False,
                    add_global_average=True,
                    architecture=50
                )
            self.assertTrue("only negative value allowed for `num_extracted_bottlenecks` is -1" in str(context.exception))
        

        try:
            ResnetFE(
                build_by_layer=True,
                num_extracted_layers=-1,
                num_extracted_bottlenecks=2,
                freeze=False,
                freeze_by_layer=True,
                add_global_average=True,
                architecture=50
            )
        except ValueError as e:
            self.fail(f"ResnetFE raised ValueError unexpectedly when build_by_layer=True and num_extracted_bottlenecks=-1: {e}")

        try:
            ResnetFE(
                build_by_layer=False,
                num_extracted_layers=2,
                num_extracted_bottlenecks=-1,
                freeze=False,
                freeze_by_layer=False,
                add_global_average=True,
                architecture=50
            )
        except ValueError as e:
            self.fail(f"ResnetFE raised ValueError unexpectedly when build_by_layer=False and num_extracted_bottlenecks=-1: {e}")

    # @unittest.skip("passed")
    def test_forward_pass_layer(self):
        for arch in self.architectures:
            for t in [True, False]:
                for i in range(1, 5): # all resnet architectures have 4 `layer` blocks
                    feature_extractor = ResnetFE(
                        build_by_layer=True,
                        num_extracted_layers=i,
                        num_extracted_bottlenecks=i, # this value doesn't matter when build_by_layer=True
                        architecture=arch,
                        freeze=False,
                        freeze_by_layer=False,
                        add_global_average=t
                    )
                    
                    # get the original network
                    constructor, weights = ResnetFE.get_model(architecture=arch)
                    net = constructor(weights=weights.DEFAULT) 
                    
                    # put both the feature extractor and the original network in evaluation mode    
                    feature_extractor.eval() 
                    net.eval()

                    # create a random input tensor
                    x = torch.randn(1, 3, 224, 224, requires_grad=False)

                    output_fe = feature_extractor.forward(x)

                    # the test is basically re-implementing the forward pass of the original network
                    output_net = net.conv1(x)
                    output_net = net.bn1(output_net)
                    output_net = net.relu(output_net)
                    output_net = net.maxpool(output_net)

                    output_net = net.layer1(output_net) 

                    if i >= 2:
                        output_net = net.layer2(output_net) 

                    if i >= 3:
                        output_net = net.layer3(output_net)

                    if i >= 4:
                        output_net = net.layer4(output_net)

                    if t:
                        output_net = net.avgpool(output_net)
                    
                    self.assertTrue(torch.allclose(output_fe, output_net), "The feature extractor construction does not seem to be correct")


    ################  test for freezing the feature extractor
    
    def _count_frozen_layer_blocks(self, feature_extractor: ResnetFE) -> int:
        """
        Counts the number of frozen layer blocks in a feature extractor.
        
        Args:
            feature_extractor: The feature extractor to check
            
        Returns:
            Number of frozen layer blocks
        """
        frozen_layer_count = 0
        
        for name, module in feature_extractor.named_children():
            if ResnetFE.LAYER_BLOCK.lower() in name.lower():
                # Check if all parameters in this layer block are frozen
                all_frozen = True
                for param in module.parameters():
                    if param.requires_grad:
                        all_frozen = False
                        break
                
                if all_frozen:
                    frozen_layer_count += 1
                    
        return frozen_layer_count
    
    def _count_frozen_bottlenecks(self, feature_extractor: ResnetFE) -> int:
        """
        Counts the number of frozen bottleneck blocks in a feature extractor.
        
        Args:
            feature_extractor: The feature extractor to check
            
        Returns:
            Number of frozen bottleneck blocks
        """
        frozen_bottleneck_count = 0
        
        for _, module in feature_extractor.named_modules():
            if isinstance(module, Bottleneck):
                # Check if all parameters in this bottleneck are frozen
                all_frozen = True
                for param in module.parameters():
                    if param.requires_grad:
                        all_frozen = False
                        break
                
                if all_frozen:
                    frozen_bottleneck_count += 1
                    
        return frozen_bottleneck_count

    def _count_frozen_bottlenecks_by_layer(self, feature_extractor: ResnetFE) -> int:
        """
        Counts the number of frozen bottleneck blocks in a feature extractor.
        
        Args:
            feature_extractor: The feature extractor to check
            
        Returns:
            Number of frozen bottleneck blocks
        """
        frozen_bottleneck_count = 0
        
        for name, module in feature_extractor.named_modules():
            if not ResnetFE.LAYER_BLOCK.lower() in name.lower():
                continue
                
            for bl in module.children():
                if not isinstance(bl, Bottleneck):
                    continue

                # Check if all parameters in this bottleneck are frozen
                all_frozen = True

                for param in bl.parameters():
                    if param.requires_grad:
                        all_frozen = False
                        break
            
                if all_frozen:
                    frozen_bottleneck_count += 1
                
        return frozen_bottleneck_count
        
    def _are_all_parameters_frozen(self, feature_extractor: ResnetFE) -> bool:
        """
        Checks if all parameters in the feature extractor are frozen.
        
        Args:
            feature_extractor: The feature extractor to check
            
        Returns:
            True if all parameters are frozen, False otherwise
        """
        for param in feature_extractor.parameters():
            if param.requires_grad:
                return False
        return True
        
    def _are_all_parameters_trainable(self, feature_extractor: ResnetFE) -> bool:
        """
        Checks if all parameters in the feature extractor are trainable.
        
        Args:
            feature_extractor: The feature extractor to check
            
        Returns:
            True if all parameters are trainable, False otherwise
        """
        for param in feature_extractor.parameters():
            if not param.requires_grad:
                return False
        return True
    
    # @unittest.skip("passed")
    def test_invalid_freeze_configuration(self):
        """
        Tests that the combination of build_by_layer=False and freeze_by_layer=True raises an error.
        """
        # Test with each architecture
        for arch in self.architectures:
            with self.assertRaises(ValueError) as context:
                ResnetFE(
                    build_by_layer=False,
                    num_extracted_layers=1,
                    num_extracted_bottlenecks=1,
                    freeze=False,
                    freeze_by_layer=True,
                    add_global_average=True,
                    architecture=arch
                )
            
            self.assertTrue(
                "cannot be built by bottleneck and frozen by layer" in str(context.exception),
                "Expected error message about incompatible build and freeze configuration"
            )
    
    # @unittest.skip("passed")
    def test_boolean_freeze_parameter(self):
        """
        Tests that when freeze=True, all parameters are frozen, and when freeze=False, 
        all parameters are trainable, regardless of build_by_layer and extraction numbers.
        """
        for arch in self.architectures:
            # Test various combinations
            for build_by_layer in [True, False]:
                # Skip invalid configuration
                if not build_by_layer:
                    freeze_by_layer_values = [False]
                else:
                    freeze_by_layer_values = [True, False]
                    
                for freeze_by_layer in freeze_by_layer_values:
                    # Test with different numbers of extracted layers/bottlenecks
                    extracted_layers = random.randint(1, 4)
                    extracted_bottlenecks = random.randint(1, 10)
                    
                    # Test with freeze=True
                    feature_extractor_frozen = ResnetFE(
                        build_by_layer=build_by_layer,
                        num_extracted_layers=extracted_layers,
                        num_extracted_bottlenecks=extracted_bottlenecks,
                        freeze=True,  # All parameters should be frozen
                        freeze_by_layer=freeze_by_layer,
                        add_global_average=True,
                        architecture=arch
                    )
                    
                    self.assertTrue(
                        self._are_all_parameters_frozen(feature_extractor_frozen),
                        f"All parameters should be frozen when freeze=True, but some are trainable. "
                        f"Config: build_by_layer={build_by_layer}, freeze_by_layer={freeze_by_layer}, "
                        f"extracted_layers={extracted_layers}, extracted_bottlenecks={extracted_bottlenecks}"
                    )
                    
                    # Test with freeze=False
                    feature_extractor_trainable = ResnetFE(
                        build_by_layer=build_by_layer,
                        num_extracted_layers=extracted_layers,
                        num_extracted_bottlenecks=extracted_bottlenecks,
                        freeze=False,  # All parameters should be trainable
                        freeze_by_layer=freeze_by_layer,
                        add_global_average=True,
                        architecture=arch
                    )
                    
                    self.assertTrue(
                        self._are_all_parameters_trainable(feature_extractor_trainable),
                        f"All parameters should be trainable when freeze=False, but some are frozen. "
                        f"Config: build_by_layer={build_by_layer}, freeze_by_layer={freeze_by_layer}, "
                        f"extracted_layers={extracted_layers}, extracted_bottlenecks={extracted_bottlenecks}"
                    )
    
    # @unittest.skip("passed")
    def test_freeze_layer_blocks(self):
        """
        Tests that when freeze=N, build_by_layer=True, and freeze_by_layer=True,
        the number of frozen layer blocks is equal to N.
        """
        for arch in self.architectures:
            total_layers = self.layer_counts[arch]
            
            # Test with different numbers of layers to freeze
            for freeze_n in range(1, total_layers + 1):
                feature_extractor = ResnetFE(
                    build_by_layer=True,
                    num_extracted_layers=total_layers,  # Extract all layers
                    num_extracted_bottlenecks=1,  # This should be ignored
                    freeze=freeze_n,  # Freeze N layer blocks
                    freeze_by_layer=True,
                    add_global_average=True,
                    architecture=arch
                )
                
                # Count frozen layer blocks
                frozen_layers = self._count_frozen_layer_blocks(feature_extractor)
                
                self.assertEqual(
                    frozen_layers,
                    freeze_n,
                    f"Architecture {arch}, freeze={freeze_n}: Expected {freeze_n} frozen layer blocks, got {frozen_layers}"
                )
                
            # Test with freeze_n > total_layers (should freeze all layers)
            freeze_n = total_layers + random.randint(1, 5)
            feature_extractor = ResnetFE(
                build_by_layer=True,
                num_extracted_layers=total_layers,
                num_extracted_bottlenecks=1,
                freeze=freeze_n,
                freeze_by_layer=True,
                add_global_average=True,
                architecture=arch
            )
            
            frozen_layers = self._count_frozen_layer_blocks(feature_extractor)
            
            self.assertEqual(
                frozen_layers,
                total_layers,
                f"Architecture {arch}, freeze={freeze_n}: Expected {total_layers} frozen layer blocks, got {frozen_layers}"
            )
    
    # @unittest.skip("passed")
    def test_freeze_bottlenecks_build_by_layer(self):
        """
        Tests that when freeze=N, build_by_layer=True, and freeze_by_layer=False,
        the number of frozen bottleneck blocks is equal to N.
        """
        for arch in self.architectures:
            # Get the total number of bottlenecks in this architecture
            total_bottlenecks = sum(self.bottleneck_counts[arch].values())
            
            # Test with different numbers of bottlenecks to freeze
            for freeze_n in range(1, min(total_bottlenecks, 10) + 1):  # Test with up to 10 bottlenecks for efficiency
                feature_extractor = ResnetFE(
                    build_by_layer=True,
                    num_extracted_layers=len(self.bottleneck_counts[arch]),  # Extract all layers
                    num_extracted_bottlenecks=1,  # This should be ignored
                    freeze=freeze_n,  # Freeze N bottleneck blocks
                    freeze_by_layer=False,
                    add_global_average=True,
                    architecture=arch
                )
                
                # Count frozen bottleneck blocks
                frozen_bottlenecks = self._count_frozen_bottlenecks_by_layer(feature_extractor)
                
                self.assertEqual(
                    frozen_bottlenecks,
                    freeze_n,
                    f"Architecture {arch}, freeze={freeze_n}: Expected {freeze_n} frozen bottleneck blocks, got {frozen_bottlenecks}"
                )
    
    # @unittest.skip("passed")
    def test_freeze_bottlenecks_build_by_bottleneck(self):
        """
        Tests that when freeze=N, build_by_layer=False, and freeze_by_layer=False,
        the number of frozen bottleneck blocks is equal to N.
        """
        for arch in self.architectures:
            # Get the total number of bottlenecks in this architecture
            total_bottlenecks = sum(self.bottleneck_counts[arch].values())
            
            # Test with different numbers of bottlenecks to freeze
            for freeze_n in range(1, min(total_bottlenecks, 10) + 1):  # Test with up to 10 bottlenecks for efficiency
                feature_extractor = ResnetFE(
                    build_by_layer=False,
                    num_extracted_layers=1,  # This should be ignored
                    num_extracted_bottlenecks=total_bottlenecks,  # Extract all bottlenecks
                    freeze=freeze_n,  # Freeze N bottleneck blocks
                    freeze_by_layer=False,
                    add_global_average=True,
                    architecture=arch
                )
                
                # Count frozen bottleneck blocks
                frozen_bottlenecks = self._count_frozen_bottlenecks(feature_extractor)
                
                self.assertEqual(
                    frozen_bottlenecks,
                    freeze_n,
                    f"Architecture {arch}, freeze={freeze_n}: Expected {freeze_n} frozen bottleneck blocks, got {frozen_bottlenecks}"
                )

    ##########################################################
    # CUSTOM MODULE BASE TESTS
    ##########################################################
    
    # @unittest.skip("passed")
    def test_eval_mode(self):
        """Test that eval mode is correctly set across the feature extractor"""
        for arch in self.architectures:  
            for avg_layer in [True, False]:
                for bl in [True, False]:
                    feature_extractor = ResnetFE(
                        build_by_layer=bl,
                        num_extracted_layers=random.randint(1, 10),
                        num_extracted_bottlenecks=random.randint(1, 10),
                        freeze=False,
                        freeze_by_layer=bl,
                        add_global_average=avg_layer,
                        architecture=arch
                    )
                    super()._test_eval_mode(feature_extractor)

    # @unittest.skip("passed")
    def test_train_mode(self):
        """Test that train mode is correctly set across the feature extractor"""
        for arch in self.architectures:  
            for avg_layer in [True, False]:
                for bl in [True, False]:
                    feature_extractor = ResnetFE(
                        build_by_layer=bl,
                        num_extracted_layers=random.randint(1, 10),
                        num_extracted_bottlenecks=random.randint(1, 10),
                        freeze=False,
                        freeze_by_layer=bl,
                        add_global_average=avg_layer,
                        architecture=arch
                    )
                    super()._test_train_mode(feature_extractor)

    # @unittest.skip("passed")
    def test_consistent_output_in_eval_mode(self):
        """Test that the feature extractor produces consistent output in eval mode"""
        for arch in self.architectures:  
            for avg_layer in [True, False]:
                for bl in [True, False]:
                    feature_extractor = ResnetFE(
                        build_by_layer=bl,
                        num_extracted_layers=random.randint(1, 10),
                        num_extracted_bottlenecks=random.randint(1, 10),
                        freeze=False,
                        freeze_by_layer=bl,
                        add_global_average=avg_layer,
                        architecture=arch
                        )
                    input_tensor = torch.randn(random.randint(1, 10), 3, 224, 224)
                    super()._test_consistent_output_in_eval_mode(feature_extractor, input_tensor)

    # @unittest.skip("passed")
    def test_batch_size_one_in_eval_mode(self):
        """Test that the feature extractor handles batch size 1 in eval mode"""
        for arch in self.architectures:  
            for avg_layer in [True, False]:
                for bl in [True, False]:
                    feature_extractor = ResnetFE(
                        build_by_layer=bl,
                        num_extracted_layers=random.randint(1, 10),
                        num_extracted_bottlenecks=random.randint(1, 10),
                        freeze=False,
                        freeze_by_layer=bl,
                        add_global_average=avg_layer,
                        architecture=arch
                    )
                    input_tensor = torch.randn(1, 3, 224, 224)
                    super()._test_batch_size_one_in_eval_mode(feature_extractor, input_tensor)
    
    # @unittest.skip("passed")
    def test_named_parameters_length(self):
        """Test that named_parameters and parameters have the same length"""
        for arch in self.architectures:  
            for avg_layer in [True, False]:
                for bl in [True, False]:
                    feature_extractor = ResnetFE(
                        build_by_layer=bl,
                        num_extracted_layers=random.randint(1, 10),
                        num_extracted_bottlenecks=random.randint(1, 10),
                        freeze=False,
                        freeze_by_layer=bl,
                        add_global_average=avg_layer,
                        architecture=arch
                    )
                    super()._test_named_parameters_length(feature_extractor)

    # @unittest.skip("passed")
    def test_batch_size_one_in_train_mode(self):
        """Test that the feature extractor handles batch size 1 in train mode"""
        for arch in self.architectures:  
            for avg_layer in [True, False]:
                for bl in [True, False]:
                    feature_extractor = ResnetFE(
                        build_by_layer=bl,
                        num_extracted_layers=random.randint(1, 10),
                        num_extracted_bottlenecks=random.randint(1, 10),
                        freeze=False,
                        freeze_by_layer=bl,
                        add_global_average=avg_layer,
                        architecture=arch
                    )
                    input_tensor = torch.randn(1, 3, 224, 224)
                    super()._test_batch_size_one_in_train_mode(feature_extractor, input_tensor)
    

    # @unittest.skip("passed")
    def test_batch_size_one_in_eval_mode(self):
        """Test that the feature extractor handles batch size 1 in eval mode"""
        for arch in self.architectures:  
            for avg_layer in [True, False]:
                for bl in [True, False]:
                    feature_extractor = ResnetFE(
                        build_by_layer=bl,
                        num_extracted_layers=random.randint(1, 10),
                        num_extracted_bottlenecks=random.randint(1, 10),
                        freeze=False,
                        freeze_by_layer=bl,
                        add_global_average=avg_layer,
                        architecture=arch
                    )
                    input_tensor = torch.randn(1, 3, 224, 224)
                    super()._test_batch_size_one_in_eval_mode(feature_extractor, input_tensor)


    # @unittest.skip("passed")
    def test_to_device(self):
        """Test that the feature extractor can be moved to a device"""
        for arch in self.architectures:  
            for avg_layer in [True, False]:
                for bl in [True, False]:
                    feature_extractor = ResnetFE(
                        build_by_layer=bl,  
                        num_extracted_layers=random.randint(1, 10),
                        num_extracted_bottlenecks=random.randint(1, 10),
                        freeze=False,
                        freeze_by_layer=bl,
                        add_global_average=avg_layer,
                        architecture=arch   
                    )
                    input_tensor = torch.randn(1, 3, 224, 224)
                    super()._test_to_device(feature_extractor, input_tensor)



if __name__ == '__main__':
    pu.seed_everything(42)
    unittest.main()    