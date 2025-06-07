import unittest
import torch
import random
import mypt.code_utils.pytorch_utils as pu

from torch import nn
from typing import Dict
from mypt.backbones.resnetFE import ResnetFE
from torchvision.models.resnet import Bottleneck

from tests.custom_base_test import CustomModuleBaseTest


# @unittest.skip("skipping resnet tests for now")
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
    




    @unittest.skip("skipping layer tests for now")
    def test_build_by_layer_1_total_layers(self):
        for arch in self.architectures:
            total_layers = self.layer_counts[arch]
            
            # Test with different numbers of layers to extract
            for i in range(1, total_layers + 1):
                # Create a feature extractor
                feature_extractor = ResnetFE(
                    build_by_layer=True,
                    num_extracted_layers=i,
                    num_extracted_bottlenecks=i,  # This should be ignored when building by layer
                    freeze=False,
                    freeze_by_layer=True,
                    add_global_average=True,
                    architecture=arch
                )
                                
                # Count actual layers in the feature extractor
                actual_layers = self._count_layers_in_feature_extractor(feature_extractor)
                
                self.assertEqual(
                    actual_layers, 
                    i, 
                    f"Architecture {arch}, num_extracted_layers={i}: Expected {i} layers, got {actual_layers}"
                )

    @unittest.skip("skipping layer tests for now")
    def test_build_by_layer_random_negative_values(self):
        for arch in self.architectures:
            total_layers = self.layer_counts[arch]
                
            for _ in range(10):
                # test random  negative values 
                num_layers = random.randint(-100, -1) 

                # Create a feature extractor
                feature_extractor = ResnetFE(
                    build_by_layer=True,
                    num_extracted_layers=num_layers,
                    num_extracted_bottlenecks=total_layers,  # This should be ignored when building by layer
                    freeze=False,
                    freeze_by_layer=True,
                    add_global_average=True,
                    architecture=arch
                )
                                
                # Count actual layers in the feature extractor
                actual_layers = self._count_layers_in_feature_extractor(feature_extractor)
                
                self.assertEqual(
                    actual_layers, 
                    total_layers, 
                    f"Architecture {arch}, num_extracted_layers={actual_layers}: Expected {total_layers} layers, got {actual_layers}"
                )
                
    @unittest.skip("skipping layer tests for now")
    def test_build_by_layer_beyond_total_layers(self):
        for arch in self.architectures:
            total_layers = self.layer_counts[arch]

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
                    add_global_average=True,
                    architecture=arch
                )
                                
                # Count actual layers in the feature extractor
                actual_layers = self._count_layers_in_feature_extractor(feature_extractor)  
                
                self.assertEqual(
                    actual_layers, 
                    total_layers, 
                    f"Architecture {arch}, num_extracted_layers={i}: Expected {total_layers} layers, got {actual_layers}"
                )

    # @unittest.skip("skipping bottleneck tests for now")      
    def test_build_by_bottleneck_1_total_bottlenecks(self):
        for arch in self.architectures:
            # Get total bottlenecks by summing across all layers
            total_bottlenecks = sum(self.bottleneck_counts[arch].values())
            
            # Test with explicit values from 1 to total bottlenecks
            for i in range(1, total_bottlenecks + 1):
                # Create a feature extractor
                feature_extractor = ResnetFE(
                    build_by_layer=False,
                    num_extracted_layers=1,  # This should be ignored when building by bottleneck
                    num_extracted_bottlenecks=i,
                    freeze=False,
                    freeze_by_layer=False,
                    add_global_average=True,
                    architecture=arch
                )
                
                # Count actual bottlenecks in the feature extractor
                actual_bottlenecks = self._count_bottlenecks_in_feature_extractor(feature_extractor)
                
                self.assertEqual(
                    actual_bottlenecks, 
                    i, 
                    f"Architecture {arch}, num_extracted_bottlenecks={i}: Expected {i} bottlenecks, got {actual_bottlenecks}"
                )
    
    @unittest.skip("skipping bottleneck tests for now")
    def test_build_by_bottleneck_random_negative_values(self):
        for arch in self.architectures:
            total_bottlenecks = sum(self.bottleneck_counts[arch].values())

            # Test with random negative values (should extract all bottlenecks)
            for _ in range(10):
                num_bottlenecks = random.randint(-100, -1)
                
                feature_extractor = ResnetFE(
                    build_by_layer=False,
                    num_extracted_layers=1,  # This should be ignored when building by bottleneck
                    num_extracted_bottlenecks=num_bottlenecks,
                    freeze=False,
                    freeze_by_layer=False,
                    add_global_average=True,
                    architecture=arch
                )
                
                actual_bottlenecks = self._count_bottlenecks_in_feature_extractor(feature_extractor)
                
                self.assertEqual(
                    actual_bottlenecks, 
                    total_bottlenecks, 
                    f"Architecture {arch}, num_extracted_bottlenecks={num_bottlenecks}: Expected {total_bottlenecks} bottlenecks, got {actual_bottlenecks}"
                )

    @unittest.skip("skipping bottleneck tests for now")
    def test_build_by_bottleneck_beyond_total_bottlenecks(self):
        for arch in self.architectures:
            total_bottlenecks = sum(self.bottleneck_counts[arch].values())

            # Test with random values beyond the total (should extract all bottlenecks)
            for _ in range(20):
                num_bottlenecks = random.randint(total_bottlenecks + 1, 1000)
                num_bottlenecks = random.randint(total_bottlenecks + 1, 1000)
                
                feature_extractor = ResnetFE(
                    build_by_layer=False,
                    num_extracted_layers=1,  # This should be ignored when building by bottleneck
                    num_extracted_bottlenecks=num_bottlenecks,
                    freeze=False,
                    freeze_by_layer=False,
                    add_global_average=True,
                    architecture=arch
                )
                
                actual_bottlenecks = self._count_bottlenecks_in_feature_extractor(feature_extractor)
                
                self.assertEqual(
                    actual_bottlenecks, 
                    total_bottlenecks, 
                    f"Architecture {arch}, num_extracted_bottlenecks={num_bottlenecks}: Expected {total_bottlenecks} bottlenecks, got {actual_bottlenecks}"
                )

    @unittest.skip("skipping input validation tests for now")
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
        self.assertTrue("number of extracted layers cannot be zeor" in str(context.exception))
        
        # Test case 2: Passing 0 for num_extracted_bottlenecks when build_by_layer=False should raise ValueError
        with self.assertRaises(ValueError) as context:
            ResnetFE(
                build_by_layer=False,
                num_extracted_layers=1,
                num_extracted_bottlenecks=0,
                freeze=False,
                freeze_by_layer=True,
                add_global_average=True,
                architecture=50
            )
        self.assertTrue("number of extracted blocks cannot be zeor" in str(context.exception))
        
        # Test case 3: Passing 0 for num_extracted_layers when build_by_layer=False should NOT raise error
        try:
            ResnetFE(
                build_by_layer=False,
                num_extracted_layers=0,
                num_extracted_bottlenecks=1,
                freeze=False,
                freeze_by_layer=True,
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

    @unittest.skip("skipping layer tests for now")
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

    @unittest.skip("skipping layer tests for now")
    # Custom module base tests for ResnetFE
    def test_eval_mode(self):
        """Test that eval mode is correctly set across the feature extractor"""
        for arch in self.architectures:  
            feature_extractor = ResnetFE(
                build_by_layer=True,
                num_extracted_layers=2,
                num_extracted_bottlenecks=2,
                freeze=False,
                freeze_by_layer=True,
                add_global_average=True,
                architecture=arch
            )

            super()._test_eval_mode(feature_extractor)
    
    @unittest.skip("skipping layer tests for now")
    def test_train_mode(self):
        """Test that train mode is correctly set across the feature extractor"""
        for arch in self.architectures:  
            feature_extractor = ResnetFE(
                build_by_layer=True,
                num_extracted_layers=2,
                num_extracted_bottlenecks=2,
                freeze=False,
                freeze_by_layer=True,
                add_global_average=True,
                architecture=arch
            )
            super()._test_train_mode(feature_extractor)
    
    @unittest.skip("skipping layer tests for now")
    def test_consistent_output_in_eval_mode(self):
        """Test that the feature extractor produces consistent output in eval mode"""
        for arch in self.architectures:  
            feature_extractor = ResnetFE(
                build_by_layer=True,
                num_extracted_layers=2,
                num_extracted_bottlenecks=2,
                freeze=False,
                freeze_by_layer=True,
                add_global_average=True,
                architecture=arch
            )
            input_tensor = torch.randn(random.randint(1, 10), 3, 224, 224)
            super()._test_consistent_output_in_eval_mode(feature_extractor, input_tensor)
    
    @unittest.skip("skipping layer tests for now")
    def test_batch_size_one_in_eval_mode(self):
        """Test that the feature extractor handles batch size 1 in eval mode"""
        for arch in self.architectures:  
            feature_extractor = ResnetFE(
                build_by_layer=True,
                num_extracted_layers=2,
                num_extracted_bottlenecks=2,
                freeze=False,
                freeze_by_layer=True,
                add_global_average=True,
                architecture=arch
            )
            input_tensor = torch.randn(1, 3, 224, 224)
            super()._test_batch_size_one_in_eval_mode(feature_extractor, input_tensor)
    
    @unittest.skip("skipping layer tests for now")
    def test_named_parameters_length(self):
        """Test that named_parameters and parameters have the same length"""
        for arch in self.architectures:  
            feature_extractor = ResnetFE(
                build_by_layer=True,
                num_extracted_layers=2,
                num_extracted_bottlenecks=2,
                freeze=False,
                freeze_by_layer=True,
                add_global_average=True,
                architecture=arch
            )
            super()._test_named_parameters_length(feature_extractor)
            


if __name__ == '__main__':
    pu.seed_everything(42)
    unittest.main()
    from mypt.backbones.resnetFE import ResnetFE 

    # constructor, weights = ResnetFE.get_model(architecture=50)
    # net = constructor(weights=weights.DEFAULT) 
    # print(net)