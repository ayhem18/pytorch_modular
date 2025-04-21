import unittest
import torch
import random
import mypt.code_utils.pytorch_utils as pu

from torch import nn
from typing import Dict
from mypt.backbones.resnetFE import ResnetFE

from torchvision.models.resnet import Bottleneck, BasicBlock
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights



class TestResnetFE(unittest.TestCase):
    """
    Test class for ResnetFE feature extractor implementation.
    Tests construction with different architectures and extraction strategies.
    """
    
    def setUp(self):
        """
        Initialize tests for all ResNet architectures.
        Calculates the number of layer blocks and residual blocks in each architecture.
        """
        self.architectures = ResnetFE.__archs__
        
        # Create dictionaries to store layer and bottleneck counts for each architecture
        self.layer_counts = {}
        self.bottleneck_counts = {}
        
        # Compute counts for each architecture
        for arch in self.architectures:
            self.layer_counts[arch] = self._compute_layers(arch)
            self.bottleneck_counts[arch] = self._compute_bottlenecks(arch)
        
        # Print summary for debugging
        print(f"Layer counts by architecture: {self.layer_counts}")
        print(f"Bottleneck counts by architecture: {self.bottleneck_counts}")
    
    def _compute_layers(self, architecture: int) -> int:
        """
        Computes the number of layer blocks in a given ResNet architecture.
        
        Args:
            architecture: ResNet architecture (50, 101, or 152)
            
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
    
    def _compute_bottlenecks(self, architecture: int) -> Dict[str, int]:
        """
        Computes the number of residual blocks in each layer of a given ResNet architecture.
        
        Args:
            architecture: ResNet architecture (50, 101, or 152)
            
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
    
    def test_build_by_layer(self):
        """
        Tests the feature extractor construction when building by layer.
        Verifies that the correct number of layers are extracted based on the input parameters.
        """
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

            for _ in range(20):
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
                    f"Architecture {arch}, num_extracted_layers={i}: Expected {i} layers, got {actual_layers}"
                )

            for _ in range(20):
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
                
    
    def test_build_by_bottleneck(self):
        """
        Tests the feature extractor construction when building by bottleneck.
        Verifies that the correct number of bottlenecks are extracted based on the input parameters.
        """
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
            
            # Test with random negative values (should extract all bottlenecks)
            for _ in range(20):
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
            
            # Test with random values beyond the total (should extract all bottlenecks)
            for _ in range(20):
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


if __name__ == '__main__':
    pu.seed_everything(42)
    unittest.main()
