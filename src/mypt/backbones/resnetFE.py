"""
This script contains functionalities to build classifiers on top of the pretrained model
'RESNET 50' provided by pytorch module

This script is mainly inspired by this paper: 'https://arxiv.org/abs/1411.1792'
as they suggest an experimental framework to find the most transferable / general layers
in pretrained network. I am applying the same framework on the resnet architecture.
"""

import warnings

from torch import nn
from typing import Callable, Union, Tuple, Any
from collections import OrderedDict, defaultdict, deque

from torchvision.models.resnet import Bottleneck, BasicBlock
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights

from mypt.building_blocks.mixins.custom_module_mixins import WrapperLikeModuleMixin  


class ResnetFE(WrapperLikeModuleMixin):
    __archs__ = [18, 34, 50, 101, 152]
    
    archs_dict = {18: (resnet18, ResNet18_Weights), 
                 34: (resnet34, ResNet34_Weights), 
                 50: (resnet50, ResNet50_Weights), 
                 101: (resnet101, ResNet101_Weights), 
                 152: (resnet152, ResNet152_Weights)}

    LAYER_BLOCK = 'layer'
    RESIDUAL_BLOCK = 'residual'
    DEFAUL_NUM_CLASSES = 1000

    @classmethod    
    def get_model(cls, architecture: int) -> Tuple[nn.Module, Any]:
        if architecture not in cls.__archs__:
            warnings.warn(f'The value {architecture} was passed as architecture. No such architecture exists. Defaulting to {50}')
            architecture = 50

        return cls.archs_dict[architecture]

    # =========================================================================
    # ResNet 50, 101, 152 Methods (Bottleneck)
    # =========================================================================
    def __extract_fe_by_layer_50_plus(self):
        layer_blocks_counter = 0
        extracted_modules = []

        for module_name, module in self.__net.named_children():
            if isinstance(module, nn.AdaptiveAvgPool2d):
                if self._add_global_average:
                    extracted_modules.append((module_name, module))
                continue

            if isinstance(module, (nn.Linear, nn.LazyLinear)):
                continue
            
            mc = list(module.named_children())

            if len(mc) == 0:
                extracted_modules.append((module_name, module))
                continue
            
            if self.LAYER_BLOCK.lower() not in module_name.lower():
                raise TypeError(f"The class is based on the wrong assumptions. Found a block with children that is not a layer block !!! it is named: {module_name}")

            self.blocks_per_layer[layer_blocks_counter + 1] = len(list(module.children()))

            if layer_blocks_counter >= self._num_extracted_layers:
                continue

            extracted_modules.append((module_name, module))
            layer_blocks_counter += 1   

        self._feature_extractor = nn.Sequential(OrderedDict(extracted_modules))
            

    def __extract_fe_by_bottleneck_50_plus(self):
        extracted_modules = deque()
        layer_counter = 0
        bottleneck_counter = 0

        for module_name, module in self.__net.named_children():
            if isinstance(module, nn.AdaptiveAvgPool2d):
                if self._add_global_average:
                    extracted_modules.append((module_name, module))
                continue
                
            if isinstance(module, (nn.Linear, nn.LazyLinear)):
                continue
            
            mc = list(module.named_children())

            if len(mc) == 0:
                extracted_modules.append((module_name, module)) 
                continue
            
            if self.LAYER_BLOCK.lower() not in module_name.lower():
                raise TypeError(f"The class is based on the wrong assumptions. Found a block with children that is not a layer block !!! it is named: {module_name}")

            layer_counter += 1

            for name, child in mc:
                if not isinstance(child, Bottleneck):
                    raise TypeError(f"The class is based on the wrong assumptions. Found a block with children that is not a Bottleneck block !!! it is named: {name}")

                if bottleneck_counter >= self._num_extracted_bottlenecks:
                    break
                
                if self.RESIDUAL_BLOCK.lower() not in name.lower():
                    name = module_name + '_' + f'{bottleneck_counter + 1}'
                
                extracted_modules.append((name, child))
                bottleneck_counter += 1    

                self.blocks_per_layer[layer_counter] += 1

        self._feature_extractor = nn.Sequential(OrderedDict(extracted_modules))


    def __freeze_by_bottleneck_build_by_layer_50_plus(self, num_bottlenecks: int):
        bottleneck_counter = 0 
        layer_count = 0

        for name, module in self._feature_extractor.named_children():
            if isinstance(module, nn.AdaptiveAvgPool2d):                    
                continue
            
            mc = list(module.named_children())

            if len(mc) == 0:
                for param in module.parameters():
                    param.requires_grad = False
                continue

            if not self.LAYER_BLOCK.lower() in name.lower():
                raise TypeError(f"The class is based on the wrong assumptions. Found a block with children that is not a layer block !!! it is named: {name}")

            layer_count += 1

            if bottleneck_counter >= num_bottlenecks:
                continue

            next_frozen_num_bottlenecks = bottleneck_counter + self.blocks_per_layer[layer_count]

            if next_frozen_num_bottlenecks <= num_bottlenecks:
                for param in module.parameters():
                    param.requires_grad = False

                bottleneck_counter = next_frozen_num_bottlenecks
                continue
                
            for bottleneck in module.children():
                if bottleneck_counter >= num_bottlenecks:
                    break

                for param in bottleneck.parameters():
                    param.requires_grad = False

                bottleneck_counter += 1

    # =========================================================================
    # ResNet 18, 34 Methods (BasicBlock)
    # =========================================================================
    def __extract_fe_by_layer_18_34(self):
        layer_blocks_counter = 0
        extracted_modules = []

        for module_name, module in self.__net.named_children():
            if isinstance(module, nn.AdaptiveAvgPool2d):
                if self._add_global_average:
                    extracted_modules.append((module_name, module))
                continue

            if isinstance(module, (nn.Linear, nn.LazyLinear)):
                continue
            
            mc = list(module.named_children())

            if len(mc) == 0:
                extracted_modules.append((module_name, module))
                continue
            
            if self.LAYER_BLOCK.lower() not in module_name.lower():
                raise TypeError(f"The class is based on the wrong assumptions. Found a block with children that is not a layer block !!! it is named: {module_name}")

            self.blocks_per_layer[layer_blocks_counter + 1] = len(list(module.children()))

            if layer_blocks_counter >= self._num_extracted_layers:
                continue

            extracted_modules.append((module_name, module))
            layer_blocks_counter += 1   

        self._feature_extractor = nn.Sequential(OrderedDict(extracted_modules))

    def __extract_fe_by_basic_block_18_34(self):
        extracted_modules = deque()
        layer_counter = 0
        basic_block_counter = 0

        for module_name, module in self.__net.named_children():
            if isinstance(module, nn.AdaptiveAvgPool2d):
                if self._add_global_average:
                    extracted_modules.append((module_name, module))
                continue
                
            if isinstance(module, (nn.Linear, nn.LazyLinear)):
                continue
            
            mc = list(module.named_children())

            if len(mc) == 0:
                extracted_modules.append((module_name, module)) 
                continue
            
            if self.LAYER_BLOCK.lower() not in module_name.lower():
                raise TypeError(f"The class is based on the wrong assumptions. Found a block with children that is not a layer block !!! it is named: {module_name}")

            layer_counter += 1

            for name, child in mc:
                if not isinstance(child, BasicBlock):
                    raise TypeError(f"The class is based on the wrong assumptions. Found a block with children that is not a BasicBlock !!! it is named: {name}")

                if basic_block_counter >= self._num_extracted_bottlenecks:
                    break
                
                if self.RESIDUAL_BLOCK.lower() not in name.lower():
                    name = module_name + '_' + f'{basic_block_counter + 1}'
                
                extracted_modules.append((name, child))
                basic_block_counter += 1    

                self.blocks_per_layer[layer_counter] += 1

        self._feature_extractor = nn.Sequential(OrderedDict(extracted_modules))

    def __freeze_by_basic_block_build_by_layer_18_34(self, num_basic_blocks: int):
        basic_block_counter = 0 
        layer_count = 0

        for name, module in self._feature_extractor.named_children():
            if isinstance(module, nn.AdaptiveAvgPool2d):                    
                continue
            
            mc = list(module.named_children())

            if len(mc) == 0:
                for param in module.parameters():
                    param.requires_grad = False
                continue

            if not self.LAYER_BLOCK.lower() in name.lower():
                raise TypeError(f"The class is based on the wrong assumptions. Found a block with children that is not a layer block !!! it is named: {name}")

            layer_count += 1

            if basic_block_counter >= num_basic_blocks:
                continue

            next_frozen_num_basic_blocks = basic_block_counter + self.blocks_per_layer[layer_count]

            if next_frozen_num_basic_blocks <= num_basic_blocks:
                for param in module.parameters():
                    param.requires_grad = False

                basic_block_counter = next_frozen_num_basic_blocks
                continue
                
            for basic_block in module.children():
                if basic_block_counter >= num_basic_blocks:
                    break

                for param in basic_block.parameters():
                    param.requires_grad = False

                basic_block_counter += 1

    # =========================================================================
    # Common Freezing Method (Used when build_by_layer == freeze_by_layer)
    # =========================================================================
    def __freeze_by_building_block(self, num_blocks: int):
        block_counter = 0

        for module in self._feature_extractor.children():
            if isinstance(module, nn.AdaptiveAvgPool2d) or isinstance(module, (nn.Linear, nn.LazyLinear)):                    
                continue
            
            mc = list(module.named_children())

            if len(mc) == 0:
                for param in module.parameters():
                    param.requires_grad = False
                continue
                        
            if block_counter >= num_blocks:
                continue

            for param in module.parameters():
                param.requires_grad = False

            block_counter += 1   

    def _freeze_feature_extractor(self):
        if isinstance(self._freeze, bool):
            if self._freeze:
                for param in self._feature_extractor.parameters():
                    param.requires_grad = False
            return 
        
        if self._build_by_layer == self._freeze_by_layer:
            self.__freeze_by_building_block(self._freeze)
            return 
        
        if self._architecture in [18, 34]:
            self.__freeze_by_basic_block_build_by_layer_18_34(self._freeze)
        else:
            self.__freeze_by_bottleneck_build_by_layer_50_plus(self._freeze)

    def _verify_input(self,
                      build_by_layer: bool,
                      num_extracted_layers: int,
                      num_extracted_bottlenecks: int,
                      freeze_by_layer: bool,
                      ):
        if not build_by_layer and freeze_by_layer:
            raise ValueError("The feature extractor cannot be built by bottleneck/basic_block and frozen by layer at the same time. However, it can be build by layers and frozen by bottleneck/basic_block.")

        if build_by_layer and num_extracted_layers == 0:
            raise ValueError("The number of extracted layers cannot be zero. Negative values indicate all layers should be extracted.")

        if not build_by_layer and num_extracted_bottlenecks == 0:
            raise ValueError("The number of extracted blocks cannot be zero. Negative values indicate all blocks should be extracted.")

        if num_extracted_layers < 0 and num_extracted_layers != -1:
            raise ValueError("The only negative value allowed for `num_extracted_layers` is -1. This indicates that all layers should be extracted.")

        if num_extracted_bottlenecks < 0 and num_extracted_bottlenecks != -1:
            raise ValueError("The only negative value allowed for `num_extracted_bottlenecks` is -1. This indicates that all bottlenecks should be extracted.")

    def __init__(self, 
                 build_by_layer: bool,
                 num_extracted_layers: int,
                 num_extracted_bottlenecks: int,
                 freeze: Union[bool, int],
                 freeze_by_layer: bool,
                 add_global_average: bool,
                 architecture: int = 50,
                 *args, **kwargs):

        super().__init__("_feature_extractor", *args, **kwargs)

        self._verify_input(build_by_layer, num_extracted_layers, num_extracted_bottlenecks, freeze_by_layer)

        self._feature_extractor: nn.Module = None
        self._build_by_layer = build_by_layer
        self._architecture = architecture

        self._num_extracted_layers = num_extracted_layers if num_extracted_layers > 0 else float('inf') 
        self._num_extracted_bottlenecks = num_extracted_bottlenecks if num_extracted_bottlenecks > 0 else float('inf') 
        
        self._freeze = freeze
        self._freeze_by_layer = freeze_by_layer
        
        self._add_global_average = add_global_average
        
        constructor, weights = self.get_model(architecture=architecture)
        self.__net = constructor(weights=weights.DEFAULT) 
        self._transform = weights.DEFAULT.transforms()

        self.blocks_per_layer = defaultdict(lambda: 0)

        if self._architecture in [18, 34]:
            if self._build_by_layer:
                self.__extract_fe_by_layer_18_34()
            else:
                self.__extract_fe_by_basic_block_18_34()
        else:
            if self._build_by_layer:
                self.__extract_fe_by_layer_50_plus()
            else:
                self.__extract_fe_by_bottleneck_50_plus()  

        del(self.__net)

        self._freeze_feature_extractor()

    @property
    def transform(self) -> Callable:
        return self._transform
