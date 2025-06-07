"""
This script contains functionalities to build classifiers on top of the pretrained model
'RESNET 50' provided by pytorch module

This script is mainly inspired by this paper: 'https://arxiv.org/abs/1411.1792'
as they suggest an experimental framework to find the most transferable / general layers
in pretrained network. I am applying the same framework on the resnet architecture.
"""

import warnings

from torch import nn
from collections import OrderedDict, defaultdict, deque
from typing import Union, Tuple, Any
from torchvision.models import resnet50, resnet101, resnet152
from torchvision.models import ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from torchvision.models.resnet import Bottleneck


from mypt.building_blocks.mixins.custom_module_mixins import WrapperLikeModuleMixin  


class ResnetFE(WrapperLikeModuleMixin):
    __archs__ = [50, 101, 152]
    
    # archs_dict = {18: (resnet18, ResNet18_Weights), # the resnet18 model does not use the bottleck block  
    # archs_dict = {34: (resnet34, ResNet34_Weights), # the resnet34 model does not use the bottleck block 
    archs_dict = {50: (resnet50, ResNet50_Weights), 
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


    def __extract_feature_extractory_by_layer(self):
        """
        This method build a feature extractor using the "layer" as a building block.
        """

        layer_blocks_counter = 0
        extracted_modules = []

        for module_name, module in self.__net.named_children():
            # there are 3 types of immediate children for the resnet model:
            # either a layer block
            # either a relu, maxpool, or a convolutional layer: those appear at the very beginning of the network
            # either an average pooling layer: this appears right after the last_layer_block


            if isinstance(module, nn.AdaptiveAvgPool2d):
                if self._add_global_average:
                    extracted_modules.append((module_name, module))
                    
                # make sure to move to the next iteration since the 3rd condition will also be met
                continue

            if isinstance(module, (nn.Linear, nn.LazyLinear)):
                # ignore fully connected layers
                continue
            
            mc = list(module.named_children())

            if len(mc) == 0:
                # if the module has no children, then it is a simple layer and must be one of the layers before the first layer block
                extracted_modules.append((module_name, module))
                continue
            
            # the assumption here is that the child is a layer block
            if self.LAYER_BLOCK.lower() not in module_name.lower():
                raise TypeError("The class is based on the wrong assumptions. Found a block with children that is not a layer block !!! it is named: {name}")

            self.bottleneck_per_layer[layer_blocks_counter + 1] = len(list(module.children()))

            # at this point, the child is a layer block
            if layer_blocks_counter >= self._num_extracted_layers:
                continue

            # add the layer block to the feature extractor  
            extracted_modules.append((module_name, module))
            layer_blocks_counter += 1   

        self._feature_extractor = nn.Sequential(OrderedDict(extracted_modules))
            

    def __extract_feature_extractory_by_bottleneck(self):
        """
        This method build a feature extractor using the "bottleneck" as a building block.
        """
        extracted_modules = deque()
        layer_counter = 0
        bottleneck_counter = 0

        for module_name, module in self.__net.named_children():
            # there are 3 types of immediate children for the resnet model:
            # either a layer block
            # either a relu, maxpool, or a convolutional layer: those appear at the very beginning of the network
            # either an average pooling layer: this appears right after the last_layer_block

            if isinstance(module, nn.AdaptiveAvgPool2d):
                if self._add_global_average:
                    extracted_modules.append((module_name, module))
                    
                # make sure to move to the next iteration since the 3rd condition will also be met
                continue
                
            if isinstance(module, (nn.Linear, nn.LazyLinear)):
                # ignore fully connected layers
                continue
            
            mc = list(module.named_children())

            if len(mc) == 0:
                # if the module has no children, then it is one of the simple modules before the first "resnet layers"
                extracted_modules.append((module_name, module)) 
                continue
            

            # the assumption here is that the child is a layer block
            if self.LAYER_BLOCK.lower() not in module_name.lower():
                raise TypeError("The class is based on the wrong assumptions. Found a block with children that is not a layer block !!! it is named: {name}")

            layer_counter += 1

            for name, child in mc:
                if not isinstance(child, Bottleneck):
                    raise TypeError("The class is based on the wrong assumptions. Found a block with children that is not a layer block !!! it is named: {name}")

                # at this point, the child is a residual block
                if bottleneck_counter >= self._num_extracted_bottlenecks:
                    # we use break since, within the current layer block, we will not add any more residual blocks to the feature extractor.
                    break
                
                if self.RESIDUAL_BLOCK.lower() not in name.lower():
                    name = module_name + '_' + f'{bottleneck_counter + 1}'
                
                # add the residual block to the feature extractor  
                extracted_modules.append((name, child))
                bottleneck_counter += 1    

                self.bottleneck_per_layer[layer_counter] += 1

        self._feature_extractor = nn.Sequential(OrderedDict(extracted_modules))


    def __freeze_by_building_block(self, num_blocks: int):
        """
        This method freezes the weights of the feature extractor when build_by_layer and freeze_by_layer are of the same value (both true or both false)

        Args:
            num_blocks (int): the number of blocks to freeze
        """
        block_counter = 0

        for module in self._feature_extractor.children():
            # there are 3 types of immediate children for the resnet model:
            # either a layer block
            # either a relu, maxpool, or a convolutional layer: those appear at the very beginning of the network
            # either an average pooling layer: this appears right after the last_layer_block

            if isinstance(module, nn.AdaptiveAvgPool2d) or isinstance(module, (nn.Linear, nn.LazyLinear)):                    
                # linear and adaptive average pooling layers are not frozen
                continue
            
            mc = list(module.named_children())

            if len(mc) == 0:
                # the module is a simple layer and must be one of the layers before the first layer block so freeze it
                for param in module.parameters():
                    param.requires_grad = False

                continue
                        

            # at this point, the child is a layer block
            if block_counter >= num_blocks:
                continue

            # freeze the block    
            for param in module.parameters():
                param.requires_grad = False

            block_counter += 1   


    def __freeze_by_bottleneck_build_by_layer(self, num_bottlenecks: int):
        """
        This method freezes the weights of the feature extractor when build_by_layer = True and freeze_by_layer = False
        """
        bottleneck_counter = 0 

        for name, module in self._feature_extractor.named_children():
            if isinstance(module, nn.AdaptiveAvgPool2d) or isinstance(module, (nn.Linear, nn.LazyLinear)):                    
                # linear and adaptive average pooling layers are not frozen
                continue
            
            mc = list(module.named_children())

            if len(mc) == 0:
                # the module is a simple layer and must be one of the layers before the first layer block 
                # freeze it
                for param in module.parameters():
                    param.requires_grad = False

            # at this point, the code assumes that the module is a layer
            if not self.LAYER_BLOCK.lower() in name.lower():
                raise TypeError("The class is based on the wrong assumptions. Found a block with children that is not a layer block !!! it is named: {name}")

            if bottleneck_counter >= num_bottlenecks:
                continue

            # at this point, we need to either freeze the entire block or some bottleneck layers within the block
            # compute the number of frozen bottleneck layers if we are to freeze the entire block
            next_frozen_num_bottlenecks = bottleneck_counter + self.bottleneck_per_layer[bottleneck_counter + 1]

            if next_frozen_num_bottlenecks <= num_bottlenecks:
                # in this case, then freeze the entire block
                for param in module.parameters():
                    param.requires_grad = False

                bottleneck_counter += next_frozen_num_bottlenecks
                continue
                
            # at this point, freezing the entire block would exceed the number of frozen bottleneck layers 
            # we need to freeze some bottleneck layers within the block 
            for bottleneck in module.children():
                if bottleneck_counter >= num_bottlenecks:
                    break

                for param in bottleneck.parameters():
                    param.requires_grad = False

                bottleneck_counter += 1


    def _freeze_feature_extractor(self):
        # if the freeze is a boolean variable, then either freeze all parameters or leave them trainable 
        if isinstance(self._freeze, bool):
            if self._freeze:
                for param in self._feature_extractor.parameters():
                    param.requires_grad = False

            return 
        
        if self._build_by_layer == self._freeze_by_layer:
            # in this case, the freeze and build_by_layer are of the same value
            # we need to freeze the weights of the feature extractor
            self.__freeze_by_building_block(self._freeze)
            return 
        
        # in this case, we know that freeze_by_layer is true and build_by_layer is false (guaranteed by the initialization code)
        self.__freeze_by_bottleneck_build_by_layer(self._freeze)


    def __init__(self, 
                 build_by_layer: bool,
                 num_extracted_layers: int,
                 num_extracted_bottlenecks: int,
                 freeze: Union[bool, int],
                 freeze_by_layer: bool,
                 add_global_average: bool,
                 architecture: int = 50,
                 *args, **kwargs):

        """
        This method initializes the ResNetFeatureExtractor class.

        Args:
            build_by_layer (bool): Whether to build the feature extractor with the "layer" or the "bottleneck" as a building block.
            num_extracted_layers (int): The number of layers to extract.
            num_extracted_block (int): The number of blocks to extract.
            freeze (Union[bool, int]): if boolean: all parameters will be frozen (or left trainable). if integer: all parameters up to the layer/block will be frozen.
            freeze_by_layer (bool): Whether to freeze the weights of the feature extractor by layer or by block.
            add_global_average (bool): Whether to add a global average pooling layer to the feature extractor.
            architecture (int): The architecture of the original ResNet model to use.
        """

        # don't forget that this is a WrapperLikeModuleMixin object and we need specify the name of the attribute that will be wrapped
        # in this case: "_feature_extractor"
        super().__init__("_feature_extractor", *args, **kwargs)

        if not build_by_layer and freeze_by_layer:
            raise ValueError("The feature extractor cannot be built by bottleneck and frozen by layer at the same time. However, it can be build by layers and frozen by bottleneck.")

        self._feature_extractor: nn.Module = None

        self._build_by_layer = build_by_layer

        self._architecture = architecture

        if  build_by_layer and num_extracted_layers == 0:
            raise ValueError("The number of extracted layers cannot be zeor. Negative values indicate all layers should be extracted.")

        if not build_by_layer and num_extracted_bottlenecks == 0:
            raise ValueError("The number of extracted blocks cannot be zeor. Negative values indicate all blocks should be extracted.")

        self._num_extracted_layers = num_extracted_layers if num_extracted_layers > 0 else float('inf') 
        self._num_extracted_bottlenecks = num_extracted_bottlenecks if num_extracted_bottlenecks > 0 else float('inf') 
        
        self._freeze = freeze
        self._freeze_by_layer = freeze_by_layer
        
        self._add_global_average = add_global_average
        
        # load the original network
        constructor, weights = self.get_model(architecture=architecture)
        self.__net = constructor(weights=weights.DEFAULT) 
        self.default_transform = weights.DEFAULT.transforms()

        self.bottleneck_per_layer = defaultdict(lambda: 0)

        # at this point, build the feature extractor
        if self._build_by_layer:
            self.__extract_feature_extractory_by_layer()
        else:
            self.__extract_feature_extractory_by_bottleneck()  

        del(self.__net) # remove the original network as it is no longer needed    

        self._freeze_feature_extractor()


    # the implementations of most methods are overridden by the WrapperLikeModuleMixin class