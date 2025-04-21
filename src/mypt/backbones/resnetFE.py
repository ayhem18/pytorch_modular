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
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from torchvision.models.resnet import Bottleneck, BasicBlock


from mypt.building_blocks.mixins.custom_module_mixins import WrapperLikeModuleMixin  


class ResnetFE(WrapperLikeModuleMixin):
    __archs__ = [50, 101, 152]
    
    # archs_dict = {18: (resnet18, ResNet18_Weights), # the resnet18 model does not use the bottleck block apparently 
    # archs_dict = {34: (resnet34, ResNet34_Weights), 
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

        # at this point, the module has children
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

        extended_modules = list(extracted_modules)
        self._feature_extractor = nn.Sequential(OrderedDict(extended_modules))
            


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

            if isinstance(module, nn.AdaptiveAvgPool2d) and self._add_global_average:
                extracted_modules.append((module_name, module))
                continue
                
            if isinstance(module, (nn.Linear, nn.LazyLinear)):
                # ignore fully connected layers
                continue
            
            mc = list(module.named_children())

            if len(mc) == 0:
                # if the module has no children, then it is a simply layer and must be one of the layers before the first layer block
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
                    # we use break since, within the layer block, we will not add any more residual blocks to the feature extractor.
                    break

                if self.RESIDUAL_BLOCK.lower() not in name.lower():
                    name = self.RESIDUAL_BLOCK.lower() + '_' + f'{bottleneck_counter + 1}'
                
                # add the residual block to the feature extractor  
                extracted_modules.append((name, child))
                bottleneck_counter += 1    

                self.bottleneck_per_layer[layer_counter] += 1

        extended_modules = list(extracted_modules)
        self._feature_extractor = nn.Sequential(OrderedDict(extended_modules))

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
    

# if __name__ == "__main__":
    res = resnet101(weights=ResNet101_Weights.DEFAULT)
    
#     # for name, module in res.named_modules():
#     #     print(name, module, sep=' : ')
#     fe = ResnetFE(build_by_layer=False, num_extracted_bottlenecks=10, num_extracted_layers=0, architecture=101, freeze=False, freeze_by_layer=False, add_global_average=True) 
#     print(fe)    