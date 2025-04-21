"""
This script contains functionalities to build classifiers on top of the pretrained model
'RESNET 50' provided by pytorch module

This script is mainly inspired by this paper: 'https://arxiv.org/abs/1411.1792'
as they suggest an experimental framework to find the most transferable / general layers
in pretrained network. I am applying the same framework on the resnet architecture.
"""

import torch
import warnings


from collections import OrderedDict
from typing import Iterator, Union, Tuple, Any, Optional

from torch import nn
from torch.nn.modules.module import Module
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights  
from torchvision.models.resnet import Bottleneck



# let's create a utility function real quick
def contains_fc_layer(module: nn.Module) -> bool:
    """
    This function returns whether the module contains a Fully connected layer or not
    """
    m = isinstance(module, nn.Linear)
    sub_m = any([isinstance(m, nn.Linear) for m in module.modules()])
    return m and sub_m



class ResNetFeatureExtractor(nn.Module):
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
            warnings.warn(f'The value {architecture} was passed as architecture. Defaulting to {50}')
            architecture = 50

        return cls.archs_dict[architecture]

    def __feature_extractor_layers(self, number_of_layers: int):
        # passing a negative value would mean retrieving the entire feature extractor
        number_of_layers = number_of_layers if number_of_layers > 0 else float('inf')

        modules_generator = self.__net.named_children()
        # the modules will be saved with their original names in an OrderedDict and later merged
        # into a single nn.Module using nn.Sequential
        modules_to_keep = []
        counter = 0

        for name, module in modules_generator:
            # we will only consider non fully connected components of the network
            if (not contains_fc_layer(module)) and (self.add_gb_avg or module != self.__net.avgpool):
                if 'layer' not in name or counter < number_of_layers:
                    modules_to_keep.append((name, module))
                    # only increment counter if the name contains 'layer': meaning it is a layer block
                    counter += int('layer' in name)

        fe = nn.Sequential(OrderedDict(modules_to_keep))
        return fe

    def _freeze(self, freeze: Union[bool, float, int], freeze_layers: bool = True) -> None:
        freeze = True if freeze is None else freeze

        # if freeze is passed as boolean variable, it does not matter which blocks we are considering, layers or residual blocks
        if isinstance(freeze, bool) :
            if freeze:
                for p in self.feature_extractor.parameters():
                    p.requires_grad = False    
            return

        # make sure to convert the 'freeze' argument to an integer
        freeze = int(freeze)

        if freeze_layers:
            counter = 0
            for name, module in self.feature_extractor.named_children():
                if counter < freeze:
                    for p in module.parameters():
                        p.requires_grad = False
                    # only increment counter if the name contains 'layer': meaning it is a layer block
                    counter += int('layer' in name)
            return 

        counter = 0
        # the first loop will iterate through the layer blocks
        for name, module in self.feature_extractor.named_children():
            # the 2nd loop will iterate through the Residual blocks within each layer block  
            for _, sub_module in module.named_children():
                if counter < freeze: 
                    for p in sub_module.parameters():
                        p.requires_grad = False
                    counter += int(isinstance(sub_module, Bottleneck))
            
            # keep in mind that certain modules might not have children components and still need to be frozen
            if len(list(module.named_children())) == 0 and counter < freeze:
                for p in module.parameters():
                    p.requires_grad = False

    def __init__(self,
                 num_layers: int,  # the number of blocks to keep
                 add_global_average: bool = True, # whether to add the 
                 freeze_layers: bool = True, # whether to freeze by layer or by block
                 freeze: Optional[Union[bool, int]] = True,  # whether to freeze the chosen layers or not
                 architecture: int = 50,
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers

        constructor, weights = self.get_model(architecture=architecture)
        self.__net = constructor(weights=weights.DEFAULT) 
        self.transform = weights.DEFAULT.transforms()

        self.add_gb_avg = add_global_average
        self.feature_extractor = self.__feature_extractor_layers(self.num_layers)

        # freeze the weights if needed
        self._freeze(freeze=freeze, freeze_layers=freeze_layers)

        # remove the self.__net field from the model
        del(self.__net)


    def forward(self, x: torch.Tensor):
        # the forward function in the ResNet class simply calls the forward function
        # of each submodule consecutively: which is equivalent to saving all modules in a nn.Sequential module
        # and calling the forward method.
        return self.feature_extractor.forward(x)

    def __str__(self):
        # the default __str__ function will display the self.__net module as well
        # which might be confusing as .__net is definitely not part of the forward pass of the model
        return self.feature_extractor.__str__()
    
    def __repr__(self):
        return self.feature_extractor.__repr__() 
    
    def children(self) -> Iterator['Module']:
        # not overloading this method will return to an iterator with 2 elements: self.__net and self.feature_extractor
        return self.feature_extractor.children()

    def modules(self) -> Iterator[nn.Module]:
        return self.feature_extractor.modules()
    
    def named_children(self) -> Iterator[Tuple[str, Module]]:
        return self.feature_extractor.named_children()

    def to(self, *args, **kwargs) -> 'ResNetFeatureExtractor':
        self.feature_extractor = self.feature_extractor.to(*args, **kwargs)
        return self 
