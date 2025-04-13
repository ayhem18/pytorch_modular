"""
This script contains some basic classes and components need to build flexible and general fully connected blocks: e.g Classification heads.
"""
import torch
from torch import nn
from typing import Iterator, Union, Optional, List, Tuple
from torch.nn import Module
from abc import ABC, abstractmethod


class LinearBlock(torch.nn.Module):
    _RELU = 'relu'
    _LEAKY_RELU = 'leaky_relu'
    _TANH = 'tanh'
    _ACTIVATIONS = [_RELU, _LEAKY_RELU, _TANH]

    _ACTIVATION_MAP = { # since activatin layers do not have weights, it is not an issue to have actual instances instead of classes in this mapping
                       _RELU: nn.ReLU(inplace=True), 
                       _TANH: nn.Tanh(),
                       _LEAKY_RELU: nn.LeakyReLU(inplace=True)
                       }

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation: str = _LEAKY_RELU,
                 dropout: Optional[float] = None,
                 is_final: bool = False,
                 add_activation: bool = True,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # the order of components mainly follows the guidelines offered in this paper: https://proceedings.mlr.press/v216/kim23a.html
        # guideline 1: dropout vs Relu does not matter. In the original paper, dropout was applied before activation: formula in page 1933 (https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
        # ==> the formula basically applies dropout, then linear layer and then the activatio function
        # guideline 2: dropout after norm layer
        
        # the final order using both guidelines would be: BatchNorm, dropout, linearlayer, relu
        # This class was designed to follow a convolutional block which justifies using BatchNormalization as the first layer

        linear_layer = nn.Linear(in_features=in_features, out_features=out_features)

        if not is_final:
            norm_layer = nn.BatchNorm1d(num_features=in_features)
            activation_layer = self._ACTIVATION_MAP[activation]

            components = [norm_layer] 
            if dropout is not None:
                components.append(nn.Dropout(p=dropout))
            
            # add the linear layer
            components.append(linear_layer)

            # add activation
            if add_activation:
                components.append(activation_layer)
        else:
            components = [linear_layer]

        self._block = nn.Sequential(*components)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._block.forward(x)

    def __str__(self) -> str:
        return self._block.__str__()

    @property
    def block(self):
        return self._block

    @block.setter
    def block(self, new_block: Union[nn.Sequential, nn.Module]):
        # make sure the new block is either 
        if not isinstance(new_block, (nn.Sequential, nn.Module)):
            raise TypeError((f"The block is expected to be either of type {nn.Module} or {nn.Sequential}\n"
                             f"Found: {type(new_block)}"))

    def children(self) -> Iterator[Module]:
        return self.block.children()

    def named_children(self) -> Iterator[Module]:
        return self.block.children()

    def modules(self) -> Iterator[Module]:
        return self.block.modules()


class ExtendedLinearBlock(ABC, nn.Module):
    def __init__(self, 
                 output:int,
                 in_features: int, 
                 num_layers:int, 
                 activation='relu'):
        
        # call nn.Module constructor before setting the fields 
        super().__init__()

        self._output = output if output > 2 else 1
        self._in_features = in_features
        self._activation = activation
        self._num_layers = num_layers

        # the actual model that does the heavy lifting
        self.classifier: Union[nn.Module, nn.ModuleList, nn.Sequential] = None

    @property
    def output(self):
        return self._output

    # a setter for output and in_features
    @output.setter
    def output(self, x: int):
        self._output = x if x > 2 else 1
        self._build_classifier()


    @property
    def in_features(self):
        return self._in_features

    @in_features.setter
    def in_features(self, x: int):
        self._in_features = x
        self._build_classifier()

    @property
    def activation(self):
        return self._activation

    @property
    def num_layers(self):
        return self._num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier.forward(x)

    def children(self) -> Iterator[nn.Module]:
        return self.classifier.children()

    def named_children(self) -> Iterator[tuple[str, Module]]:
        return self.classifier.named_children()

    def modules(self) -> Iterator[nn.Module]:
        return self.classifier.modules()

    def to(self, *args, **kwargs) -> 'ExtendedLinearBlock':                
        # from ..code_utilities import pytorch_utilities as pu 
        # this line is left for debugging purposes...
        self.classifier = self.classifier.to(*args, **kwargs)  # apply the changes to the classifier (don't forget the assignment operator)
        return self 

    @abstractmethod
    def _build_classifier(self):
        # this function represents the main design of the classification head
        pass


class FullyConnectedBlock(ExtendedLinearBlock):
    # all classifiers should have the 'output' and 'in_features' attributes
    def __init__(self, 
                 output: int,
                 in_features: int,
                 num_layers:int, 
                 activation='relu',
                 dropout: Optional[Union[List[float], float]]=None):

        # make sure works in a specific way
        if not (isinstance(dropout, float) or dropout is None):
            if isinstance(dropout, (List, Tuple)) and len(dropout) != num_layers - 1:
                raise ValueError(f"The number of dropouts should be the same as the number of layers minus one")

        else:
            dropout = [dropout for _ in range(num_layers - 1)] # using both a list of Nones or floats is acceptable

        super().__init__(output=output, 
                         in_features=in_features,
                         num_layers=num_layers,
                         activation=activation)
        
        self._dropout = dropout

    @property
    def dropout(self):
        return self._dropout
    
    @abstractmethod
    def _build_classifier(self):
        pass
    

class ResidualLinearBlock(ExtendedLinearBlock):
    def __init__(self, 
                 output: int,
                 in_features: int,
                 num_layers:int,
                 units:List[int], 
                 activation='relu',
                 dropout: Optional[Union[List[float], float]]=None):

        if num_layers < 2:
            raise ValueError(f"A residual Fc block must contain at least 2 layers.")

        if len(units) != num_layers + 1:
            raise ValueError(f"The number of passed units must be equal to the number of layers + 1")

        if units[0] != in_features or units[-1] != output:
            raise ValueError(f"Make sure the first number in units is the same as 'in_features' and the last number in units is the same as 'output'")        

        if not (isinstance(dropout, float) or dropout is None):
            if isinstance(dropout, (List, Tuple)) and len(dropout) != num_layers:
                raise ValueError(f"The number of dropouts should be the same as the number of layers")

        else:
            dropout = [dropout for _ in range(num_layers)] # using both a list of Nones or floats is acceptable

        super().__init__(output=output,
                         in_features=in_features,
                         num_layers=num_layers,
                         activation=activation)
        
        self._dropout = dropout
        self.units = units
        self.adaptive_layer = nn.Linear(in_features=in_features,  
                                        out_features=units[-1], # mapping the initial input to the output of the last layer: units[-1]
                                        ) 

        # this function will set the self.classifier field
        self._build_classifier()


    def _build_classifier(self):
        # implementation using ModuleList
        blocks = [LinearBlock(
                            in_features=self.units[i], 
                            out_features=self.units[i + 1], 
                            activation=self.activation,
                            dropout=self.dropout[i],
                            is_final=False)
                    for i in range(0, len(self.units) - 2)
                ]
        
        # append another block without the activation function
        blocks.append(LinearBlock(
                            in_features=self.units[-2], 
                            out_features=self.units[-1], 
                            activation=self.activation,
                            dropout=self.dropout[-1],
                            is_final=False,
                            add_activation=False)
                    )

        # add the final activation function seperately
        blocks.append(LinearBlock._ACTIVATION_MAP[self.activation])

        self.classifier = nn.ModuleList(blocks)

    @property
    def dropout(self):
        return self._dropout


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        from ..code_utilities import pytorch_utilities as pu
        # make a copy of the input
        y = x
        for layer in self.classifier[:-1]:
            x = layer.forward(x)

        # the residual connection right before the last activation layer
        return self.classifier[-1].forward(x + self.adaptive_layer.forward(y))


    def to(self, *args, **kwargs):
        # apply the 'to' method both to the classifier and the adaptive layer
        # make sure to assign the output of 'to' to the self fields
        self.classifier = self.classifier.to(*args, **kwargs)
        self.adaptive_layer = self.adaptive_layer.to(*args, **kwargs)
        return self


class ModuleListMixin:
    def _verify_instance(self):
        if not hasattr(self, 'classifier'):
            raise AttributeError(f"the child class is expected to have the attribute 'classifier'")
        
        if not isinstance(self.classifier, torch.nn.ModuleList):
            raise TypeError(f"The SequentialModuleListMixin expects the self.classifier attribute to be of type {torch.nn.ModuleList}. Found: {type(self.classifier)}")

    def module_list_to(self, *args, **kwargs):
        self._verify_instance()

        # call the '.to' method for each Module in the ModuleList
        for i in range(len(self.classifier)):
            self.classifier[i] = self.classifier[i].to(*args, **kwargs)
        
        # always return self
        return self


class SequentialModuleListMixin(ModuleListMixin):    
    def module_list_forward(self, x: torch.Tensor) -> torch.Tensor:
        self._verify_instance()
        # call each module sequentially
        for m in self.classifier:
            x = m.forward(x)
        return x
