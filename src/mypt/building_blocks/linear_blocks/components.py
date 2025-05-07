"""
This script contains some basic classes and components need to build flexible and general fully connected blocks: e.g Classification heads.
"""
import torch

from torch import nn
from torch.nn import Module
from abc import ABC, abstractmethod
from typing import Iterable, Iterator, Union, Optional, List, Tuple

from mypt.building_blocks.mixins.residual_mixins import GeneralResidualMixin
from mypt.building_blocks.mixins.custom_module_mixins import CloneableModuleMixin, WrapperLikeModuleMixin


class BasicLinearBlock(WrapperLikeModuleMixin, CloneableModuleMixin):
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
        
        # initialize the WrapperLikeModuleMixin parent
        WrapperLikeModuleMixin.__init__(self, '_block', *args, **kwargs)

        # Store init parameters as read-only properties
        self._in_features = in_features
        self._out_features = out_features
        self._activation = activation
        self._dropout = dropout
        self._is_final = is_final
        self._add_activation = add_activation

        # the order of components mainly follows the guidelines offered in this paper: https://proceedings.mlr.press/v216/kim23a.html
        # guideline 1: dropout vs Relu does not matter. In the original paper, dropout was applied before activation: formula in page 1933 (https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
        # ==> the formula basically applies dropout, then linear layer and then the activation function
        
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

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # although this method is already implemented by the torch.nn.Module class
        # it doesn't hurt to have it explicitly defined 
        return self.forward(x)

    # Read-only properties
    @property
    def in_features(self) -> int:
        return self._in_features
    
    @property
    def out_features(self) -> int:
        return self._out_features
    
    @property
    def activation(self) -> str:
        return self._activation
    
    @property
    def dropout(self) -> Optional[float]:
        return self._dropout
    
    @property
    def is_final(self) -> bool:
        return self._is_final
    
    @property
    def add_activation(self) -> bool:
        return self._add_activation

    @property
    def block(self):
        return self._block

    @block.setter
    def block(self, _: Union[nn.Sequential, nn.Module]):
        raise ValueError("The block attribute is read-only")

    def get_constructor_args(self) -> dict:
        return {
            'in_features': self.in_features,
            'out_features': self.out_features,
            'activation': self.activation,
            'dropout': self.dropout,
            'is_final': self.is_final,
            'add_activation': self.add_activation
        }



class FullyConnectedBlock(WrapperLikeModuleMixin, CloneableModuleMixin):
    # all fully connected blocks should have the 'output' and 'in_features' attributes
    def __init__(self, 
                 output: int,
                 in_features: int,
                 num_layers:int, 
                 activation='relu',
                 dropout: Optional[Union[List[float], float]]=None):

        # make sure works in a specific way
        if not (isinstance(dropout, float) or dropout is None):
            if isinstance(dropout, Iterable) and len(dropout) != num_layers - 1:
                raise ValueError(f"The number of dropouts should be the same as the number of layers minus one")

        else:
            dropout = [dropout for _ in range(num_layers - 1)] # using both a list of None or floats is acceptable

        # this constructor call signals that this class is a wrapper-like module to another module saved in the `_block` field
        WrapperLikeModuleMixin.__init__(self, '_block')

        # Store init parameters as read-only properties
        self._output = output
        self._in_features = in_features
        self._num_layers = num_layers
        self._activation = activation
        self._dropout = dropout

        self._block: nn.Module = None

    @property
    def dropout(self):
        return self._dropout

    @property
    def output(self):
        return self._output

    @property
    def in_features(self):
        return self._in_features


    @property
    def num_layers(self):
        return self._num_layers


    @property
    def activation(self):
        return self._activation 
    
    @property
    def block(self):
        return self._block


    @abstractmethod
    def _build(self):
        pass
    
    

class ResidualFullyConnectedBlock(GeneralResidualMixin, FullyConnectedBlock, CloneableModuleMixin):
    def __init__(self, 
                 output: int,
                 in_features: int,
                 num_layers:int, 
                 activation='relu',
                 dropout: Optional[Union[List[float], float]]=None,
                 force_residual:bool=False,
                 *args, **kwargs):
        
        FullyConnectedBlock.__init__(self,
                         output=output,
                         in_features=in_features,
                         num_layers=num_layers,
                         activation=activation,
                         dropout=dropout, *args, **kwargs)

        self._force_residual = force_residual 
        self._adaptive_layer = None 

        main_stream_field_name = '_block'

        if self.output != self.in_features or self._force_residual:
            residual_stream_field_name = '_adaptive_layer'
        else:
            residual_stream_field_name = None

        GeneralResidualMixin.__init__(self, 
                                      main_stream_field_name=main_stream_field_name, 
                                      residual_stream_field_name=residual_stream_field_name)



    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, debug=False)

    
    def forward(self, x: torch.Tensor, debug:bool=False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return self.residual_forward(x, debug=debug) 
    
    def children(self) -> Iterator[Module]:
        return self.residual_children()
    
    def named_children(self) -> Iterator[Tuple[str, Module]]:
        return self.residual_named_children()   
    
    def modules(self) -> Iterator[Module]:
        return self.residual_modules()
    
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.residual_parameters()
    
    def named_parameters(self) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        return self.residual_named_parameters() 

    def to(self, *args, **kwargs):
        return self.residual_to(*args, **kwargs)    
    
    def train(self, mode: bool = True):
        return self.residual_train(mode)    

    def eval(self):
        return self.residual_eval()
    
    def get_constructor_args(self) -> dict:
        return {
            'output': self.output,
            'in_features': self.in_features,
            'num_layers': self.num_layers,
            'activation': self.activation,
            'dropout': self.dropout,
            'force_residual': self.force_residual
        }
