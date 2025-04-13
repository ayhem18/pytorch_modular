"""
This script contains some basic classes and components need to build flexible and general fully connected blocks: e.g Classification heads.
"""
import torch
from torch import nn
from typing import Iterator, Union, Optional, List, Tuple
from torch.nn import Module
from abc import ABC, abstractmethod


class BasicLinearBlock(torch.nn.Module):
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

        # Store init parameters as read-only properties
        self._in_features = in_features
        self._out_features = out_features
        self._activation = activation
        self._dropout = dropout
        self._is_final = is_final
        self._add_activation = add_activation

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
    def block(self, new_block: Union[nn.Sequential, nn.Module]):
        raise ValueError("The block attribute is read-only")

    def children(self) -> Iterator[Module]:
        return self._block.children()

    def named_children(self) -> Iterator[Module]:
        return self._block.named_children()

    def modules(self) -> Iterator[Module]:
        return self._block.modules()

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self._block.parameters()

    def named_parameters(self) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        return self._block.named_parameters()

    def to(self, *args, **kwargs) -> 'BasicLinearBlock':
        self._block = self._block.to(*args, **kwargs)
        return self 
    
    def train(self, mode: bool = True) -> 'BasicLinearBlock':
        self._block.train(mode)
        return self
    
    def eval(self) -> 'BasicLinearBlock':
        self._block.eval()
        return self


class ModuleListMixin:
    def _verify_instance(self):
        if not hasattr(self, 'block'):
            raise AttributeError(f"the child class is expected to have the attribute 'block'")
        
        if not isinstance(self.block, (torch.nn.ModuleList, nn.Sequential)):
            raise TypeError(f"The SequentialModuleListMixin expects the self.block attribute to be of type {torch.nn.ModuleList} or {nn.Sequential}. Found: {type(self.block)}")

    def module_list_to(self, *args, **kwargs):
        self._verify_instance()

        # call the '.to' method for each Module in the ModuleList
        for i in range(len(self._block)):
            self._block[i] = self._block[i].to(*args, **kwargs)
        
        # always return self
        return self


# class SequentialModuleListMixin(ModuleListMixin):    
#     def module_list_forward(self, x: torch.Tensor) -> torch.Tensor:
#         self._verify_instance()
#         # call each module sequentially
#         for m in self.block:
#             x = m.forward(x)
#         return x


            
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
        self._block: Union[nn.Module, nn.ModuleList, nn.Sequential] = None

    @property
    def output(self):
        return self._output

    # Remove setter to make read-only
    
    @property
    def in_features(self):
        return self._in_features

    # Remove setter to make read-only
    
    @property
    def num_layers(self):
        return self._num_layers

    # Remove setter to make read-only
    
    @property
    def activation(self):
        return self._activation 
    
    @property
    def block(self):
        return self._block



    def children(self) -> Iterator[Module]:
        return self._block.children()

    def named_children(self) -> Iterator[tuple[str, Module]]:
        return self._block.named_children()
    
    def modules(self) -> Iterator[nn.Module]:
        return self._block.modules()


    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self._block.parameters()
    
    def named_parameters(self) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        return self._block.named_parameters()


    def to(self, *args, **kwargs) -> 'ExtendedLinearBlock':                
        self._block = self._block.to(*args, **kwargs)
        return self 

    def eval(self) -> 'ExtendedLinearBlock':
        self._block.eval()
        return self

    def train(self, mode: bool = True) -> 'ExtendedLinearBlock':
        self._block.train(mode)
        return self
    
    @abstractmethod
    def _build_block(self):
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
            dropout = [dropout for _ in range(num_layers - 1)] # using both a list of None or floats is acceptable

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
    """
    A residual linear block is an extended linear block with a residual connection. 
    A residual block must contain at least 2 layers (basic linear blocks) 

    The linear blocks 
    """

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
            dropout = [dropout for _ in range(num_layers)] # using both a list of None or floats is acceptable

        super().__init__(output=output,
                         in_features=in_features,
                         num_layers=num_layers,
                         activation=activation)
        
        self._dropout = dropout
        self._units = units.copy()  # Make a copy to prevent external modification
        self._adaptive_layer = nn.Linear(in_features=in_features,  
                                        out_features=units[-1], # mapping the initial input to the output of the last layer: units[-1]
                                        ) 

        # this function will set the self.block field
        self._build_classifier()

    @property
    def units(self):
        return self._units.copy()  # Return a copy to prevent modification
    
    @property
    def dropout(self):
        return self._dropout
    
    @property
    def adaptive_layer(self):
        return self._adaptive_layer

    def _build_classifier(self):
        # implementation using ModuleList
        
        # each of the blocks below will contain [a batch norm layer, a dropout layer, a linear layer, an activation layer]
        blocks = [BasicLinearBlock(
                            in_features=self._units[i], 
                            out_features=self._units[i + 1], 
                            activation=self._activation,
                            dropout=self._dropout[i],
                            is_final=False)
                    for i in range(0, len(self._units) - 2)
                ]
        
        # this last block will contain [a batch norm, dropout layer, linear layer (and no activation)]
        blocks.append(BasicLinearBlock(
                            in_features=self._units[-2], 
                            out_features=self._units[-1], 
                            activation=self._activation,
                            dropout=self._dropout[-1],
                            is_final=False,
                            add_activation=False
                            )
                    )
        # having 'x' as the input, the forward flow will be x through [block1, block2, ... block N]. the output will be denoted as 'y'
        # the residual connection will be y + adaptive_layer.forward(x)

        # add the final activation function seperately
        blocks.append(BasicLinearBlock._ACTIVATION_MAP[self._activation])

        self._block = nn.ModuleList(blocks)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # make a copy of the input
        y = x
        for layer in self._block[:-1]:
            x = layer.forward(x)

        # the residual connection right before the last activation layer
        return self._block[-1].forward(x + self._adaptive_layer.forward(y))

    # Override parent methods to include adaptive_layer
    def children(self) -> Iterator[nn.Module]:
        # Include both block children and adaptive_layer
        yield from self._block.children()
        yield self._adaptive_layer

    def named_children(self) -> Iterator[tuple[str, Module]]:
        # Include both named children from block and adaptive_layer
        yield from self._block.named_children()
        yield ('adaptive_layer', self._adaptive_layer)
    
    def modules(self) -> Iterator[nn.Module]:
        # Include both block modules and adaptive_layer
        yield self
        yield from self._block.modules()
        yield self._adaptive_layer

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        # Include parameters from both block and adaptive_layer
        yield from self._block.parameters()
        yield from self._adaptive_layer.parameters()
    
    def named_parameters(self) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        # Include named parameters from both block and adaptive_layer
        block_params = {f"block.{name}": param for name, param in self._block.named_parameters()}
        adaptive_params = {f"adaptive_layer.{name}": param for name, param in self._adaptive_layer.named_parameters()}
        
        for name, param in block_params.items():
            yield (name, param)
        
        for name, param in adaptive_params.items():
            yield (name, param)

    def to(self, *args, **kwargs):
        # Apply to() to both block and adaptive_layer
        self._block = self._block.to(*args, **kwargs)
        self._adaptive_layer = self._adaptive_layer.to(*args, **kwargs)
        return self

    def eval(self) -> 'ResidualLinearBlock':
        # Set eval mode for both block and adaptive_layer
        self._block.eval()
        self._adaptive_layer.eval()
        return self

    def train(self, mode: bool = True) -> 'ResidualLinearBlock':
        # Set train mode for both block and adaptive_layer
        self._block.train(mode)
        self._adaptive_layer.train(mode)
        return self


