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

    _ACTIVATION_MAP = {_RELU: nn.ReLU(inplace=True),
                       _TANH: nn.Tanh(),
                       _LEAKY_RELU: nn.LeakyReLU(inplace=True)}

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation: str = _LEAKY_RELU,
                 dropout: Optional[float] = None,
                 is_final: bool = True,
                 add_activation: bool = True,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # the idea here is quite simple. 
        components = [nn.Linear(in_features=in_features, out_features=out_features)]
        # depending on the value of 'is_final' 
        if not is_final:
            norm_layer = nn.BatchNorm1d(num_features=out_features)
            activation_layer = self._ACTIVATION_MAP[activation]
            
            if dropout is not None:
                components.extend([norm_layer, activation_layer, nn.Dropout(p=dropout)])
            else:
                components.extend([norm_layer, activation_layer])            

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


class FullyConnectedBlock(ABC, nn.Module):
    # all classifiers should have the 'output' and 'in_features' attributes
    def __init__(self, 
                 output: int,
                 in_features: int,
                 num_layers:int, 
                 activation='relu',
                 dropout: Optional[Union[List[float], float]]=None):

        # make sure dropout works 
        if not (isinstance(dropout, float) or dropout is None):
            if isinstance(dropout, (List, Tuple)) and len(dropout) != num_layers:
                raise ValueError(f"The number of dropouts should be the same as the number of layers")

        else:
            dropout = [dropout for _ in range(num_layers - 1)] # using both a list of Nones or floats is acceptable

        super().__init__()
        # take into account the case of binary-classification
        self._output = output if output > 2 else 1
        self._in_features = in_features
        self._activation = activation
        self._num_layers = num_layers
        self._dropout = dropout

        # the actual model that does the heavy lifting
        self.classifier: nn.Module = None

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
    def dropout(self):
        return self._dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier.forward(x)

    def children(self) -> Iterator[nn.Module]:
        return self.classifier.children()

    def named_children(self) -> Iterator[tuple[str, Module]]:
        return self.classifier.named_children()

    def modules(self) -> Iterator[nn.Module]:
        return self.classifier.modules()

    @abstractmethod
    def _build_classifier(self):
        # this function represents the main design of the classification head
        pass


class ResidualFullyConnectedBlock(FullyConnectedBlock):
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
        
        super().__init__(output, 
                         in_features, 
                         num_layers, 
                         activation, 
                         dropout)

        self.units = units
        self.adaptive_layer = nn.Linear(in_features=in_features, # mapping the initial input to the input of the last layer: which is units[-2] (not units[-1]) 
                                        out_features=units[-2]) 

    def _build_classifier(self):
        blocks = [LinearBlock(in_features=self.units[i], 
                              out_features=self.units[i + 1], 
                              activation=self.activation,
                              dropout=self.dropout[i],
                              is_final=False
                              )
                for i in range(len(self.units) - 2)
                ]

        # append another block
        blocks.append(LinearBlock(
                            in_features=self.units[-2], 
                            out_features=self.units[-1], 
                            activation=self.activation,
                            dropout=self.dropout[-1],
                            is_final=False,
                            add_activation=False)
                    )



    def forward(self, x:torch.Tensor) -> torch.Tensor:
        pass
