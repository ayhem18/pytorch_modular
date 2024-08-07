"""
This script contains different general classes of classification heads with different design choices: either my own
design or inspired from other sources.
"""

import torch, math
import numpy as np

from torch import nn
from typing import Sequence, Iterator, Union, List, Optional
from torch.nn import Module
from abc import ABC, abstractmethod
from collections import OrderedDict


class LinearBlock(nn.Module):
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
        if isinstance(new_block, (nn.Sequential, nn.Module)):
            raise TypeError((f"The block is expected to be either of type {nn.Module} or {nn.Sequential}\n"
                             f"Found: {type(new_block)}"))

    def children(self) -> Iterator[Module]:
        return self.block.children()

    def named_children(self) -> Iterator[Module]:
        return self.block.children()

    def modules(self) -> Iterator[Module]:
        return self.block.modules()

class ClassificationHead(ABC, nn.Module):
    # all classifiers should have the 'num_classes' and 'in_features' attributes
    def __init__(self, 
                 num_classes: int,
                 in_features: int,
                 activation='leaky_relu'):
        super().__init__()
        # take into account the case of binary-classification
        self._num_classes = num_classes if num_classes > 2 else 1
        self._in_features = in_features
        self._activation = activation
        # the actual model that does the heavy lifting
        self.classifier = None

    @property
    def num_classes(self):
        return self._num_classes

    # a setter for num_classes and in_features
    @num_classes.setter
    def num_classes(self, x: int):
        self._num_classes = x if x > 2 else 1
        self._build_classifier()

    @property
    def in_features(self):
        return self._in_features

    @in_features.setter
    def in_features(self, x: int):
        self._in_features = x
        self._build_classifier()

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


class ExponentialClassifier(ClassificationHead):
    def __init__(self,
                 num_classes: int,
                 in_features: int,
                 num_layers: int,
                 dropout: Optional[Union[List[float], float]]=None,
                 activation: str = 'relu'):
        

        # the usual parent's class call
        super().__init__(num_classes, in_features, activation)
        self.num_layers = num_layers
        # self.num_layers = max(num_layers, 2)
        self.output = 1 if num_classes == 2 else num_classes
        
        max_dim, min_dim = max(self.output, in_features), min(self.output, in_features)

        if 2 ** (int(np.log2(min_dim)) + 1) > max_dim and num_layers >= 2:
            raise ValueError(f"Please make sure the difference is large enough between the number of classes and 'in_features' argument\n"
                             f"smallest dimension: {min_dim}. Largest dimension: {max_dim}")


        if not (isinstance(dropout, float) or dropout is None):
            if isinstance(dropout, Sequence) and len(dropout) != num_layers:
                raise ValueError(f"The number of dropouts should be the same as the number of layers")

        else:
            dropout = [dropout for _ in range(self.num_layers - 1)]
        
        self.dropouts = dropout
        self._build_classifier()


    def _build_classifier(self):
        if self.num_layers > 1:            
            # the log of the in_features
            log_input = np.log2(self.in_features)
            base_power = int(log_input)
            
            if math.ceil(log_input) == log_input:
                # this mean log_input is an integer, and in_features is a power of 2
                powers = np.linspace(start=int(np.log2(self.output)) + 1, stop=base_power, num=self.num_layers)
                # make sure to convert to integers
                num_units = [int(2 ** p) for p in powers][::-1]

            else:    
                powers = np.linspace(start=base_power, stop=int(np.log2(self.output)) + 1, num=self.num_layers - 1)
                # # first we build n - 1 layers starting from the highest power of 2 less than in_features
                # powers = np.linspace(start=int(np.log2(self.output)) + 1 , stop=base_power, num=self.num_layers - 1)
                # make sure to convert to integers
                num_units = [int(2 ** p) for p in powers]
                # set the last element to the actual output
                num_units = [self.in_features] + num_units

            num_units.append(self.output)

            blocks = [LinearBlock(in_features=num_units[i],
                                    out_features=num_units[i + 1],
                                    is_final=False,
                                    activation=self._activation,
                                    dropout=self.dropouts[i]) for i in range(len(num_units) - 2)]
        else: 
            blocks = []
            num_units = [self.in_features, self.output]

        # add the last layer by setting the 'is_final' argument to True
        blocks.append(LinearBlock(in_features=num_units[-2],
                                    out_features=num_units[-1],
                                    is_final=True))

        # convert the list of blocks into an OrderedDict so the different inner blocks can be accessed
        blocks = OrderedDict([(f'fc_{i}', b) for i, b in enumerate(blocks, start=1)])

        self.classifier = nn.Sequential(blocks)
        