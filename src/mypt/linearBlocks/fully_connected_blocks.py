"""
This script contains different general classes of classification heads with different design choices: either my own
design or inspired from other sources.
"""

import torch
import numpy as np

from torch import nn
from typing import Sequence, Union, List, Optional
from collections import OrderedDict

from .mixin_blocks import ExponentialMixin
from .fc_block_components import FullyConnectedBlock, LinearBlock

class ExponentialFCBlock(ExponentialMixin, # first time using Mixins ... (dope !!) 
                            FullyConnectedBlock):
    def __init__(self,
                 output: int,
                 in_features: int,
                 num_layers: int,
                 dropout: Optional[Union[List[float], float]]=None,
                 activation: str = 'relu'):
        

        # the usual parent's class call
        super().__init__(output, 
                         in_features,
                         num_layers, 
                         activation)
                
        max_dim, min_dim = max(self._output, in_features), min(self._output, in_features)

        if 2 ** (int(np.log2(min_dim)) + 1) > max_dim and num_layers >= 2:
            raise ValueError(f"Please make sure the difference is large enough between the number of classes and 'in_features' argument\n"
                             f"smallest dimension: {min_dim}. Largest dimension: {max_dim}")
        
        self._build_classifier()


    def _build_classifier(self):
        # get the number of units using the ExponentialMixin 
        num_units = self._set_units()

        assert len(num_units) == self._num_layers + 1, "The number of units must be equal to number of layers + 1"

        # save the non final linear blocks first (th ones with activation, bn and dropout)
        blocks = [LinearBlock(in_features=num_units[i],
                                out_features=num_units[i + 1],
                                is_final=False,
                                activation=self._activation,
                                dropout=self.dropouts[i]) for i in range(len(num_units) - 2)]
        
        # add the last layer by setting the 'is_final' argument to True
        blocks.append(LinearBlock(in_features=num_units[-2],
                                    out_features=num_units[-1],
                                    is_final=True))

        # convert the list of blocks into an OrderedDict so the different inner blocks can be accessed
        blocks = OrderedDict([(f'fc_{i}', b) for i, b in enumerate(blocks, start=1)])

        self.classifier = nn.Sequential(blocks)
        

class ResidualMixin:
    # this Mixin assumes the child class has the following 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (hasattr(self, 'layers_per_residual_block') 
                and hasattr(self, 'classifier')
                and hasattr(self, 'adative_layers')
                ):
            raise ValueError(f"The ResidualMixin class expects the child class to contain the following attributes: self.layers_per_residual_block")

        block_count = 0
        adaptive_layer_count = 0

        y = x # 

        for block in self.children():
            x = block.forward(x)
            block_count += 1
            
            if block_count == self.layers_per_residual_block:
                block_count = 0
                x = x + self.adaptive_layers[adaptive_layer_count].forward(y)
                adaptive_layer_count += 1
        
        return x
    
    def __build_adaptive_layers(self):
        if not (hasattr(self, 'layers_per_residual_block') 
                and hasattr(self, 'classifier')
                ):
            raise ValueError(f"The ResidualMixin class expects the child class to contain the following attributes: self.layers_per_residual_block")

        self.adaptive_layers = []

        in_feats = block.in_features
        out_feats = None

        counter = 0
        for block in self.children():
            counter += 1


class ResidualExponentialClassifier(ResidualMixin, ExponentialClassifier):
    def __init__(self,
                 num_classes: int,
                 in_features: int,
                 num_layers: int,
                 layers_per_residual_block:int,
                 dropout: Optional[Union[List[float], float]]=None,
                 activation: str = 'relu'):

        if layers_per_residual_block > num_layers:
            raise ValueError(f"The total number of layers must be larger than the number of layers per residual block")

        # the super class initialization
        super().__init__(num_classes=num_classes,
                        in_features=in_features,
                        num_layers=num_layers,
                        dropout=dropout,
                        activation=activation)

        # make sure to set the fields necessary for the ResidualMixin
        self.layers_per_residual_block = layers_per_residual_block

