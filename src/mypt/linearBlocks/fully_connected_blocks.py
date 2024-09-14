"""
This script contains different general classes of classification heads with different design choices: either my own
design or inspired from other sources.
"""

import math, torch
import numpy as np

from torch import nn
from typing import Union, List, Optional
from collections import OrderedDict

from .fc_block_components import FullyConnectedBlock, LinearBlock, ResidualFullyConnectedBlock


class GenericBuildMixin:   
    """
    GenericBuildMixin, given a self.units field, builds a nn.SequentialModel.
    The number of layers (n) is assumed to be self.units - 1, the first n - 1 are non-final linear blocks and the last one is final
    """
    def _build_block(self) -> torch.nn.Sequential:

        attrs = ['units', 'activation', 'dropout']
        for att in attrs:
            if not hasattr(self, att):
                raise AttributeError(f"The ResidualMixin expects the child class to include the following attributes: {attrs}. Found {att} missing")            

        if len(self.units) < 2:
            raise ValueError(f"the number of elements in the self.units must be at least 2")

        blocks = [LinearBlock(in_features=self.units[i],
                                out_features=self.units[i + 1],
                                is_final=False,
                                activation=self.activation,
                                dropout=self.dropout[i]
                            ) 
                    for i in range(len(self.units) - 2)
                ]
        
        # add the last layer by setting the 'is_final' argument to True
        blocks.append(LinearBlock(in_features=self.units[-2],
                                    out_features=self.units[-1],
                                    is_final=True)
                    )

        # convert the list of blocks into an OrderedDict so the different inner blocks can be accessed
        blocks = OrderedDict([(f'fc_{i}', b) for i, b in enumerate(blocks, start=1)])

        return torch.nn.Sequential(blocks)


class GenericFCBlock(
            GenericBuildMixin,
            FullyConnectedBlock
            ):
    def __init__(self, 
                 output: int,
                 in_features: int,
                 num_layers:int, 
                 units: List[int],
                 activation='relu',
                 dropout: Optional[Union[List[float], float]]=None
                 ):
        
        if len(units) != num_layers + 1:
            raise ValueError(f"The Generic Fully Connected block expects the 'units' argument to contain {num_layers + 1} elements. Found: {len(units)}")

        super().__init__(output=output,
                         in_features=in_features,
                         num_layers=num_layers,
                         activation=activation,
                         dropout=dropout)

        self.units = units

        self._build_classifier()

    def _build_classifier(self):
        # the idea is quite simple
        self.classifier = self._build_block()


# this mixin, given the number of input and output features sets the number of units inside in an exponential scale
class ExponentialMixin:
    def _set_units(self) -> List[int]:
        if not (hasattr(self, 'num_layers') 
                and hasattr(self, 'output') 
                and hasattr(self, 'in_features')):
            
            raise AttributeError(f"The child class extending the 'ExponentialMinin' class must have the following attributes: {['num_layers''output', 'in_features']}")
            
        # let's start with making sure the child class has the necessary attribute
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
                num_units = [int(2 ** p) for p in powers]
                # set the last element to the actual output
                num_units = [self.in_features] + num_units

            num_units.append(self.output)

        else:
            num_units = [self.in_features, self.output]
        
        return num_units


class ExponentialFCBlock(
                        GenericBuildMixin,
                        ExponentialMixin, 
                        FullyConnectedBlock):
    def __init__(self,
                 output: int,
                 in_features: int,
                 num_layers: int,
                 dropout: Optional[Union[List[float], float]]=None,
                 activation: str = 'relu'):
        
        # the usual parent's class call
        super().__init__(output=output, 
                         in_features=in_features,
                         num_layers=num_layers,
                         activation=activation,
                         dropout=dropout
                         )
                
        max_dim, min_dim = max(self.output, in_features), min(self.output, in_features)

        if 2 ** (int(np.log2(min_dim)) + 1) > max_dim and num_layers >= 2:
            raise ValueError(f"Please make sure the difference is large enough between the number of classes and 'in_features' argument\n"
                             f"smallest dimension: {min_dim}. Largest dimension: {max_dim}")
        
        self._build_classifier()


    def _build_classifier(self):
        # get the number of units using the ExponentialMixin 
        self.units = self._set_units()
        self.classifier = self._build_block()


class ResidualMixin:
    def _build_classifier_blocks(self) -> List[torch.nn.Module]:
        attrs = ['layers_per_residual_block', 'units', 'activation', 'dropout']
        for att in attrs:
            if not hasattr(self, att):
                raise AttributeError(f"The ResidualMixin expects the child class to include the following attributes: {attrs}. Found {att} missing")            

        if self.layers_per_residual_block <= 1:
            raise ValueError(f"Residual blocks must contain at least 2 layers. Found: {self.layers_per_residual_block} layer(s) per residual block")

        num_layers = len(self.units) - 1

        if num_layers % self.layers_per_residual_block == 0:
            residual_range = num_layers - self.layers_per_residual_block
        else:
            residual_range = (num_layers // self.layers_per_residual_block) * self.layers_per_residual_block

        blocks = [
            ResidualFullyConnectedBlock(output=self.units[i + self.layers_per_residual_block],
                                        in_features=self.units[i],
                                        units=self.units[i: i + self.layers_per_residual_block + 1],
                                        num_layer=self.layers_per_residual_block,
                                        dropout=self.dropout[i: i + self.layers_per_residual_block],
                                        activation=self.activation
                                        )
            for i in range(0, residual_range, self.layers_per_residual_block)
        ]

        # add the last block as a standard Fully connected block
        blocks.append(GenericFCBlock(in_features=self.units[residual_range + 1], 
                                    output=self.units[-1], 
                                    num_layers=num_layers - residual_range, 
                                    units=self.units[residual_range + 1:],
                                    dropout=self.dropout[residual_range + 1:],
                                    activation=self.activation,
                                    )
                    )


class ExponentialResidualFCBlock(
                        ResidualMixin, 
                        ExponentialMixin, 
                        FullyConnectedBlock):
    def __init__(self,
                 output: int,
                 in_features: int,
                 num_layers: int,
                 layers_per_residual_block:int,
                 dropout: Optional[Union[List[float], float]]=None,
                 activation: str = 'relu'):
        
        # the usual parent's class call
        super().__init__(output=output, 
                         in_features=in_features,
                         num_layers=num_layers,
                         activation=activation,
                         dropout=dropout
                         )
                
        max_dim, min_dim = max(self.output, in_features), min(self.output, in_features)

        if 2 ** (int(np.log2(min_dim)) + 1) > max_dim and num_layers >= 2:
            raise ValueError(f"Please make sure the difference is large enough between the number of classes and 'in_features' argument\n"
                             f"smallest dimension: {min_dim}. Largest dimension: {max_dim}")
        
        self.layers_per_residual_block = layers_per_residual_block
        self._build_classifier()


    def _build_classifier(self):
        # get the number of units using the ExponentialMixin 
        self.units = self._set_units()
        self.classifier = self._build_classifier_blocks()

