"""
This script contains the differnt schemas / ideas to determine the number of units in the intermediate layers of a fully connected block / classification head
"""

import math
import numpy as np

from typing import List

from .fc_block_components import LinearBlock

class ResidualMixin:
    def _set_residual_block(self, res_block_units: List[int]):
        if not (hasattr(self, 'layers_per_residual_block') 
                and 
                hasattr(self, '_activation')
                ):

            raise AttributeError(f"The ResidualMixin class expects the child class to contain the an attribute 'layers_per_residual_block'")

        if len(res_block_units) != self.layers_per_residual_block + 1:
            raise ValueError(f"Make sure to pass the correct number of units when buidling the residual blocks")
        
        # the idea here is quite simple:
        blocks = [LinearBlock(in_features=res_block_units[i], 
                              out_features=res_block_units[i + 1], 
                              activation=self._activation,
                              dropout=self.dropout[i],
                              )
                for i in range(res_block_units - 1)
                ]

        # append another block
        blocks.append(LinearBlock())

    def _set_block():
        pass


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
