
import math
import torch
import numpy as np

from typing import List
from collections import OrderedDict

from mypt.building_blocks.linear_blocks.components import BasicLinearBlock


class GeneralLinearBlockMixin:   
    """
    GenericBuildMixin, given a self.units field, builds a nn.SequentialModel.
    The number of layers (n) is assumed to be self.units - 1, the first n - 1 are non-final linear blocks and the last one is final
    """

    def _verify_instance_generalLinearBlockMixin(self):
        attrs = ['units', 'activation', 'dropout']
        
        for att in attrs:
            if not hasattr(self, att):
                raise AttributeError(f"The GeneralLinearBlockMixin expects the child class to include the following attributes: {attrs}. Found {att} missing")            
                
        if len(self.units) < 2:
            raise ValueError(f"the number of elements in the self.units must be at least 2")



    def _build(self) -> torch.nn.Sequential:
        self._verify_instance_generalLinearBlockMixin()

        blocks = [BasicLinearBlock(in_features=self.units[i],
                                out_features=self.units[i + 1],
                                is_final=False,
                                activation=self.activation,
                                dropout=self.dropout[i]
                            ) 
                    for i in range(len(self.units) - 2)
                ]
        
        # add the last layer by setting the 'is_final' argument to True
        blocks.append(BasicLinearBlock(in_features=self.units[-2],
                                    out_features=self.units[-1],
                                    is_final=True)
                    )

        # convert the list of blocks into an OrderedDict so the different inner blocks can be accessed
        blocks = OrderedDict([(f'fc_{i}', b) for i, b in enumerate(blocks, start=1)])

        return torch.nn.Sequential(blocks)



class ExponentialLinearBlockMixin:
    """
    This mixin is used to set the number of units in a linear block in an exponential scale (linear in log2 space)
    """

    def _verify_instance_exponentialLinearBlockMixin(self):
        attrs = ['num_layers', 'output', 'in_features']
        for att in attrs:
            if not hasattr(self, att):
                raise AttributeError(f"The child class extending the 'ExponentialMinin' class must have the following attributes: {attrs}. Found {att} missing")
        
        max_dim, min_dim = max(self.output, self.in_features), min(self.output, self.in_features)

        if int(np.log2(max_dim)) - int(np.log2(min_dim)) < 1 and self.num_layers >= 2:
            raise ValueError(f"Please make sure the difference is large enough between the number of classes and 'in_features' argument\n"
                             f"smallest dimension: {min_dim}. Largest dimension: {max_dim}")

        

    def _set_units(self) -> List[int]:
        # let's start with making sure the child class has the necessary attributes
        self._verify_instance_exponentialLinearBlockMixin()
            
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
        
        # just to make sure the number of units is correct
        if len(num_units) != self.num_layers + 1:
            raise ValueError(f"The number of units is not correct. Expected {self.num_layers + 1}, got {len(num_units)}")

        return num_units

