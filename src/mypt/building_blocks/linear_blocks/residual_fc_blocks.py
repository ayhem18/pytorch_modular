"""
This script contains implementations of residual fully connected blocks with different architecture patterns.
"""

from torch import nn
from typing import Union, List, Optional


from .components import ResidualFullyConnectedBlock
from mypt.building_blocks.linear_blocks.mixins import ExponentialLinearBlockMixin, GeneralLinearBlockMixin


class GenericResidualFCBlock(GeneralLinearBlockMixin, ResidualFullyConnectedBlock):
    """
    A generic fully connected block with residual connections.
    This class combines the functionality of GenericFCBlock with residual connections.
    """
    def __init__(self, 
                 output: int,
                 in_features: int,
                 num_layers: int, 
                 units: List[int],
                 activation='relu',
                 dropout: Optional[Union[List[float], float]]=None,
                 force_residual: bool=False):
        
        # Initialize the ResidualFullyConnectedBlock first
        ResidualFullyConnectedBlock.__init__(
            self,
            output=output,
            in_features=in_features,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
            force_residual=force_residual
        )
        
        # Set additional attributes from GenericFCBlock
        self.units = units
        
        self._build()

    def _build(self):
        # the super()._build() method is defined by both ResidualFullyConnectedBlock and GenericFCBlock  
        # which implementation is called depends on the order of inheritance 
        # if ResidualFullyConnectedBlock is before GenericFCBlock, then the implementation of ResidualFullyConnectedBlock is called 
        # which is funny enough an abstract method

        # so I either need to somehow call the implementation of _build() of a specific parent class (don't know how to do that yet)
        # or simply make GeneralLinearBlockMixin the first parent class (hence the _build method will be the default one)

        self._block = super()._build()  
    
        if self._residual_stream_field_name is not None:
            self._adaptive_layer = nn.Linear(in_features=self._in_features, out_features=self._output) 


    def get_constructor_args(self) -> dict:
        """Get arguments needed to reconstruct this object"""
        return {
            'output': self.output,
            'in_features': self.in_features,
            'num_layers': self.num_layers,
            'units': self.units,
            'activation': self.activation,
            'dropout': self.dropout,
            'force_residual': self._force_residual
        }


class ExponentialResidualFCBlock(GeneralLinearBlockMixin, ExponentialLinearBlockMixin, ResidualFullyConnectedBlock):
    """
    An exponential fully connected block with residual connections.
    This class combines the functionality of ExponentialFCBlock with residual connections.
    """
    def __init__(self,
                 output: int,
                 in_features: int,
                 num_layers: int,
                 activation: str = 'relu',
                 dropout: Optional[Union[List[float], float]]=None,
                 force_residual: bool=False):
        
        # Initialize the ResidualFullyConnectedBlock first
        ResidualFullyConnectedBlock.__init__(
            self,
            output=output,
            in_features=in_features,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
            force_residual=force_residual
        )
        
        # Set the units using the exponential approach
        self.units = self._set_units()
        
        # Build the block
        self._build()



    def _build(self):
        # the _build() method is defined by both ResidualFullyConnectedBlock and GenericFCBlock 
        # however, it is abstract in the ResidualFullyConnectedBlock class 
        # so the line below should call the _build() method of the GenericFCBlock class 
        self._block = super()._build()  
    
        if self._residual_stream_field_name is not None:
            self._adaptive_layer = nn.Linear(in_features=self._in_features, out_features=self._output) 


    def get_constructor_args(self) -> dict:
        """Get arguments needed to reconstruct this object"""
        return {
            'output': self.output,
            'in_features': self.in_features,
            'num_layers': self.num_layers,
            'activation': self.activation,
            'dropout': self.dropout,
            'force_residual': self._force_residual
        }
