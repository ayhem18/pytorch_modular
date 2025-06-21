"""
This script contains different general classes of classification heads with different design choices: either my own
design or inspired from other sources.
"""


from typing import Union, List, Optional

from .components import FullyConnectedBlock
from .mixins import ExponentialLinearBlockMixin, GeneralLinearBlockMixin


class GenericFCBlock(
            GeneralLinearBlockMixin,
            FullyConnectedBlock
            ):
    def __init__(self, 
                 output: int,
                 in_features: int,
                 num_layers:int, 
                 units: List[int],
                 activation='relu',
                 dropout: Optional[Union[List[float], float]]=None,
                 norm_layer: str = "batchnorm1d"
                 ):
        
        if len(units) != num_layers + 1:
            raise ValueError(f"The Generic Fully Connected block expects the 'units' argument to contain {num_layers + 1} elements. Found: {len(units)}")

        super().__init__(output=output,
                         in_features=in_features,
                         num_layers=num_layers,
                         activation=activation,
                         dropout=dropout,
                         norm_layer=norm_layer)

        self.units = units
        self._block = self._build() # the self._build is implemented in the GeneralLinearBlockMixin

    def get_constructor_args(self) -> dict:
        return {
            'output': self.output,
            'in_features': self.in_features,
            'num_layers': self.num_layers,
            'units': self.units,
            'activation': self.activation,
            'dropout': self.dropout,
            'norm_layer': self.norm_layer,
        }

class ExponentialFCBlock(
                        GeneralLinearBlockMixin,
                        ExponentialLinearBlockMixin, 
                        FullyConnectedBlock,
                        ):
    def __init__(self,
                 output: int,
                 in_features: int,
                 num_layers: int,
                 dropout: Optional[Union[List[float], float]]=None,
                 activation: str = 'relu',
                 norm_layer: str = "batchnorm"):
        
        super().__init__(output=output, 
                         in_features=in_features,
                         num_layers=num_layers,
                         activation=activation,
                         dropout=dropout,
                         norm_layer=norm_layer
                         )
        # set the units to None first 
        # self.units = None 
        self.units = self._set_units() # implemented in the ExponentialLinearBlockMixin
        self._block = self._build() # implemented in the GeneralLinearBlockMixin


    def get_constructor_args(self) -> dict:
        return {
            'output': self.output,
            'in_features': self.in_features,
            'num_layers': self.num_layers,
            'units': self.units,
            'activation': self.activation,
            'dropout': self.dropout,
            'norm_layer': self.norm_layer,
        }

