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
        self._block = self._build() # the self._build is implemented in the GeneralLinearBlockMixin

    def get_constructor_args(self) -> dict:
        return {
            'output': self.output,
            'in_features': self.in_features,
            'num_layers': self.num_layers,
            'units': self.units,
            'activation': self.activation,
            'dropout': self.dropout,
        }

class ExponentialFCBlock(
                        GeneralLinearBlockMixin,
                        ExponentialLinearBlockMixin, 
                        FullyConnectedBlock):
    def __init__(self,
                 output: int,
                 in_features: int,
                 num_layers: int,
                 dropout: Optional[Union[List[float], float]]=None,
                 activation: str = 'relu'):
        
        super().__init__(output=output, 
                         in_features=in_features,
                         num_layers=num_layers,
                         activation=activation,
                         dropout=dropout
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
        }





# class ExponentialResidualFCBlock(
#                         ResidualLinearBlockMixin,  
#                         ExponentialLinearBlockMixin, 
#                         ModuleListMixin,
#                         FullyConnectedBlock
#                         ):
#     """
#     This class inherits the following classes to cover different functionalities:

#         FullyConnectedBlock : the base class for any fully connected block with a final layer (no activation / dropout at the last layer)
#         ResidualMixin: build the classifier as a sequence of Residual linear blocks
#         ExponentialMixin : set the number of hidden units in each layer using in an exponential distribution
#         SequentialModuleListMixin : correctly set the classifier field as a ModuleList object  

#     """
#     def __init__(self,
#                  output: int,
#                  in_features: int,
#                  num_layers: int,
#                  layers_per_residual_block:int,
#                  dropout: Optional[Union[List[float], float]]=None,
#                  activation: str = 'relu'):
        
#         # the usual parent's class call
#         super().__init__(output=output, 
#                          in_features=in_features,
#                          num_layers=num_layers,
#                          activation=activation,
#                          dropout=dropout
#                          )
                
#         max_dim, min_dim = max(self.output, in_features), min(self.output, in_features)

#         if 2 ** (int(np.log2(min_dim)) + 1) > max_dim and num_layers >= 2:
#             raise ValueError(f"Please make sure the difference is large enough between the number of classes and 'in_features' argument\n"
#                              f"smallest dimension: {min_dim}. Largest dimension: {max_dim}")
        
#         self.layers_per_residual_block = layers_per_residual_block

#         self.units: List[int] = None

#         self._build()


#     def _build(self):
#         # get the number of units using the ExponentialMixin 
#         self.units = self._set_units()
#         self._block = self._build()

#     def to(self, *args, **kwargs):
#         return self.module_list_to(*args, **kwargs)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.module_list_forward(x)
