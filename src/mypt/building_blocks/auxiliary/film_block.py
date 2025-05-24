import torch

from torch import nn
from typing import Callable, Union

from mypt.building_blocks.auxiliary.norm_act import ConditionedNormActBlock
from mypt.building_blocks.auxiliary.normalization.utils import get_normalization
from mypt.building_blocks.auxiliary.activations.activations import get_activation


class OneDimFiLMBlock(ConditionedNormActBlock):
    """
    A film block that learns both from the input and a condition tensor using the FiLM transformation as described in the paper:
    https://arxiv.org/abs/1709.07871
    """

    def __init__(self, 
                out_channels: int, # the number of channels in the output tensor
                cond_dimension: int, # the dimension of the condition tensor

                normalization: Union[str, Callable],
                normalization_params: dict,
                activation: Union[str, Callable],
                activation_params: dict,

                inner_dim: int = 256,
                film_activation: Union[str, Callable] = "relu",
                film_activation_params: dict = {},
                ):
        super().__init__(normalization, normalization_params, activation, activation_params, ['_film_layer', '_normalization', '_activation'])

        # although the condition tensor is 2 dimensional, the input is expected to be 3 dimensional (c, h, w)
        # the constructor should accept only 3d normalization functions
        if not any([isinstance(self._normalization, nn.BatchNorm2d), 
                    isinstance(self._normalization, nn.GroupNorm), 
                    isinstance(self._normalization, nn.LayerNorm)]):
            raise ValueError("The normalization function should be a 3d normalization function. Found: {}".format(normalization))

        self._cond_dimension = cond_dimension
        self._out_channels = out_channels

        self._film_layer = nn.Sequential(
            nn.Linear(cond_dimension, inner_dim),
            get_activation(film_activation, film_activation_params),
            nn.Linear(inner_dim, 2 * out_channels)
        )



    def forward(self, 
                x: torch.Tensor, 
                condition: torch.Tensor) -> torch.Tensor:
        
        # make sure the condition tensor is 2 dimensional 
        if condition.dim() != 2:
            raise ValueError("The condition tensor should be 2 dimensional") 

        if condition.shape[1] != self._cond_dimension:
            raise ValueError(f"The condition tensor should be of shape (batch_size, {self._cond_dimension}). Found: {condition.shape}")

        film_params = self._film_layer(condition) # film_params of the shape (batch_size, 2 * out_channels, height, width)
        gamma, beta = torch.chunk(film_params, 2, dim=1) # this function call returns two tensors of the shape (batch_size, out_channels, height, width)
        
        # since gamma and beta are 2 dimensional, they should be reshaped for broadcasting
        gamma = gamma.view(gamma.shape[0], gamma.shape[1], 1, 1)
        beta = beta.view(beta.shape[0], beta.shape[1], 1, 1)

        out = self._normalization(x)
        out = out * gamma + beta
        return self._activation(out)
    
    @property
    def cond_dimension(self) -> int:
        return self._cond_dimension
    
    @property
    def out_channels(self) -> int:
        return self._out_channels


class ThreeDimFiLMBlock(ConditionedNormActBlock):
    """
    A film block that learns both from the input and a condition tensor using the FiLM transformation adapted to 3d inputs 
    (inspired by th diffusers library implementation !!)
    """

    def __init__(self, 
                out_channels: int, # the number of channels in the output tensor
                cond_dimension: int, # the dimension of the condition tensor

                normalization: Union[str, Callable],
                normalization_params: dict,
                activation: Union[str, Callable],
                activation_params: dict,

                inner_dim: int = 256,
                film_activation: Union[str, Callable] = "relu",
                film_activation_params: dict = {}
                ):

        super().__init__(normalization, normalization_params, activation, activation_params, ['_film_layer', '_normalization', '_activation'])

        # the constructor should accept only 3d normalization functions
        if not any([isinstance(self._normalization, nn.BatchNorm2d), 
                    isinstance(self._normalization, nn.GroupNorm), 
                    isinstance(self._normalization, nn.LayerNorm)]):
            raise ValueError("The normalization function should be a 3d normalization function. Found: {}".format(normalization))

        self._cond_dimension = cond_dimension
        self._out_channels = out_channels

        self._film_layer = nn.Sequential(
            nn.Conv2d(cond_dimension, inner_dim, kernel_size=3, padding=1, stride=1),
            get_activation(film_activation, film_activation_params),
            nn.Conv2d(inner_dim, 2 * out_channels, kernel_size=3, padding=1, stride=1)
        )

    def forward(self, 
                x: torch.Tensor, 
                condition: torch.Tensor) -> torch.Tensor:
        
        # make sure the condition tensor is 3 dimensional 
        if condition.dim() != 4:
            raise ValueError(f"The condition tensor should be of 4 dimensional. Found: {condition.shape}") 

        if condition.shape[1] != self._cond_dimension:
            raise ValueError(f"The condition tensor should be of shape (batch_size, {self._cond_dimension}, height, width). Found: {condition.shape}")

        film_params = self._film_layer(condition) # film_params of the shape (batch_size, 2 * out_channels, height, width)
        gamma, beta = torch.chunk(film_params, 2, dim=1) # this function call returns two tensors of the shape (batch_size, out_channels, height, width)
        
        out = self._normalization(x)
        out = out * gamma + beta
        return self._activation(out)

