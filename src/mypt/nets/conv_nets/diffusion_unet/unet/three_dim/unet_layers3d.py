import torch

from torch import nn
from typing import List, Optional, Union, Callable

from mypt.nets.conv_nets.diffusion_unet.unet.abstract_unet_layers import AbstractDownLayer, AbstractUpLayer
from mypt.building_blocks.conv_blocks.conditioned.three_dim.resnet_con3d import (
    CondThreeDimWResBlock, 
    UpCondThreeDimWResBlock,
    DownCondThreeDimWResBlock
)


class UnetDownLayer3D(AbstractDownLayer):
    """
    Downsampling layer for UNet architecture using DownLayer components.
    """
    def __init__(self, 
        num_resnet_blocks: int,
        in_channels: int,
        cond_dimension: int,
        out_channels: Union[List[int], int],
        inner_dim: int = 256,
        dropout_rate: float = 0.0,
        norm1: Optional[nn.Module] = None,
        norm1_params: Optional[dict] = None,
        norm2: Optional[nn.Module] = None,
        norm2_params: Optional[dict] = None,
        activation: Optional[Union[str, Callable]] = None,
        activation_params: Optional[dict] = None,
        film_activation: Union[str, Callable] = "relu",
        film_activation_params: dict = {},
        force_residual: bool = False,
        downsample_type: str = 'con'):

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_dimension=cond_dimension,
            num_resnet_blocks=num_resnet_blocks,
        )

        resnet_parameters = {
            'norm1': norm1,
            'norm1_params': norm1_params,
            'norm2': norm2,
            'norm2_params': norm2_params,
            
            'activation': activation,
            'activation_params': activation_params,
            'film_activation': film_activation, 
            'film_activation_params': film_activation_params,
            'force_residual': force_residual,
            'dropout_rate': dropout_rate,
            'inner_dim': inner_dim,
        }

        resnet_blocks = [None for _ in range(num_resnet_blocks)]

        for i in range(num_resnet_blocks - 1):
            resnet_blocks[i] = CondThreeDimWResBlock(
                in_channels=self.in_channels if i == 0 else self.out_channels[i-1],
                out_channels=self.out_channels[i],
                cond_dimension=self.cond_dimension,
                stride=1, # the stride must be 1, the down sampling is handled by the downsample_layer
                **resnet_parameters
            )

        resnet_blocks[-1] = DownCondThreeDimWResBlock(
            in_channels=self.out_channels[-2],
            out_channels=self.out_channels[-1],
            cond_dimension=self.cond_dimension,
            downsample_type=downsample_type, # the downsample argument must be passed to the CondThreeDimWResBlock class
            **resnet_parameters
        )

        self._resnet_blocks = nn.ModuleList(resnet_blocks)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # just use the super class forward method 
        return super().forward(x, condition) 
    
    def __call__(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.forward(x, condition)
    
    @property
    def downsample_type(self) -> str:
        return self._resnet_blocks[-1]._downsample_type
    

class UnetUpLayer3D(AbstractUpLayer):
    """
    Upsampling layer for UNet architecture using UpLayer components.
    """
    def __init__(self, 
        num_resnet_blocks: int,
        in_channels: int,
        cond_dimension: int,
        out_channels: List[int],
        inner_dim: int = 256,
        dropout_rate: float = 0.0,
        norm1: Optional[nn.Module] = None,
        norm1_params: Optional[dict] = None,
        norm2: Optional[nn.Module] = None,
        norm2_params: Optional[dict] = None,
        activation: Optional[Union[str, Callable]] = None,
        activation_params: Optional[dict] = None,
        film_activation: Union[str, Callable] = "relu",
        film_activation_params: dict = {},
        force_residual: bool = False,
        upsample_type: str = 'con'):

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_dimension=cond_dimension,
            num_resnet_blocks=num_resnet_blocks,
        )

        resnet_parameters = {
            'norm1': norm1,
            'norm1_params': norm1_params,
            'norm2': norm2,
            'norm2_params': norm2_params,
            'activation': activation,
            'activation_params': activation_params,
            'film_activation': film_activation,
            'film_activation_params': film_activation_params,
            'force_residual': force_residual,
            'dropout_rate': dropout_rate,
            'inner_dim': inner_dim,
        }

        resnet_blocks = [None for _ in range(num_resnet_blocks)]

        for i in range(num_resnet_blocks - 1):
            resnet_blocks[i] = CondThreeDimWResBlock(
                in_channels=self.in_channels if i == 0 else self.out_channels[i-1],
                out_channels=self.out_channels[i],
                cond_dimension=self.cond_dimension,
                stride=1, # the stride must be 1, the up sampling is handled by the upsample_layer
                **resnet_parameters
            )

        resnet_blocks[-1] = UpCondThreeDimWResBlock(
            in_channels=self.out_channels[-2],
            out_channels=self.out_channels[-1],
            cond_dimension=self.cond_dimension,
            upsample_type=upsample_type, # the upsample argument must be passed to the CondThreeDimWResBlock class
            **resnet_parameters
        )

        self._resnet_blocks = nn.ModuleList(resnet_blocks)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # just use the super class forward method 
        return super().forward(x, condition)

    def __call__(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.forward(x, condition)   
        
    @property
    def upsample_type(self) -> str:
        return self._resnet_blocks[-1]._upsample_type
    