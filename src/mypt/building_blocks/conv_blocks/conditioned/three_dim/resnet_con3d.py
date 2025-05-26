import torch
import torch.nn as nn

from typing import Optional, Tuple, Union, Callable

from mypt.building_blocks.auxiliary.film_block import ThreeDimFiLMBlock
from mypt.building_blocks.mixins.general import NonSequentialModuleMixin
from mypt.building_blocks.conv_blocks.conditioned.abstract_cond_resnet import AbstractCondResnetBlock, AbstractCondUpWResBlock

class CondThreeDimWResBlock(AbstractCondResnetBlock):
    """
    A conditioned version of the WideResnet block that incorporates feature-wise
    conditioning through FiLM (Feature-wise Linear Modulation).
    """

    def set_components(self) -> nn.ModuleDict:
        
        components = nn.ModuleDict()

        components["film1"] = ThreeDimFiLMBlock(
            **self._film_params1
        )
        
        components["conv1"] = nn.Conv2d(
            self._in_channels, 
            self._out_channels, 
            kernel_size=3, 
            stride=self._stride, 
            padding=1, 
        )
        
        components["dropout"] = nn.Dropout(self._dropout_rate)

        components["film2"] = ThreeDimFiLMBlock(
            **self._film_params2
        )
        
        components["conv2"] = nn.Conv2d(
            self._out_channels, 
            self._out_channels, 
            kernel_size=3, 
            stride=1,  # Always 1 for the second conv
            padding=1, 
        )

        return components


    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        cond_dimension: int,
        
        inner_dim: int = 256,
        stride: int = 1, 
        dropout_rate: float = 0.0,

        # normalization parameters  
        norm1: Optional[nn.Module] = None,
        norm1_params: Optional[dict] = None,
        norm2: Optional[nn.Module] = None,
        norm2_params: Optional[dict] = None,
       
        # activation parameters
        activation: Optional[Union[str, Callable]] = None,
        activation_params: Optional[dict] = None,
        
        # FiLM activation parameters
        film_activation: Union[str, Callable] = "relu",
        film_activation_params: dict = {},

        # whether to use a convolutional layer as the shortcut connection
        force_residual: bool = False, 
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_dimension=cond_dimension,
            inner_dim=inner_dim,
            stride=stride,
            dropout_rate=dropout_rate,
            norm1=norm1,
            norm1_params=norm1_params,
            norm2=norm2,
            norm2_params=norm2_params,
            activation=activation,
            activation_params=activation_params,    
            film_activation=film_activation,
            film_activation_params=film_activation_params,
            force_residual=force_residual,
        )


    def _forward_main_stream(self, x: torch.Tensor, condition: torch.Tensor, return_condition: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the main stream with conditioning.
        Overriding the GeneralResidualMixin._forward_main_stream method to call the GeneralResidualMixin.residual_forward()

        Args:
            x: Input feature tensor [B, C, H, W]
            condition: Conditioning tensor [B, cond_dim]
            return_condition: added mainly for testing purposes.
        Returns:
            Output tensor after passing through the main stream
        """
        # First FiLM -> Conv -> Dropout sequence
        out = self._components['film1'](x, condition)
        out = self._components['conv1'](out)
        out = self._components['dropout'](out)
        
        # modify the condition to match the dimensions of the current output 
        # the call below shouldn't modify the condition tensor if there stride was set to '1'
        condition = torch.nn.functional.interpolate(condition, size=out.shape[2:], mode='nearest-exact')

        # Second FiLM -> Conv sequence
        out = self._components['film2'](out, condition)
        out = self._components['conv2'](out)
        

        if return_condition:
            return out, condition

        return out
    


class UpCondThreeDimWResBlock(AbstractCondUpWResBlock):
    """
    A conditioned WideResnet block with upsampling capabilities.
    The block passes the input through a CondOneDimWResBlock and then applies upsampling.
    
    Supports multiple upsampling methods:
    - transpose_conv: Uses ConvTranspose2d
    - conv: Uses Conv2d after interpolation
    - interpolate: Uses only interpolation
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        cond_dimension: int,
        inner_dim: int = 256,
        stride: int = 1,
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

        upsample_type: str = "transpose_conv",
    ):

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_dimension=cond_dimension,
            upsample_type=upsample_type
        )

        # Create the conditional resnet block
        self._resnet_block = CondThreeDimWResBlock(
            in_channels=in_channels,  
            out_channels=out_channels,
            cond_dimension=cond_dimension,
            inner_dim=inner_dim,
            stride=stride,
            dropout_rate=dropout_rate,
            norm1=norm1,
            norm1_params=norm1_params,
            norm2=norm2,
            norm2_params=norm2_params,
            activation=activation,
            activation_params=activation_params,
            film_activation=film_activation,
            film_activation_params=film_activation_params,
            force_residual=force_residual
        )
        


class DownCondThreeDimWResBlock(NonSequentialModuleMixin, nn.Module):
    """
    A conditioned WideResnet block with downsampling capabilities.
    The block passes the input through a CondOneDimWResBlock and then applies downsampling.
    
    Supports multiple downsampling methods:
    - conv: Uses strided convolution
    - avg_pool: Uses average pooling
    - max_pool: Uses max pooling
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        cond_dimension: int,
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
        downsample_type: str = "conv",
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_dimension=cond_dimension,
            downsample_type=downsample_type
        )
        
        # Create the conditional resnet block
        self._resnet_block = CondThreeDimWResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_dimension=cond_dimension,
            inner_dim=inner_dim,
            stride=1, # Stride is always 1 as downsampling is handled separately
            dropout_rate=dropout_rate,
            norm1=norm1,
            norm1_params=norm1_params,
            norm2=norm2,
            norm2_params=norm2_params,
            activation=activation,
            activation_params=activation_params,
            film_activation=film_activation,
            film_activation_params=film_activation_params,
            force_residual=force_residual
        )

