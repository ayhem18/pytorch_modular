import torch
import torch.nn as nn

from typing import List, Union, Optional, Callable, Tuple

from mypt.building_blocks.conv_blocks.conditioned.three_dim.resnet_con3d import CondThreeDimWResBlock
from mypt.nets.conv_nets.diffusion_unet.unet.three_dim.unet_layers3d import UnetDownLayer3D, UnetUpLayer3D
from mypt.nets.conv_nets.diffusion_unet.unet.abstract_unet_blocks import AbstractUnetDownBlock, AbstractUnetUpBlock, AbstractUnetMidBlock


class UnetDownBlock3D(AbstractUnetDownBlock):
    """
    Downsampling block for UNet architecture using DownLayer components.
    """
    def __init__(self, 
                 num_down_layers: int,
                 num_resnet_blocks: int,
                 in_channels: int,
                 cond_dimension: int,
                 out_channels: Union[List[int], List[List[int]]],
                 downsample_types: Union[str, List[str]] = "conv",  
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
                 *args, **kwargs):
        
        super().__init__(
            num_down_layers,
            num_resnet_blocks,
            in_channels,
            cond_dimension,
            out_channels,
            downsample_types,
            inner_dim,
            dropout_rate,
            norm1,
            norm1_params,
            norm2,
            norm2_params,
            activation,
            activation_params,
            film_activation,
            film_activation_params,
            force_residual,
            *args, **kwargs
        )

        # Create the down layers
        down_layers = [None for _ in range(num_down_layers)]

        for i in range(num_down_layers):
            down_layers[i] = UnetDownLayer3D(
                in_channels=self.in_channels if i == 0 else self.out_channels[i-1][0],
                out_channels=self.out_channels[i],
                downsample_type=self.downsample_types[i],
                **self._layer_params
            )
        # set the self.down_layers field
        self.down_layers = nn.ModuleList(down_layers)



    # override the forward method
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # return super().forward(x, condition)
        skip_connections = [None for _ in self.down_layers]

        for i, layer in enumerate(self.down_layers):
            x = layer.forward(x, condition)
            skip_connections[i] = x
            # the condition variable must be interpolated to the correct shape
            # thanks to this mysterious gentlman on stack overflow 
            # https://stackoverflow.com/questions/75836118/pytorch-interpolate-input-and-output-must-have-the-same-number-of-spatial-dim  
            condition = nn.functional.interpolate(condition, size=tuple(x.shape[2:]), mode="nearest-exact")

        return x, skip_connections



class UnetUpBlock3D(AbstractUnetUpBlock):
    """
    Upsampling block for UNet architecture using UpLayer components.
    """
    def __init__(self, 
                 num_up_layers: int,
                 num_resnet_blocks: int,
                 in_channels: int,
                 cond_dimension: int,
                 out_channels: Union[List[int], List[List[int]]],
                 upsample_types: Union[str, List[str]] = "transpose_conv",  
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
                 *args, **kwargs):
        super().__init__(
            num_up_layers,
            num_resnet_blocks,
            in_channels,
            cond_dimension,
            out_channels,
            upsample_types,
            inner_dim,
            dropout_rate,
            norm1,
            norm1_params,
            norm2,
            norm2_params,
            activation,
            activation_params,
            film_activation,
            film_activation_params,
            force_residual,
            *args, **kwargs)

        # Create the up layers
        up_layers = [None for _ in range(num_up_layers)]

        for i in range(num_up_layers):
            up_layers[i] = UnetUpLayer3D(
                in_channels=self.in_channels if i == 0 else self.out_channels[i-1][0],
                out_channels=self.out_channels[i],
                upsample_type=self.upsample_types[i],
                **self._layer_params
            )

        self.up_layers = nn.ModuleList(up_layers)


    def forward(self, 
                x: torch.Tensor, 
                skip_outputs: List[torch.Tensor], 
                condition: torch.Tensor, 
                reverse_skip_connections: bool = True) -> torch.Tensor:
        
        if len(skip_outputs) != len(self.up_layers):
            raise ValueError(f"Expected skip_outputs to have {len(self.up_layers)} elements, but got {len(skip_outputs)}")
        
        # Reverse skip connections to match the order of up layers
        if reverse_skip_connections:
            skip_outputs = skip_outputs[::-1]

        # the condition is a variable of shape (batch_size, cond_dimension, height, width)
        # the inner components of the UnetBlock expect condition vectors to be of the same shape as the input 'x'
        # so we need to interpolate the condition to the correct shape 
        for i, layer in enumerate(self.up_layers):
            if skip_outputs[i].shape != x.shape:
                raise ValueError(f"Expected skip_outputs[{i}] to have shape {x.shape}, but got {skip_outputs[i].shape}")
            layer_input = x + skip_outputs[i]
            # the condition variable must be interpolated to the correct shape 
            x = layer.forward(layer_input, condition)
            # the condition variable must be interpolated to the correct shape 
            condition = nn.functional.interpolate(condition, size=x.shape[2:], mode="nearest-exact")

        return x



class UNetMidBlock3D(AbstractUnetMidBlock):
    """
    Middle block for UNet architecture using CondResnet components.
    """
    def __init__(self, 
            num_resnet_blocks: int,
            in_channels: int,
            cond_dimension: int,
            out_channels: Optional[int] = None,
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
                 *args, **kwargs):
        super().__init__(
            num_resnet_blocks,
            in_channels,
            cond_dimension,
            out_channels,
            inner_dim,
            dropout_rate,
            norm1,
            norm1_params,
            norm2,
            norm2_params,
            activation,
            activation_params,
            film_activation,
            film_activation_params,
            force_residual,
            *args, **kwargs
        )

        # Create the mid blocks directly as a sequence of ConditionalWResBlock
        mid_blocks = [None for _ in range(num_resnet_blocks)]

        for i in range(num_resnet_blocks):
            mid_blocks[i] = CondThreeDimWResBlock(
                in_channels=self.in_channels if i == 0 else self.out_channels,
                out_channels=self.out_channels,
                cond_dimension=self.cond_dimension,
                **self._block_params
            )
        
        self._mid_blocks = nn.ModuleList(mid_blocks)

