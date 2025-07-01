import torch
import torch.nn as nn

from typing import List, Union, Optional, Callable

from mypt.nets.conv_nets.diffusion_unet.unet.abstract_unet import AbstractUNetCond
from mypt.nets.conv_nets.diffusion_unet.unet.one_dim.unet_blocks1d import (
    UnetDownBlock1D,
    UnetUpBlock1D,
    UNetMidBlock1D
)


class UNet1DCond(AbstractUNetCond):
    """
    UNet architecture implementation for 1D conditioning.
    
    This UNet consists of three main components:
    1. Down Block: Series of downsampling layers
    2. Middle Block: Processing at the bottleneck
    3. Up Block: Series of upsampling layers
    
    Each component can be configured separately using the build_* methods.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels 
        cond_dimension: Dimension of the conditioning input
    """
    def __init__(self, in_channels: int, out_channels: int, cond_dimension: int, *args, **kwargs):
        super().__init__(in_channels, out_channels, cond_dimension, *args, **kwargs)
        
        # Initialize components as None
        self._down_block: UnetDownBlock1D = None
        self._middle_block: UNetMidBlock1D = None
        self._up_block: UnetUpBlock1D = None


        self._is_built: bool = False
    
    
    def build_down_block(
        self,
        num_down_layers: int,
        num_resnet_blocks: int,
        out_channels: List[int],
        downsample_types: Union[str, List[str]] = "conv",
        inner_dim: int = 256,
        dropout_rate: float = 0.0,
        norm1: Optional[nn.Module] = None,
        norm1_params: Optional[dict] = None,
        norm2: Optional[nn.Module] = None,
        norm2_params: Optional[dict] = None,
        activation: Optional[Union[str, Callable]] = None,
        activation_params: Optional[dict] = None,
        film_activation: Union[str, Callable] = "silu", # updated after inspecting the diffusers library.
        film_activation_params: dict = {},
        force_residual: bool = False,
    ):
        """
        Configure and build the downsampling block.
        
        Args:
            num_down_layers: Number of downsampling layers
            num_resnet_blocks: Number of residual blocks per stage
            out_channels: List of output channels for each stage
            downsample_types: Type of downsampling to use
            inner_dim: Inner dimension for FiLM layer
            dropout_rate: Dropout rate
            norm1, norm2: Normalization layers
            norm1_params, norm2_params: Parameters for normalization layers
            activation: Activation function
            activation_params: Parameters for activation function
            film_activation: Activation function for FiLM
            film_activation_params: Parameters for FiLM activation
            force_residual: Whether to force residual connections
        """
        self.num_down_layers = num_down_layers
        self.down_block_num_resnet_blocks = num_resnet_blocks
        self.down_block_out_channels = out_channels

        # if not isinstance(out_channels, list) or len(out_channels) != num_down_layers + 1:
        #     raise ValueError(f"out_channels must be a list of length num_down_layers + 1 (got {len(out_channels)} and {num_down_layers + 1})")

        # Build the down block
        self._down_block = UnetDownBlock1D(
            num_down_layers=num_down_layers,
            num_resnet_blocks=num_resnet_blocks,
            in_channels=self.input_channels,
            out_channels=out_channels,
            downsample_types=downsample_types,
            cond_dimension=self.cond_dimension,
            inner_dim=inner_dim,
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
        
        return self

    def build_middle_block(
        self,
        num_resnet_blocks: int,
        inner_dim: int = 256,
        dropout_rate: float = 0.0,
        norm1: Optional[nn.Module] = None,
        norm1_params: Optional[dict] = None,
        norm2: Optional[nn.Module] = None,
        norm2_params: Optional[dict] = None,
        activation: Optional[Union[str, Callable]] = None,
        activation_params: Optional[dict] = None,
        film_activation: Union[str, Callable] = "silu", # updated after inspecting the diffusers library.
        film_activation_params: dict = {},
        force_residual: bool = False,
    ):
        """
        Configure and build the middle block.
        
        Args:
            num_resnet_blocks: Number of residual blocks
            inner_dim: Inner dimension for FiLM layer
            dropout_rate: Dropout rate
            norm1, norm2: Normalization layers
            norm1_params, norm2_params: Parameters for normalization layers
            activation: Activation function
            activation_params: Parameters for activation function
            film_activation: Activation function for FiLM
            film_activation_params: Parameters for FiLM activation
            force_residual: Whether to force residual connections
        """
        if self._down_block is None:
            raise ValueError("Cannot build a middle block without building the down block first")

        self.middle_block_num_resnet_blocks = num_resnet_blocks

        # Build the middle block
        self._middle_block = UNetMidBlock1D(
            in_channels=self.down_block_out_channels[-1],  # the input channels of the middle block is the output channels of the last downsampling layer
            cond_dimension=self.cond_dimension,
            num_resnet_blocks=num_resnet_blocks,
            inner_dim=inner_dim,
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
        
        return self
    
    
    

    
    def build_up_block(
        self,
        num_resnet_blocks: int,
        upsample_types: Union[str, List[str]] = "transpose_conv",
        inner_dim: int = 256,
        dropout_rate: float = 0.0,
        norm1: Optional[nn.Module] = None,
        norm1_params: Optional[dict] = None,
        norm2: Optional[nn.Module] = None,
        norm2_params: Optional[dict] = None,
        activation: Optional[Union[str, Callable]] = None,
        activation_params: Optional[dict] = None,
        film_activation: Union[str, Callable] = "silu", # updated after inspecting the diffusers library.
        film_activation_params: dict = {},
        force_residual: bool = False,
    ):
        """
        Configure and build the upsampling block.
        
        Args:
            num_resnet_blocks: Number of residual blocks per stage
            upsample_types: Type of upsampling to use
            inner_dim: Inner dimension for FiLM layer
            dropout_rate: Dropout rate
            norm1, norm2: Normalization layers
            norm1_params, norm2_params: Parameters for normalization layers
            activation: Activation function
            activation_params: Parameters for activation function
            film_activation: Activation function for FiLM
            film_activation_params: Parameters for FiLM activation
            force_residual: Whether to force residual connections
        """
        if self._down_block is None or self._middle_block is None:
            raise ValueError("Cannot build an up block without building the down and middle blocks first")
        
        self.up_block_num_resnet_blocks = num_resnet_blocks

        # the last output channel of the down block is the same as the input channel of the up block 
        # so block_down_output_channels[::-1][1:] is the list of output channels of the up block  
        # and the last output channel must match the entire architecture output channel saved in self.final_out_channels
        up_out_channels = self.down_block_out_channels[::-1][1:] + [self.final_out_channels]

        # Build the up block
        self._up_block = UnetUpBlock1D(
            num_up_layers = self.num_down_layers, # the number of upsampling layers must be the same the number of downsampling layers
            num_resnet_blocks=num_resnet_blocks,
            in_channels=self.down_block_out_channels[-1], # the number of input channels is the same as the number of output channels of the last downsampling layer 
            out_channels=up_out_channels,
            upsample_types=upsample_types,
            cond_dimension=self.cond_dimension,

            inner_dim=inner_dim,
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

        self._is_built = True   
        return self

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the UNet.
        
        Args:
            x: Input tensor
            condition: Conditioning tensor (1D)
            
        Returns:
            Output tensor
        """
        if not self._is_built:
            raise RuntimeError("UNet1DCond has not been built. Call build_down_block, build_middle_block, and build_up_block first.")
        
        self._verify_input(x)   

        # Downsampling path - returns features and skip connections
        x, skip_connections = self._down_block(x, condition)
        
        # Middle block - processes features at the bottleneck
        x = self._middle_block(x, condition)
        
        # Upsampling path - uses skip connections
        x = self._up_block(x, skip_connections, condition)
        
        return x

    def __call__(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.forward(x, condition) 
    

    # the NonSequentialModuleMixin proceeds the torch.nn.Module in the class definition order
    # hence the methods to(), train(), eval(), children(), modules(), parameters(), named_parameters() 
    # will use the NonSequentialModuleMixin's implementation, not the torch.nn.Module's 