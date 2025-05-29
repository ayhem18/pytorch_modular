import torch
import torch.nn as nn

from typing import List, Union, Optional, Callable

from mypt.nets.conv_nets.diffusion_unet.unet.three_dim.unet_blocks3d import (
    UnetDownBlock3D,
    UnetUpBlock3D,
    UNetMidBlock3D
)


class UNet3DCond(nn.Module):
    """
    UNet architecture implementation using the builder design pattern.
    
    This UNet consists of three main components:
    1. Down Block: Series of downsampling layers
    2. Middle Block: Processing at the bottleneck
    3. Up Block: Series of upsampling layers
    
    Each component can be configured separately using the build_* methods.
    
    Args:
        cond_dimension: Dimension of the conditioning input
    """
    def __init__(self, cond_dimension: int, film_dimension: int):
        super().__init__()
        
        self.cond_dimension = cond_dimension
        self.film_dimension = film_dimension
        
        self.num_down_layers: int = None
        self.out_channels: Union[List[int], List[List[int]]] = None
        
        # Initialize components as None
        self._down_block: UnetDownBlock1D = None
        self._middle_block: UNetMidBlock1D = None
        self._up_block: UnetUpBlock1D = None
    
    def build_down_block(
        self,
        num_down_layers: int,
        num_resnet_blocks: int,
        in_channels: int,
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
        film_activation: Union[str, Callable] = "relu",
        film_activation_params: dict = {},
        force_residual: bool = False,
    ):
        """
        Configure and build the downsampling block.
        
        Args:
            num_down_blocks: Number of downsampling stages
            num_resnet_blocks: Number of residual blocks per stage
            in_channels: Input channels for the network
            out_channels: List of output channels for each stage
            downsample_types: Type of downsampling to use
            inner_dim: Inner dimension for FiLM layer
            dropout_rate: Dropout rate
            norm1, norm1_params: Normalization for the first block
            norm2, norm2_params: Normalization for the second block
            activation, activation_params: Activation function and parameters
            film_activation, film_activation_params: FiLM activation
            force_residual: Whether to force residual connections
            
        Returns:
            self (for method chaining)
        """
        # save the important parameters
        self.num_down_layers = num_down_layers
        self.out_channels = out_channels

        self._down_block = UnetDownBlock1D(
            num_down_layers=num_down_layers,
            num_resnet_blocks=num_resnet_blocks,
            in_channels=in_channels,
            cond_dimension=self.cond_dimension,
            out_channels=out_channels,
            downsample_types=downsample_types,
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
        in_channels: int,
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
    ):
        """
        Configure and build the middle block.
        
        Args:
            num_resnet_blocks: Number of residual blocks
            in_channels: Input channels
            out_channels: Output channels (default: same as input)
            inner_dim: Inner dimension for FiLM layer
            dropout_rate: Dropout rate
            norm1, norm1_params: Normalization for the first block
            norm2, norm2_params: Normalization for the second block
            activation, activation_params: Activation function and parameters
            film_activation, film_activation_params: FiLM activation
            force_residual: Whether to force residual connections
            
        Returns:
            self (for method chaining)
        """
        self._middle_block = UNetMidBlock(
            num_resnet_blocks=num_resnet_blocks,
            in_channels=in_channels,
            film_dimension=self.film_dimension,
            out_channels=in_channels, # this ensures that output of the middle block has the same shape as its input
            cond_channels=self.cond_channels,
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
        film_activation: Union[str, Callable] = "relu",
        film_activation_params: dict = {},
        force_residual: bool = False,
    ):
        """
        Build an upsampling block that mirrors the downsampling block.
        This convenience method ensures that the up block matches the down block structure.
        
        Args:
            num_resnet_blocks: Number of residual blocks per stage
            in_channels: Input channels at the bottleneck
            upsample_types: Type of upsampling to use
            inner_dim: Inner dimension for FiLM layer
            dropout_rate: Dropout rate
            norm1, norm1_params: Normalization for the first block
            norm2, norm2_params: Normalization for the second block
            activation, activation_params: Activation function and parameters
            film_activation, film_activation_params: FiLM activation
            force_residual: Whether to force residual connections
            
        Returns:
            self (for method chaining)
        """
        if self._down_block is None:
            raise ValueError("Cannot build symmetric up block - down block has not been built yet")
        
        # reverse the order of the output channels of the down block
        up_out_channels = self.out_channels[::-1]
    
        # Build the up block
        self._up_block = UnetUpBlock(
            num_up_blocks=self.num_down_blocks,
            num_resnet_blocks=num_resnet_blocks,
            in_channels=self.out_channels[-1],
            out_channels=up_out_channels,
            upsample_types=upsample_types,
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
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the UNet.
        
        Args:
            x: Input tensor
            condition: Conditioning tensor
            
        Returns:
            Output tensor
        """
        # Validate that all components have been built
        if self._down_block is None:
            raise RuntimeError("Down block has not been built. Call build_down_block first.")
        if self._middle_block is None:
            raise RuntimeError("Middle block has not been built. Call build_middle_block first.")
        if self._up_block is None:
            raise RuntimeError("Up block has not been built. Call build_symmetric_up_block first.")
        
        # Downsampling path - returns features and skip connections
        x, skip_connections = self._down_block(x, condition)
        
        # Middle block - processes features at the bottleneck
        x = self._middle_block(x, condition)
        
        # Upsampling path - uses skip connections
        x = self._up_block(x, skip_connections, condition)
        
        return x

    