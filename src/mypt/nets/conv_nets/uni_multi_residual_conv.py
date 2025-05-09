import torch
from torch import nn
from typing import List, Union, Optional

from mypt.building_blocks.conv_blocks.residual_conv_block import ResidualConvBlock
from mypt.building_blocks.mixins.custom_module_mixins import WrapperLikeModuleMixin
from mypt.building_blocks.conv_blocks.conv_block_design.conv_design_utils import compute_log_linear_sequence


class UniformMultiResidualNet(WrapperLikeModuleMixin):
    """
    A network composed of multiple sequential ResidualConvBlocks with uniform structure.
    
    This network has a uniform channel progression from in_channels to out_channels,
    with each block having a consistent structure.
    """
    
    def __init__(self, 
                 num_conv_blocks: int,
                 in_channels: int,
                 out_channels: int,
                 conv_layers_per_block: Union[int, List[int]],
                 channels: Optional[List[int]] = None,
                 kernel_sizes: Union[int, List[int]] = 3,
                 strides: Union[int, List[int]] = 1,
                 paddings: Union[int, str, List[Union[int, str]]] = 'same',
                 use_bn: Union[bool, List[bool]] = True,
                 activation_after_each_layer: Union[bool, List[bool]] = True,
                 activation: Optional[Union[nn.Module, List[nn.Module]]] = None,
                 activation_params: Optional[Union[dict, List[dict]]] = None,
                 final_bn_layer: Union[bool, List[bool]] = False,
                 force_residual: Union[bool, List[bool]] = True,
                 *args, **kwargs):
        """
        Initialize a UniformMultiResidualNet.
        
        Args:
            num_conv_blocks: Number of residual blocks in the network
            in_channels: Number of input channels
            out_channels: Number of output channels
            conv_layers_per_block: Number of conv layers in each residual block (int or list)
            channels: Optional list of channel dimensions (length must be num_conv_blocks + 1)
                     If None, a log-linear progression from in_channels to out_channels is used
            kernel_sizes: Kernel size(s) for each block. Can be an int, a list of ints (one per block),
                          or a list of lists (kernel sizes for each layer in each block).
            strides: Stride(s) for each block. Format options same as kernel_sizes.
            paddings: Padding value(s) for each block. Format options same as kernel_sizes.
            use_bn: Whether to use batch normalization in each block.
            activation_after_each_layer: Whether to apply activation after each conv layer in each block.
            activation: Activation function to use for each block.
            activation_params: Parameters for the activation function for each block.
            final_bn_layer: Whether to add a batch norm layer at the end of each block.
            force_residual: Whether to force residual connections in each block.
        """
        super().__init__('_block', *args, **kwargs)
        
        # Store basic parameters
        self.num_conv_blocks = num_conv_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Process conv_layers_per_block
        if isinstance(conv_layers_per_block, (list, tuple)) and len(conv_layers_per_block) != num_conv_blocks:
            raise ValueError(f"Expected {num_conv_blocks} values in conv_layers_per_block, got {len(conv_layers_per_block)}")

        self.conv_layers_per_block = [conv_layers_per_block] * num_conv_blocks  if isinstance(conv_layers_per_block, int) else conv_layers_per_block

        # Process channels
        if channels is None:
            # Generate log-linear channel progression
            self.channels = compute_log_linear_sequence(in_channels, out_channels, num_conv_blocks + 1)
        else:
            # Validate provided channels
            if len(channels) != num_conv_blocks + 1:
                raise ValueError(f"Expected {num_conv_blocks + 1} values in channels, got {len(channels)}")
            if channels[0] != in_channels:
                raise ValueError(f"First channel must be {in_channels}, got {channels[0]}")
            if channels[-1] != out_channels:
                raise ValueError(f"Last channel must be {out_channels}, got {channels[-1]}")
            self.channels = channels
        
        # Process other parameters - convert to lists if they're single values
        self.kernel_sizes = self._normalize_parameter(kernel_sizes, num_conv_blocks, "kernel_sizes")
        self.strides = self._normalize_parameter(strides, num_conv_blocks, "strides")
        self.paddings = self._normalize_parameter(paddings, num_conv_blocks, "paddings")
        self.use_bn = self._normalize_parameter(use_bn, num_conv_blocks, "use_bn")
        self.activation_after_each_layer = self._normalize_parameter(activation_after_each_layer, num_conv_blocks, "activation_after_each_layer")
        self.activation = self._normalize_parameter(activation, num_conv_blocks, "activation")
        self.activation_params = self._normalize_parameter(activation_params, num_conv_blocks, "activation_params")
        self.final_bn_layer = self._normalize_parameter(final_bn_layer, num_conv_blocks, "final_bn_layer")
        self.force_residual = self._normalize_parameter(force_residual, num_conv_blocks, "force_residual")
        
        # Create the residual blocks
        blocks = [None for _ in range(num_conv_blocks)]
        for i in range(num_conv_blocks):
            # Each residualConvBlock expects a list of channels of length num_conv_layers + 1 
            # the first num_conv_layers will be channels[i] and the last one will be channels[i+1]  
            block_channels = [self.channels[i]] * self.conv_layers_per_block[i] + [self.channels[i+1]]

            block = ResidualConvBlock(
                num_conv_layers=self.conv_layers_per_block[i],
                channels=block_channels,
                kernel_sizes=self.kernel_sizes[i],
                strides=self.strides[i],
                paddings=self.paddings[i],
                use_bn=self.use_bn[i],
                activation_after_each_layer=self.activation_after_each_layer[i],
                activation=self.activation[i],
                activation_params=self.activation_params[i],
                final_bn_layer=self.final_bn_layer[i],
                force_residual=self.force_residual[i]
            )
            blocks[i] = block
        
        # Store blocks as a sequential module
        self._block = nn.Sequential(*blocks)
    
    def _normalize_parameter(self, param, length: int, param_name: str):
        """
        Normalize a parameter to a list of the specified length.
        
        Args:
            param: The parameter to normalize (can be a single value or a list)
            length: The desired length of the output list
            param_name: Name of the parameter (for error messages)
            
        Returns:
            A list of length `length` containing the parameter values
        """
        if param is None:
            return [None] * length
        
        if not isinstance(param, list):
            # Single value - repeat for all blocks
            return [param] * length
        
        if len(param) == 1:
            # Single-element list - repeat the value
            return param * length
        
        if len(param) != length:
            raise ValueError(f"Parameter '{param_name}' has {len(param)} elements, expected {length}")
        
        return param
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self._block(x)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
