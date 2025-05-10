import torch

from torch import nn
from copy import copy
from typing import List, Tuple, Optional, Union

from mypt.nets.conv_nets.unet_skip_connections import UnetSkipConnections
from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser
from mypt.nets.conv_nets.uni_multi_residual_conv import UniformMultiResidualNet
from mypt.building_blocks.mixins.custom_module_mixins import WrapperLikeModuleMixin
from mypt.building_blocks.conv_blocks.composite_blocks import ContractingBlock, ExpandingBlock


class AdaptiveUNet(WrapperLikeModuleMixin):
    """
    A flexible UNet architecture that adapts to input and output shapes.
    
    This implementation uses the builder pattern to construct the network in
    separate steps, allowing for greater flexibility and control over the UNet
    architecture. The network consists of three main components:
    
    1. Contracting path: Reduces spatial dimensions to the bottleneck
    2. Bottleneck: Processes features at the smallest spatial dimension
    3. Expanding path: Increases spatial dimensions back to the output shape
    
    Skip connections link corresponding levels of the contracting and expanding paths.
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int],
                 output_shape: Tuple[int, int, int],
                 bottleneck_shape: Tuple[int, int, int],
                 bottleneck_out_channels: Optional[int] = None,
                 *args, **kwargs):
        """
        Initialize the AdaptiveUNet.
        
        Args:
            input_shape: Tuple of (channels, height, width) for the input
            output_shape: Tuple of (channels, height, width) for the output
            bottleneck_shape: Tuple of (channels, height, width) for the bottleneck
            bottleneck_out_channels: Number of output channels from bottleneck (defaults to bottleneck_shape[0])
        """
        super().__init__('_unet', *args, **kwargs)
        
        # Validate shapes
        if len(input_shape) != 3 or len(output_shape) != 3 or len(bottleneck_shape) != 3:
            raise ValueError("All shapes must be 3D tuples (channels, height, width)")
            
        # Validate spatial dimensions
        if (bottleneck_shape[1] > input_shape[1] or bottleneck_shape[2] > input_shape[2] or
            bottleneck_shape[1] > output_shape[1] or bottleneck_shape[2] > output_shape[2]):
            raise ValueError("Bottleneck spatial dimensions must be smaller than both input and output dimensions")
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.bottleneck_input_shape = bottleneck_shape
        self.bottleneck_output_shape = copy(bottleneck_shape)
        self.bottleneck_output_shape[0] = bottleneck_out_channels or bottleneck_shape[0]
        
        # These will be set by the build methods
        self.contracting_path = None
        self.bottleneck = None
        self.expanding_path = None
        self.skip_connections = None
        self._is_built = False
    
    def build_contracting_path(self, 
                              max_conv_layers_per_block: int = 4,
                              min_conv_layers_per_block: int = 2,
                              **kwargs) -> 'AdaptiveUNet':
        """
        Build the contracting path of the UNet.
        
        Args:
            max_conv_layers_per_block: Maximum number of conv layers per block
            min_conv_layers_per_block: Minimum number of conv layers per block
            **kwargs: Additional arguments to pass to ContractingBlock
            
        Returns:
            Self for method chaining
        """
        # Check if already built
        if self.contracting_path is not None:
            return self
            
        # Create the contracting path
        self.contracting_path = ContractingBlock(
            input_shape=self.input_shape,
            output_shape=self.bottleneck_shape,
            max_conv_layers_per_block=max_conv_layers_per_block,
            min_conv_layers_per_block=min_conv_layers_per_block,
            **kwargs
        )
        
        return self
    
    def build_bottleneck(self, 
                        kernel_sizes: Union[int, List[int]],
                        num_blocks: int,
                        conv_layers_per_block: Union[int, List[int]] = 2,
                        **kwargs) -> 'AdaptiveUNet':
        """
        Build the bottleneck of the UNet.
        
        Args:
            kernel_sizes: Kernel size(s) for the bottleneck convolutions
            num_blocks: Number of residual blocks in the bottleneck
            conv_layers_per_block: Number of conv layers per residual block
            **kwargs: Additional arguments to pass to UniformMultiResidualNet
            
        Returns:
            Self for method chaining
        """
        # Check if contracting path is built
        if self.contracting_path is None:
            raise ValueError("Contracting path must be built before bottleneck")
            
        # Check if already built
        if self.bottleneck is not None:
            return self
            
        # Create the bottleneck
        self.bottleneck = UniformMultiResidualNet(
            num_conv_blocks=num_blocks,
            in_channels=self.bottleneck_shape[0],
            out_channels=self.bottleneck_out_channels,
            conv_layers_per_block=conv_layers_per_block,
            kernel_sizes=kernel_sizes,
            strides=1,
            paddings='same',
            input_shape=self.bottleneck_shape,
            **kwargs
        )
        
        return self
    
    def build_expanding_path(self,
                            max_conv_layers_per_block: int = 4,
                            min_conv_layers_per_block: int = 2,
                            **kwargs) -> 'AdaptiveUNet':
        """
        Build the expanding path of the UNet.
        
        Args:
            max_conv_layers_per_block: Maximum number of conv layers per block
            min_conv_layers_per_block: Minimum number of conv layers per block
            **kwargs: Additional arguments to pass to ExpandingBlock
            
        Returns:
            Self for method chaining
        """
        # Check if previous components are built
        if self.contracting_path is None or self.bottleneck is None:
            raise ValueError("Contracting path and bottleneck must be built before expanding path")
            
        # Check if already built
        if self.expanding_path is not None:
            return self
        
        # Create a starting shape for the expanding path - this will come from the bottleneck
        bottleneck_output_shape = (self.bottleneck_out_channels,) + self.bottleneck_shape[1:]
        
        # Create the expanding path
        self.expanding_path = ExpandingBlock(
            input_shape=bottleneck_output_shape,
            output_shape=self.output_shape,
            max_conv_layers_per_block=max_conv_layers_per_block,
            min_conv_layers_per_block=min_conv_layers_per_block,
            **kwargs
        )
        
        return self



    def _set_skip_connections(self) -> None:
        """
        Set the skip connections for the UNet.
        """
        
        # step1: find the number of blocks in each path
        num_contracting_blocks = len(self.contracting_path.blocks)
        num_expanding_blocks = len(self.expanding_path.blocks)
    
        # Number of skip connections is the minimum of the two
        num_skip_connections = min(num_contracting_blocks, num_expanding_blocks)
        
        # the skip connections need to link the the last N blocks of the contracting path to the first N blocks of the expanding path
        
        # step2: find the output shapes of the first N blocks of the expanding path

        expanding_shapes = [None for _ in range(num_skip_connections)]

        # keep in mind that the input to the expanding path is the bottleneck output
        traversal_input = torch.zeros((2,) + self.bottleneck_output_shape)

        dim_analyzer = DimensionsAnalyser(method='static')

        for i in range(num_skip_connections):
            shape = dim_analyzer.analyse_dimensions(traversal_input.shape, self.expanding_path[i])
            expanding_shapes[i] = shape[1:]
            traversal_input = torch.zeros(shape)

        # step3: find the output shapes of the last N blocks of the contracting path
        contracting_shapes = [None for _ in range(num_skip_connections)]

        for i in range(num_skip_connections):
            # the only part we really need is the number of output channels
            shape = dim_analyzer.analyse_dimensions(traversal_input.shape, self.contracting_path[num_contracting_blocks - 1 - i].conv)
            contracting_shapes[i] = shape[1:]
            traversal_input = torch.zeros(shape)

        # the skip connection between the n - 1 - i)th block of the contracting path and the i)th block of the expanding path 
        # is a combination of a (1 * 1 convolutional layer) mapping the number of channels from the contracting block to the expanding block 
        # and then an adaptive average pooling to match the spatial dimensions

        skip_connections = [nn.Sequential(
            nn.Conv2d(contracting_shapes[i][0], expanding_shapes[i][0], kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(expanding_shapes[i][1:])
        ) for i in range(num_skip_connections)]

        self.skip_connections = UnetSkipConnections(skip_connections)


    def build(self) -> 'AdaptiveUNet':
        """
        Build the complete UNet architecture with skip connections.
        
        This method connects the contracting path, bottleneck, and expanding path,
        and creates the skip connections between corresponding levels.
        
        Returns:
            Self for method chaining
        """
        # Check if all components are ready
        if self.contracting_path is None or self.bottleneck is None or self.expanding_path is None:
            raise ValueError("All components (contracting_path, bottleneck, expanding_path) must be built before calling build()")
        
        if self._is_built:
            return self

        self._set_skip_connections()
        
        self._is_built = True
        return self
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the UNet.
        
        This method handles the complete forward pass through all components,
        including skip connections.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        if not self._is_built:
            raise RuntimeError("UNet must be built before forward pass. Call build() first.")
            
        # Verify input shape
        if x.shape[1:] != self.input_shape:
            raise ValueError(f"Input shape must be {self.input_shape}, got {x.shape[1:]}")
        
        # Get intermediate outputs from contracting path
        x, contracting_outputs = self.contracting_path(x, full=True)
        
        # Pass through bottleneck
        x = self.bottleneck(x)

        # 'x' at this point is the bottleneck output
        # for each block in the expanding path, we need to pass the current output + skipconnection(intermediate contracting block output)


        num_skip_connections = self.skip_connections.num_connections 

        for i in range(num_skip_connections):
            # pass the corrresponding intermediate output through the skip connection
            skip_input = self.skip_connections.apply_connection(contracting_outputs[i], i)

            # pass the current output + skip_input through the expanding block
            x = self.expanding_path[i](x, skip_input)    

        # at this point either the expanding or the contracting path is used
        if num_skip_connections == len(self.expanding_path.blocks):
            # at this point the expanding path is used; nothing left to do
            return x

        # at this point, we have still some expanding blocks left to use
        for i in range(num_skip_connections, len(self.expanding_path.blocks)):
            x = self.expanding_path[i](x)

        return x

