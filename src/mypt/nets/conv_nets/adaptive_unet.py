import torch
from torch import nn
from typing import List, Tuple, Optional, Union, Dict

from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser
from mypt.building_blocks.mixins.custom_module_mixins import WrapperLikeModuleMixin
from mypt.building_blocks.conv_blocks.composite_blocks import ContractingBlock, ExpandingBlock
from mypt.nets.conv_nets.uni_multi_residual_conv import UniformMultiResidualNet
from mypt.nets.conv_nets.unet_skip_connections import UnetSkipConnections


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
        self.bottleneck_shape = bottleneck_shape
        self.bottleneck_out_channels = bottleneck_out_channels or bottleneck_shape[0]
        
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
            
        # Get the number of blocks in each path
        num_contracting_blocks = len(self.contracting_path.blocks)
        num_expanding_blocks = len(self.expanding_path.blocks)
        
        # Number of skip connections is the minimum of the two
        num_skip_connections = min(num_contracting_blocks, num_expanding_blocks)
        
        # Initialize skip connections
        self.skip_connections = UnetSkipConnections()
        
        # Use DimensionsAnalyzer to find the shapes at each level
        dim_analyzer = DimensionsAnalyser(method='static')
        
        # Get the intermediate output shapes from the contracting path
        # We use a dummy batch dimension (2) for the analysis
        x = torch.zeros((2,) + self.input_shape)
        contracting_shapes = []
        
        # Get intermediate shapes from the contracting path
        for i in range(num_contracting_blocks):
            block = self.contracting_path.blocks[i]
            # Use dimension analyzer to get output shape without running the model
            shape = dim_analyzer.analyse_dimensions(x.shape, block)
            contracting_shapes.append(shape[1:])  # Remove batch dimension
            x = torch.zeros(shape)  # Update x for next block
        
        # Get intermediate shapes from the expanding path (in reverse order)
        # Start with the bottleneck output
        bottleneck_output_shape = dim_analyzer.analyse_dimensions(
            torch.zeros((2,) + self.bottleneck_shape).shape, 
            self.bottleneck
        )
        
        x = torch.zeros(bottleneck_output_shape)
        expanding_shapes = []
        
        for i in range(num_expanding_blocks):
            block = self.expanding_path.blocks[i]
            shape = dim_analyzer.analyse_dimensions(x.shape, block)
            expanding_shapes.append(shape[1:])  # Remove batch dimension
            x = torch.zeros(shape)  # Update x for next block
        
        # Create skip connections for matching levels
        # We connect contracting blocks to expanding blocks in reverse order
        for i in range(num_skip_connections):
            contracting_idx = i
            expanding_idx = num_expanding_blocks - 1 - i
            
            if expanding_idx < 0:
                continue  # Skip if we've run out of expanding blocks
            
            contracting_shape = contracting_shapes[contracting_idx]
            expanding_input_shape = expanding_shapes[expanding_idx]
            
            # Check if spatial dimensions match
            if contracting_shape[1:] != expanding_input_shape[1:]:
                # Need adaptive pooling to match spatial dimensions
                self.skip_connections.add_connection(
                    in_channels=contracting_shape[0],
                    out_channels=expanding_input_shape[0]
                )
            else:
                # Simple 1x1 conv or identity for channel matching
                self.skip_connections.add_connection(
                    in_channels=contracting_shape[0],
                    out_channels=expanding_input_shape[0]
                )
        
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
        
        # Compute how many skip connections we're using
        num_skip_connections = self.skip_connections.num_connections
        
        # Prepare for expanding path
        # We need to reverse contracting_outputs to match expanding path
        contracting_outputs.reverse()
        
        # Pass through expanding path with skip connections
        # Each block in expanding path gets a skip connection from contracting path
        x_expanding, expanding_outputs = self.expanding_path(x, full=True)
        
        # Apply skip connections
        final_output = x_expanding
        
        for i in range(min(num_skip_connections, len(expanding_outputs))):
            # Get expanding block output
            expanding_output = expanding_outputs[i]
            
            # Get corresponding contracting output
            if i < len(contracting_outputs):
                contracting_output = contracting_outputs[i]
                
                # Apply skip connection
                skip_output = self.skip_connections.apply_connection(contracting_output, i)
                
                # Add skip output to expanding output
                # We can only add if spatial dimensions match
                if skip_output.shape[2:] == expanding_output.shape[2:]:
                    final_output = expanding_output + skip_output
                else:
                    # Use adaptive pooling to match dimensions
                    skip_output = nn.functional.adaptive_avg_pool2d(
                        skip_output, 
                        output_size=expanding_output.shape[2:]
                    )
                    final_output = expanding_output + skip_output
        
        return final_output