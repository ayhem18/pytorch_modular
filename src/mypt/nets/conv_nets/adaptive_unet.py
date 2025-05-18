import torch

from torch import nn
from typing import List, Tuple, Optional, Union

from mypt.nets.conv_nets.unet_skip_connections import UnetSkipConnections
from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser
from mypt.nets.conv_nets.uni_multi_residual_conv import UniformMultiResidualNet
from mypt.building_blocks.conv_blocks.composite_blocks import ContractingBlock, ExpandingBlock


class AdaptiveUNet(torch.nn.Module):
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
        super().__init__(*args, **kwargs)
        
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
    
        # make sure the output shape takes the 'bottleneck_out_channels' into account
        self.bottleneck_output_shape = list(bottleneck_shape)
        self.bottleneck_output_shape[0] = bottleneck_out_channels or bottleneck_shape[0]
        self.bottleneck_output_shape = tuple(self.bottleneck_output_shape)

        # These will be set by the build methods
        self.contracting_path = None
        self.bottleneck = None
        self.expanding_path = None
        self.skip_connections = None
        self._is_built = False

        self.contracting_start_point = None
    
    def build_contracting_path(self, 
                              max_conv_layers_per_block: int = 5,
                              min_conv_layers_per_block: int = 1,
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
            output_shape=self.bottleneck_input_shape,
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
            in_channels=self.bottleneck_input_shape[0],
            out_channels=self.bottleneck_output_shape[0],
            conv_layers_per_block=conv_layers_per_block,
            kernel_sizes=kernel_sizes,
            strides=1,
            paddings='same',
            input_shape=self.bottleneck_input_shape,
            **kwargs
        )
        
        return self
    
    def build_expanding_path(self,
                            max_conv_layers_per_block: int = 5,
                            min_conv_layers_per_block: int = 1,
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
        
        # Create the expanding path
        self.expanding_path = ExpandingBlock(
            input_shape=self.bottleneck_output_shape,
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
        num_contracting_blocks = len(self.contracting_path)
        num_expanding_blocks = len(self.expanding_path)
    
        # Number of skip connections is the minimum of the two
        num_skip_connections = min(num_contracting_blocks, num_expanding_blocks)
        
        # the skip connections need to link the the last N blocks of the contracting path to the first N inputs of the expanding path
        # in other words (self.bottleneck_output_shape) + the outputs of the (N - 1) first blocks of the expanding path
        
        # step2: find the N - 1 output shapes of the expanding path

        expanding_shapes = [self.bottleneck_output_shape] + [None for _ in range(num_skip_connections - 1)]

        # keep in mind that the input to the expanding path is the bottleneck output
        exp_shape = (2,) + self.bottleneck_output_shape

        dim_analyzer = DimensionsAnalyser(method='static')

        for i in range(num_skip_connections - 1):
            exp_shape = dim_analyzer.analyse_dimensions(exp_shape, self.expanding_path[i])
            expanding_shapes[i + 1] = exp_shape[1:]


        # step3: find the output shapes of the last N blocks of the contracting path
        con_shape = (2,) + self.input_shape

        contracting_shapes = [None for _ in range(num_skip_connections)]

        contracting_start_point = len(self.contracting_path) - num_skip_connections        
        
        self.contracting_start_point = contracting_start_point

        for index in range(len(self.contracting_path)):
            # there are two cases: either we reached the starting point
            # in this case we need to save 2 shapes: the output after the convolutional part of the block (using with the skip connection)
            # and the output after the pooling part (used to calculated the correct shape)

            # calculate the shape after the convolutional part of the block
            con_shape = dim_analyzer.analyse_dimensions(con_shape, self.contracting_path[index].conv)
            
            if index >= contracting_start_point:
                # save the shape for the skip connection
                contracting_shapes[index - contracting_start_point] = con_shape[1:]

            # calculate the shape after the pooling part of the block (make sure to use the `shape` calculated after the convolutional part)
            con_shape = dim_analyzer.analyse_dimensions(con_shape, self.contracting_path[index].pool) 
            
        # shape at this point must be the same as the bottleneck_input_shape    
        assert con_shape[1:] == self.bottleneck_input_shape, "Make sure the calculation of the shapes through the contracting path is correct!!!"
        assert all(s is not None for s in contracting_shapes), "Make sure the calculation of the shapes through the contracting path is correct!!!"


        # step4: build the skip connections. This is still tricky, so let's approach it step by step 
        # contracting_shapes is a list of the outputs of the last N blocks of the contracting path 
        # contracting_shapes[i] is the output shape of contracting_starting_point + i block
        # contracting_shapes[i] must be matched with the i-th expanding block in the reverse order: expanding_shapes[num_skip_connections - 1 - i]
        

        # the skip connection between the n - 1 - i)th block of the contracting path and the i)th block of the expanding path 
        # is a combination of a (1 * 1 convolutional layer) mapping the number of channels from the contracting block to the expanding block 
        # and then an adaptive average pooling to match the spatial dimensions

        skip_connections = [nn.Sequential(
            nn.Conv2d(contracting_shapes[i][0], expanding_shapes[num_skip_connections - 1 - i][0], kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(expanding_shapes[num_skip_connections - 1 - i][1:])
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
            skip_connection_index = len(contracting_outputs) - 1 - i
            skip_input = self.skip_connections.apply_connection(contracting_outputs[skip_connection_index], skip_connection_index)

            # pass the current output + skip_input through the expanding block
            x = self.expanding_path[i].forward(x + skip_input)    

        # at this point either the expanding or the contracting path is used
        if num_skip_connections == len(self.expanding_path.blocks):
            # at this point the expanding path is used; nothing left to do
            return x

        # at this point, we have still some expanding blocks left to use
        for i in range(num_skip_connections, len(self.expanding_path)):
            x = self.expanding_path[i].forward(x)

        return x

    def train(self, mode: bool = True) -> 'AdaptiveUNet':
        """
        Set the UNet and all its components to training mode.
        
        Args:
            mode: Boolean indicating whether to set to training mode (True) 
                 or evaluation mode (False)
                 
        Returns:
            Self for method chaining
        """
        # Set the training attribute of the model itself
        self.training = mode
        
        # Set training mode for each component if they've been built
        if self.contracting_path is not None:
            self.contracting_path.train(mode)
            
        if self.bottleneck is not None:
            self.bottleneck.train(mode)
            
        if self.expanding_path is not None:
            self.expanding_path.train(mode)
            
        if self.skip_connections is not None:
            self.skip_connections.train(mode)
            
        return self
    
    def eval(self) -> 'AdaptiveUNet':
        """
        Set the UNet and all its components to evaluation mode.
        
        Returns:
            Self for method chaining
        """
        return self.train(mode=False)
    
    def to(self, *args, **kwargs) -> 'AdaptiveUNet':
        """
        Move the UNet and all its components to the specified device or dtype.
        
        Args:
            *args, **kwargs: Arguments to pass to torch.nn.Module.to()
            
        Returns:
            Self for method chaining
        """
        # Move each component if they've been built
        if self.contracting_path is not None:
            self.contracting_path.to(*args, **kwargs)
            
        if self.bottleneck is not None:
            self.bottleneck.to(*args, **kwargs)
            
        if self.expanding_path is not None:
            self.expanding_path.to(*args, **kwargs)
            
        if self.skip_connections is not None:
            self.skip_connections.to(*args, **kwargs)
            
        return self

    def parameters(self, recurse: bool = True):
        """
        Return an iterator over UNet's parameters.
        
        Args:
            recurse: If True, yields parameters of this module and all submodules.
                    Otherwise, yields only parameters that are direct members of this module.
                    
        Returns:
            Iterator over parameters
        """
        # First, check if components are built
        if not self._is_built:
            # If no components are built, return empty iterator
            return iter([])
            
        # Yield parameters from each component that has been built
        for param in self.contracting_path.parameters(recurse):
            yield param

        for param in self.bottleneck.parameters(recurse):
            yield param
                
        for param in self.expanding_path.parameters(recurse):
            yield param
                
        for param in self.skip_connections.parameters(recurse):
            yield param
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """
        Return an iterator over UNet's parameters, yielding both the name and the parameter.
        
        Args:
            prefix: Prefix to prepend to all parameter names
            recurse: If True, yields parameters of this module and all submodules.
                    Otherwise, yields only parameters that are direct members of this module.
                    
        Returns:
            Iterator over (name, parameter) tuples
        """
        if not self._is_built:
            return iter([])
            
        # self._is_build == True, implies all the components are initialized

        # Prepare component prefixes
        cp_prefix = f"{prefix}.contracting_path." if prefix else "contracting_path."
        bn_prefix = f"{prefix}.bottleneck." if prefix else "bottleneck."
        ep_prefix = f"{prefix}.expanding_path." if prefix else "expanding_path."
        sc_prefix = f"{prefix}.skip_connections." if prefix else "skip_connections."
        
        # Yield named parameters from each component that has been built
        for name, param in self.contracting_path.named_parameters(recurse=recurse):
            yield cp_prefix + name, param
                
        for name, param in self.bottleneck.named_parameters(recurse=recurse):
            yield bn_prefix + name, param
                
        for name, param in self.expanding_path.named_parameters(recurse=recurse):
            yield ep_prefix + name, param
                
        for name, param in self.skip_connections.named_parameters(recurse=recurse):
            yield sc_prefix + name, param
    
    def modules(self):
        """
        Return an iterator over all modules in the UNet network.
        
        Returns:
            Iterator over modules
        """
        # Yield self first, as per PyTorch convention
        yield self
        
        # Yield modules from each component that has been built
        if self.contracting_path is not None:
            for module in self.contracting_path.modules():
                yield module
                
        if self.bottleneck is not None:
            for module in self.bottleneck.modules():
                yield module
                
        if self.expanding_path is not None:
            for module in self.expanding_path.modules():
                yield module
                
        if self.skip_connections is not None:
            for module in self.skip_connections.modules():
                yield module


    def children(self):
        """
        Return an iterator over all children modules in the UNet network.
        
        Returns:
            Iterator over children modules
        """        
        if not self._is_built:
            return iter([])

        return iter([self.contracting_path, self.bottleneck, self.expanding_path, self.skip_connections])
                
    
    def named_children(self):
        """
        Return an iterator over all named children modules in the UNet network.
        
        Returns:
            Iterator over (name, module) tuples
        """
        if not self._is_built:
            return iter([])

        return iter([
            ("contracting_path", self.contracting_path),
            ("bottleneck", self.bottleneck),
            ("expanding_path", self.expanding_path),
            ("skip_connections", self.skip_connections)
        ])



    # def state_dict(self):
    #     """
    #     Return a dictionary containing the model's state.
        
    #     Returns:
    #         Dictionary containing the model's state
    #     """
    #     if not self._is_built:
    #         raise RuntimeError("Model must be built before saving. Call build() first.")

    #     state_dict = super().state_dict()

    #     # Add the is_built attribute to the state dict
    #     state_dict['is_built'] = self._is_built

    #     state_dict['input_shape'] = self.input_shape
    #     state_dict['output_shape'] = self.output_shape
    #     state_dict['bottleneck_input_shape'] = self.bottleneck_input_shape
    #     state_dict['bottleneck_output_shape'] = self.bottleneck_output_shape

    #     state_dict['contracting_path'] = self.contracting_path.state_dict()
    #     state_dict['bottleneck'] = self.bottleneck.state_dict()
    #     state_dict['expanding_path'] = self.expanding_path.state_dict()
    #     state_dict['skip_connections'] = self.skip_connections.state_dict()

    #     return state_dict



    # def save_model(self, path: str) -> None:
    #     """
    #     Save the model state dict to the specified path.
        
    #     Args:
    #         path: Path where the model state dict will be saved
            
    #     Raises:
    #         RuntimeError: If the model hasn't been built before saving
    #     """
    #     if not self._is_built:
    #         raise RuntimeError("Model must be built before saving. Call build() first.")
        
    #     # Save model state dict with metadata
    #     model_state = {
    #         'model_state_dict': self.state_dict(),
    #         'is_built': True,
    #         'input_shape': self.input_shape,
    #         'output_shape': self.output_shape,
    #         'bottleneck_input_shape': self.bottleneck_input_shape,
    #         'bottleneck_output_shape': self.bottleneck_output_shape
    #     }
        
    #     torch.save(model_state, path)
    
    # def load_state_dict(self, state_dict: dict)     -> None:
    #     """
    #     Load the model's state from a dictionary.
        
    #     Args:
    #         state_dict: Dictionary containing the mo    del's state
    #     """
    #     # call the parent's load_state_dict (load all fields I did not override)
    #     super().load_state_dict(state_dict)

    #     # Extract the is_built attribute from the state dict
    #     self._is_built = state_dict.get('is_built', False)

    #     # make sure the parameters are the same as the ones in the state_dict
    #     if self.input_shape != state_dict.get('input_shape', None):
    #         raise ValueError("The input shape is not the same as the one in the state dict.")       

    #     if self.output_shape != state_dict.get('output_shape', None):
    #         raise ValueError("The output shape is not the same as the one in the state dict.")

    #     if self.bottleneck_input_shape != state_dict.get('bottleneck_input_shape', None):
    #         raise ValueError("The bottleneck input shape is not the same as the one in the state dict.")

    #     if self.bottleneck_output_shape != state_dict.get('bottleneck_output_shape', None):
    #         raise ValueError("The bottleneck output shape is not the same as the one in the state dict.")

    #     # load the state dict of the components
    #     self.contracting_path.load_state_dict(state_dict.get('contracting_path'))
    #     self.bottleneck.load_state_dict(state_dict.get('bottleneck'))
    #     self.expanding_path.load_state_dict(state_dict.get('expanding_path'))
    #     self.skip_connections.load_state_dict(state_dict.get('skip_connections'))   
