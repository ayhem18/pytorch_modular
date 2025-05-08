"""
This module contains the ContractingCbDesigner class for designing convolutional blocks
that reduce dimensions from input_shape to output_shape.
"""

from torch import nn
from typing import Dict, List, OrderedDict, Tuple


from mypt.building_blocks.conv_blocks.conv_block import BasicConvBlock
from mypt.building_blocks.conv_blocks.conv_block_design.contracting_helper import best_conv_block
from mypt.building_blocks.conv_blocks.conv_block_design.conv_design_utils import compute_log_linear_sequence


class ContractingCbDesigner:
    """
    Designer class for creating optimal convolutional blocks that reduce spatial dimensions.
    
    This class takes an input shape and output shape, then designs convolutional blocks
    that optimally reduce the dimensions using the contracting_helper module.
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int],
                 output_shape: Tuple[int, int, int],
                 max_conv_layers_per_block: int = 4,
                 min_conv_layers_per_block: int = 2):
        """
        Initialize the ContractingCbDesigner.
        
        Args:
            input_shape: Tuple[int, int, int] - Input shape (channels, height, width)
            output_shape: Tuple[int, int, int] - Output shape (channels, height, width)
            max_conv_layers_per_block: Maximum number of convolutional layers per block
            min_conv_layers_per_block: Minimum number of convolutional layers per block
        """
        # Validate input dimensions
        if input_shape[1] <= output_shape[1] or input_shape[2] <= output_shape[2]:
            raise ValueError("Input spatial dimensions must be larger than output spatial dimensions. The shapes are expected as (channels, height, width).")

        # Store input parameters
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.max_conv_layers_per_block = max_conv_layers_per_block
        self.min_conv_layers_per_block = min_conv_layers_per_block
                
        self.height_block: List[Dict] = []
        self.width_block: List[Dict] = []
        self.merged_blocks: List[nn.Sequential] = []
        
    def _design_blocks(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Design the height and width reduction blocks.
        
        Returns:
            Tuple of (height_block, width_block) where each block is a list of layer dicts
        """
        # Check if height and width are the same
        if self.input_shape[1] == self.input_shape[2] and self.output_shape[1] == self.output_shape[2]:
            # For equal dimensions, compute only once
            height_block, _ = best_conv_block(
                self.input_shape[1], 
                self.output_shape[1],
                self.min_conv_layers_per_block,
                self.max_conv_layers_per_block
            )
            return height_block, height_block.copy()
        
        # Otherwise compute separately for height and width
        height_block, _ = best_conv_block(
            self.input_shape[1], 
            self.output_shape[1],
            self.min_conv_layers_per_block,
            self.max_conv_layers_per_block
        )
        
        width_block, _ = best_conv_block(
            self.input_shape[2], 
            self.output_shape[2],
            self.min_conv_layers_per_block,
            self.max_conv_layers_per_block
        )
        
        if height_block is None:
            raise ValueError("The internal algorithm failed to find a valid convolutional block mapping the input height to the output height with the given parameters. Consider increasing the max_conv_layers_per_block parameter.")

        if width_block is None:
            raise ValueError("The internal algorithm failed to find a valid convolutional block mapping the input width to the output width with the given parameters. Consider increasing the max_conv_layers_per_block parameter.")

        return height_block, width_block
    
    def _split_into_sub_blocks(self, block: List[Dict]) -> List[List[Dict]]:
        """
        Split a block into sub-blocks based on layer types.
        A sub-block is defined as a sequence of convolutions followed by a pooling layer.
        
        Args:
            block: List of layer dictionaries from contracting_helper
            
        Returns:
            List of sub-blocks, where each sub-block is a list of layer dicts
        """
        sub_blocks = []
        current_sub_block = []
        
        for layer in block:
            if layer["type"] == "conv":
                current_sub_block.append(layer)
            elif layer["type"] == "pool":
                # Add pool layer to current sub-block and start a new sub-block
                current_sub_block.append(layer)
                sub_blocks.append(current_sub_block)
                current_sub_block = []
        
        # Add any remaining layers as the final sub-block
        if current_sub_block:
            sub_blocks.append(current_sub_block)
            
        return sub_blocks
    
    def _equalize_num_sub_blocks(self, 
                                height_sub_blocks: List[List[Dict]], 
                                width_sub_blocks: List[List[Dict]]) -> Tuple[List[List[Dict]], List[List[Dict]]]:
        """
        Equalize the number of sub-blocks between height and width blocks.
        
        Args:
            height_sub_blocks: List of sub-blocks for height reduction
            width_sub_blocks: List of sub-blocks for width reduction
            
        Returns:
            Tuple of equalized height and width sub-blocks
        """
        if len(height_sub_blocks) == len(width_sub_blocks):
            return height_sub_blocks, width_sub_blocks
        
        # Determine which has more sub-blocks
        if len(height_sub_blocks) > len(width_sub_blocks):
            larger = height_sub_blocks
            smaller = width_sub_blocks
            is_height_larger = True
        else:
            larger = width_sub_blocks
            smaller = height_sub_blocks
            is_height_larger = False
        
        # Add equivalent sub-blocks with kernel size 1 to the smaller list
        difference = len(larger) - len(smaller)
        for i in range(difference):
            # Create a new sub-block equivalent to larger[i] but with kernel size 1
            reference_block = larger[i]
            new_sub_block = []
            
            for layer in reference_block:
                if layer["type"] == "conv":
                    # Add conv layer with kernel size 1 (doesn't change dimensions)
                    new_sub_block.append({"type": "conv", "kernel_size": 1, "stride": 1})
                elif layer["type"] == "pool":
                    # Add identity pooling (doesn't change dimensions)
                    new_sub_block.append({"type": "pool", "kernel_size": 1, "stride": 1})
            
            smaller.append(new_sub_block)
        
        # Return in the correct order
        return (larger, smaller) if is_height_larger else (smaller, larger)
    
    def _equalize_sub_block_layers(self, 
                                   height_sub_block: List[Dict], 
                                   width_sub_block: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Equalize the number of layers within corresponding sub-blocks.
        
        Args:
            height_sub_block: A sub-block for height reduction
            width_sub_block: A sub-block for width reduction
            
        Returns:
            Tuple of equalized height and width sub-blocks
        """
        # Count conv layers in each sub-block
        height_conv_count = sum(1 for layer in height_sub_block if layer["type"] == "conv")
        width_conv_count = sum(1 for layer in width_sub_block if layer["type"] == "conv")
        
        # If equal, no changes needed
        if height_conv_count == width_conv_count:
            return height_sub_block, width_sub_block
        
        # Determine which has more layers
        if height_conv_count > width_conv_count:
            larger = height_sub_block
            smaller = width_sub_block
            is_height_larger = True
        else:
            larger = width_sub_block
            smaller = height_sub_block
            is_height_larger = False


        # although the initial idea was to add convolutional layers with the same kernel size as the last convolutional layer (in the smaller block)
        # this approach does not work, since a convolutional layer can have either "same" for dimensions or a numerical value for both dimensions: width and height
        # in other words, having a convolutional layer with a padding arguments (same, 0) or (0, same) raises an error.  
        # so we need to add a convolutional layer with a kernel size of 1 and a stride of 1 to the smaller block

        # Add padding layers to the smaller block
        difference = abs(height_conv_count - width_conv_count)
        result_smaller = smaller[:-1].copy()
        
        result_smaller.extend(
            ([{"type": "conv", "kernel_size": 1, "stride": 1}] * difference) # add convolutional layers with a kernel size 1, padding 0 and stride 1
              + 
            [smaller[-1]] # add the last convolutional layer of the smaller block
            )
        
        # Return in the correct order
        return (larger, result_smaller) if is_height_larger else (result_smaller, larger)
    
    def _set_kernel_sizes_and_paddings(self, 
                                       h_equalized: List[Dict], 
                                       w_equalized: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Set the kernel sizes and paddings for the height and width sub-blocks.
        """
        assert len(h_equalized) == len(w_equalized), "kernel sizes and paddings must be extracted from sub-blocks of the same size"

        # make sure there is exactly one pool layer in the height and width sub-blocks
        assert sum(1 for l in h_equalized if l["type"] == "pool") == 1 and sum(1 for l in w_equalized if l["type"] == "pool") == 1, "there must be exactly one pool layer in the height and width sub-blocks"
        # make sure the only pooly layer is the last one
        assert h_equalized[-1]["type"] == "pool" and w_equalized[-1]["type"] == "pool", "the last layer of the height and width sub-blocks must be a pool layer"

        # Extract kernel sizes
        h_kernel_sizes = [l["kernel_size"] for l in h_equalized[:-1]]
        w_kernel_sizes = [l["kernel_size"] for l in w_equalized[:-1]]

        # make sure to set the paddings correctly   
        h_paddings = [l.get('padding', 0) for l in h_equalized[:-1]]
        w_paddings = [l.get('padding', 0) for l in w_equalized[:-1]]

        paddings = [(h, w) for h, w in zip(h_paddings, w_paddings)]
        kernel_sizes = [(h, w) for h, w in zip(h_kernel_sizes, w_kernel_sizes)]
        return kernel_sizes, paddings   


    def _merge_blocks_singular(self, height_block: List[Dict], width_block: List[Dict]) -> List[nn.Sequential]:
        """
        Merge the height and width blocks where both of them have only one sub-block.
        """

        # Equalize layers within the sub-blocks
        h_equalized, w_equalized = self._equalize_sub_block_layers(height_block, width_block)
                
        # compute the channels
        channels = compute_log_linear_sequence(self.input_shape[0], self.output_shape[0], len(h_equalized)) # height block contains a pool layer at the end... (so passing the height is correct)

        kernel_sizes, paddings = self._set_kernel_sizes_and_paddings(h_equalized, w_equalized)

        conv = BasicConvBlock(
            num_conv_layers=len(kernel_sizes),
            channels=channels,
            kernel_sizes=kernel_sizes,
            paddings=paddings,
            use_bn=True
        )

        pool_layer = nn.AvgPool2d(h_equalized[-1]["kernel_size"], h_equalized[-1]["stride"])

        return [nn.Sequential(OrderedDict([("conv", conv), ("pool", pool_layer)]))]
        

    def _merge_blocks(self) -> List[nn.Sequential]:
        """
        Merge the height and width blocks into a single architecture.
        
        Returns:
            List of ModuleLists, where each ModuleList contains ConvBlock objects
        """
        # Split blocks into sub-blocks
        height_sub_blocks = self._split_into_sub_blocks(self.height_block)
        width_sub_blocks = self._split_into_sub_blocks(self.width_block)
        
        # Equalize the number of sub-blocks
        height_sub_blocks, width_sub_blocks = self._equalize_num_sub_blocks(
            height_sub_blocks, width_sub_blocks
        )
        
        if len(height_sub_blocks) == 1 and len(width_sub_blocks) == 1:
            return self._merge_blocks_singular(height_sub_blocks[0], width_sub_blocks[0])

        # compute the channels for each block.
        channels = compute_log_linear_sequence(self.input_shape[0], self.output_shape[0], len(height_sub_blocks))

        merged_blocks = []

        merged_blocks = [None for _ in range(len(height_sub_blocks))]
        
        # For each pair of sub-blocks
        for block_index, (h_sub_block, w_sub_block) in enumerate(zip(height_sub_blocks, width_sub_blocks)):
            # Equalize layers within the sub-blocks
            h_equalized, w_equalized = self._equalize_sub_block_layers(h_sub_block, w_sub_block)
            
            kernel_sizes, paddings = self._set_kernel_sizes_and_paddings(h_equalized, w_equalized) 

            block_channels = ([channels[block_index]] * len(kernel_sizes)) + [channels[min(block_index + 1, len(channels) - 1)]]

            assert len(block_channels) == len(kernel_sizes) + 1

            conv = BasicConvBlock(
                num_conv_layers=len(kernel_sizes),
                channels=block_channels,
                kernel_sizes=kernel_sizes,
                paddings=paddings,
                use_bn=True
            )

            # it is possible that the height and width sub-blocks have different kernel sizes and strides
            pool_layer = nn.AvgPool2d((h_equalized[-1]["kernel_size"], w_equalized[-1]["kernel_size"]), (h_equalized[-1]["stride"], w_equalized[-1]["stride"]))

            merged_blocks[block_index] = nn.Sequential(OrderedDict([("conv", conv), ("pool", pool_layer)]))

            
        return merged_blocks
    

    def get_contracting_block(self) -> List[nn.Sequential]:
        """
        Get the contracting block.
        """
        self.height_block, self.width_block = self._design_blocks()
        self.merged_blocks = self._merge_blocks()

        return self.merged_blocks

