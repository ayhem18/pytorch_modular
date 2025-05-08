"""
This module contains the ExpandingCbDesigner class for designing transpose convolutional blocks
that expand dimensions from input_shape to output_shape.
"""

from torch import nn
from typing import Dict, List, OrderedDict, Tuple

from mypt.building_blocks.conv_blocks.transpose_conv_block import TransposeConvBlock
from mypt.building_blocks.conv_blocks.conv_block_design.expanding_helper import best_transpose_conv_block, get_output_dim
from mypt.building_blocks.conv_blocks.conv_block_design.conv_design_utils import compute_log_linear_sequence


class ExpandingCbDesigner:
    """
    Designer class for creating optimal transpose convolutional blocks that expand spatial dimensions.
    
    This class takes an input shape and output shape, then designs transpose convolutional blocks
    that optimally expand the dimensions using the expanding_helper module.
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int],
                 output_shape: Tuple[int, int, int],
                 max_conv_layers_per_block: int = 4,
                 min_conv_layers_per_block: int = 2):
        """
        Initialize the ExpandingCbDesigner.
        
        Args:
            input_shape: Tuple[int, int, int] - Input shape (channels, height, width)
            output_shape: Tuple[int, int, int] - Output shape (channels, height, width)
            max_conv_layers_per_block: Maximum number of transpose convolutional layers per block
            min_conv_layers_per_block: Minimum number of transpose convolutional layers per block
        """
        # Validate output dimensions are larger than input dimensions
        if input_shape[1] >= output_shape[1] or input_shape[2] >= output_shape[2]:
            raise ValueError("Output spatial dimensions must be larger than input spatial dimensions. "
                            "The shapes are expected as (channels, height, width).")

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
        Design the height and width expansion blocks.
        
        Returns:
            Tuple of (height_block, width_block) where each block is a list of layer dicts
        """
        # Check if height and width are the same
        if self.input_shape[1] == self.input_shape[2] and self.output_shape[1] == self.output_shape[2]:
            # For equal dimensions, compute only once
            height_block, _ = best_transpose_conv_block(
                self.input_shape[1], 
                self.output_shape[1],
                self.min_conv_layers_per_block,
                self.max_conv_layers_per_block
            )
            return height_block, height_block.copy()
        
        # Otherwise compute separately for height and width
        height_block, _ = best_transpose_conv_block(
            self.input_shape[1], 
            self.output_shape[1],
            self.min_conv_layers_per_block,
            self.max_conv_layers_per_block
        )
        
        width_block, _ = best_transpose_conv_block(
            self.input_shape[2], 
            self.output_shape[2],
            self.min_conv_layers_per_block,
            self.max_conv_layers_per_block
        )
        
        return height_block, width_block
    
    def _split_into_sub_blocks(self, block: List[Dict]) -> List[List[Dict]]:
        """
        Split a block into sub-blocks based on layer types.
        A sub-block is defined as a sequence of transpose convolutions with stride=1
        followed by a transpose convolution with stride>1.
        
        Args:
            block: The block to split
            
        Returns:
            List of sub-blocks
        """
        sub_blocks = []
        current_sub_block = []
        
        for layer in block:
            # if the layer has a stride larger than 1 or it has a end_block flag, it is the end of a sub-block
            if layer["stride"] > 1 or layer.get("end_block", False):
                # Add the strided transpose conv to the current sub-block
                current_sub_block.append(layer)
                
                # Finalize the sub-block and start a new one
                sub_blocks.append(current_sub_block)
                current_sub_block = []
            else:
                # Regular transpose conv - add to current sub-block
                current_sub_block.append(layer)
        
        # Add the last sub-block if it exists
        if current_sub_block:
            sub_blocks.append(current_sub_block)
            
        return sub_blocks
    
    def _equalize_num_sub_blocks(self, height_sub_blocks: List[List[Dict]], 
                                width_sub_blocks: List[List[Dict]]) -> Tuple[List[List[Dict]], List[List[Dict]]]:
        """
        Ensure that height and width blocks have the same number of sub-blocks.
        If they don't, add identity sub-blocks (kernel size 1, stride 1) to the shorter list.
        This preserves the dimensions while equalizing block counts.
        
        Args:
            height_sub_blocks: List of height sub-blocks
            width_sub_blocks: List of width sub-blocks
            
        Returns:
            Tuple of (equalized_height_sub_blocks, equalized_width_sub_blocks)
        """
        if len(height_sub_blocks) == len(width_sub_blocks):
            return height_sub_blocks, width_sub_blocks
        
        # Create an identity sub-block that doesn't change dimensions
        shape_preserving_layer = [{"type": "tconv", "kernel_size": 1, "stride": 1, "output_padding": 0}]
        
        if len(height_sub_blocks) < len(width_sub_blocks):
            for i in range(len(height_sub_blocks), len(width_sub_blocks)):
                # each new block should have the same number of layers as the width sub-block but as "identity_block"
                block_to_add = shape_preserving_layer * len(width_sub_blocks[i] - 1)
                # the very last layer should somehow identify the end of a sub-blokc
                end_block_preserving_layer = shape_preserving_layer[0].copy()
                end_block_preserving_layer['end_block'] = True
                block_to_add.append(end_block_preserving_layer)
                height_sub_blocks.append(block_to_add)

        else:
            for i in range(len(width_sub_blocks), len(height_sub_blocks)):
                # each new block should have the same number of layers as the height sub-block but as "identity_block"  
                block_to_add = shape_preserving_layer * (len(height_sub_blocks[i]) - 1)
                # the very last layer should somehow identify the end of a sub-blokc
                end_block_preserving_layer = shape_preserving_layer[0].copy()
                end_block_preserving_layer['end_block'] = True
                block_to_add.append(end_block_preserving_layer)
                width_sub_blocks.append(block_to_add)
        
        return height_sub_blocks, width_sub_blocks
    
    
    def _equalize_sub_block_layers(self, height_sub_block: List[Dict], 
                                  width_sub_block: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Ensure that height and width sub-blocks have the same number of layers.
        If they don't, duplicate the appropriate stride-1 layers.
        
        Args:
            height_sub_block: Height sub-block
            width_sub_block: Width sub-block
            
        Returns:
            Tuple of (equalized_height_sub_block, equalized_width_sub_block)
        """
        
        # the last layer is supposed to be a strided layer (stride > 1)
        # however, one exception is when the block is added as an identity block.

        h_block = height_sub_block[:-1]
        w_block = width_sub_block[:-1]

        shape_preserving_layer = [{"type": "tconv", "kernel_size": 1, "stride": 1, "output_padding": 0}]

        # the idea is to add identity layers to the shorter block so that both blocks have the same number of layers
        if len(h_block) < len(w_block):
            h_block.extend(shape_preserving_layer * (len(w_block) - len(h_block)))
        else:
            w_block.extend(shape_preserving_layer * (len(h_block) - len(w_block)))

        # before adding the last layer, make sure it is either a strided layer or a layer with the end_block flag.

        if height_sub_block[-1]["stride"] == 1:
            height_sub_block[-1]["end_block"] = True

        if width_sub_block[-1]["stride"] == 1:
            width_sub_block[-1]["end_block"] = True

        # then block block add their last layer
        h_block.append(height_sub_block[-1])
        w_block.append(width_sub_block[-1])

        return h_block, w_block
    

    def _set_layer_parameters(self, h_sub_block: List[Dict], 
                             w_sub_block: List[Dict]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Set the parameters for the layers in the height and width sub-blocks.
        """

        assert len(h_sub_block) == len(w_sub_block), "height and width sub-blocks must have the same number of layers"

        kernel_sizes = [(h_layer["kernel_size"], w_layer["kernel_size"]) for h_layer, w_layer in zip(h_sub_block, w_sub_block)]

        strides = [(h_layer["stride"], w_layer["stride"]) for h_layer, w_layer in zip(h_sub_block, w_sub_block)]

        output_paddings = [(h_layer["output_padding"], w_layer["output_padding"]) for h_layer, w_layer in zip(h_sub_block, w_sub_block)]

        paddings = [(h_layer.get("padding", 0), w_layer.get("padding", 0)) for h_layer, w_layer in zip(h_sub_block, w_sub_block)]

        return kernel_sizes, strides, output_paddings, paddings


    # def _set_kernel_sizes_and_paddings(self, h_sub_block: List[Dict], 
    #                                   w_sub_block: List[Dict]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    #     """
    #     Create 2D kernel sizes and padding tuples from height and width sub-blocks.
        
    #     Args:
    #         h_sub_block: Equalized height sub-block
    #         w_sub_block: Equalized width sub-block
            
    #     Returns:
    #         Tuple of (kernel_sizes, paddings) lists with 2D tuples
    #     """
    #     # Extract stride-1 layers
    #     h_stride1_layers = [layer for layer in h_sub_block if layer["stride"] == 1]
    #     w_stride1_layers = [layer for layer in w_sub_block if layer["stride"] == 1]
        
    #     # Create 2D kernel sizes
    #     kernel_sizes = []
    #     for h_layer, w_layer in zip(h_stride1_layers, w_stride1_layers):
    #         kernel_sizes.append((h_layer["kernel_size"], w_layer["kernel_size"]))
            
    #     # Default paddings of 0 for transpose convolutions
    #     paddings = [(0, 0) for _ in kernel_sizes]
        
    #     return kernel_sizes, paddings
    
    def _merge_blocks_singular(self, height_block: List[Dict], width_block: List[Dict]) -> List[nn.Sequential]:
        """
        Merge the height and width blocks where both of them have only one sub-block.
        """
        # Equalize layers within the sub-blocks
        h_equalized, w_equalized = self._equalize_sub_block_layers(height_block, width_block)
                
        # Compute the channels
        channels = compute_log_linear_sequence(self.input_shape[0], self.output_shape[0], 2) 

        # Get any strided transpose conv layers
        h_strided = next((layer for layer in h_equalized if layer["stride"] > 1), None)
        w_strided = next((layer for layer in w_equalized if layer["stride"] > 1), None)
        
        # Extract kernel sizes and paddings for stride-1 layers
        kernel_sizes, paddings = self._set_kernel_sizes_and_paddings(h_equalized, w_equalized)
        
        # Create the transpose conv block for stride-1 layers
        tconv = TransposeConvBlock(
            num_transpose_conv_layers=len(kernel_sizes),
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=1,
            paddings=paddings,
            output_paddings=0,
            use_bn=True
        )
        
        # Create the strided transpose conv layer if needed
        if h_strided and w_strided:
            strided_tconv = nn.ConvTranspose2d(
                channels[-1], channels[-1],
                kernel_size=(h_strided["kernel_size"], w_strided["kernel_size"]),
                stride=(h_strided["stride"], w_strided["stride"]),
                output_padding=(h_strided.get("output_padding", 0), w_strided.get("output_padding", 0)),
                padding=0  # No padding for transpose conv
            )
            return [nn.Sequential(OrderedDict([("tconv", tconv), ("strided_tconv", strided_tconv)]))]
        else:
            return [nn.Sequential(OrderedDict([("tconv", tconv)]))]
    
    def _merge_blocks(self) -> List[nn.Sequential]:
        """
        Merge the height and width blocks into a single architecture.
        
        Returns:
            List of nn.Sequential blocks that implement the expansion
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

        # Compute the channels for each block
        channels = compute_log_linear_sequence(self.input_shape[0], self.output_shape[0], len(height_sub_blocks) + 1)
        
        merged_blocks = []
        
        # For each pair of sub-blocks
        for block_index, (h_sub_block, w_sub_block) in enumerate(zip(height_sub_blocks, width_sub_blocks)):
            # Equalize layers within the sub-blocks
            h_equalized, w_equalized = self._equalize_sub_block_layers(h_sub_block, w_sub_block)
            
            # Extract kernel sizes and paddings for stride-1 layers
            kernel_sizes, strides, output_paddings, paddings = self._set_layer_parameters(h_equalized, w_equalized)
            
            # Create the transpose conv block for stride-1 layers
            block_channels = [channels[block_index]] * (len(kernel_sizes) + 1)
            
            tconv = TransposeConvBlock(
                num_transpose_conv_layers=len(kernel_sizes),
                channels=block_channels,
                kernel_sizes=kernel_sizes,
                strides=strides,
                paddings=paddings,
                output_paddings=output_paddings,
                use_bn=True
            )

            merged_blocks.append(tconv)

            # # Create the strided transpose conv layer if needed
            # if h_strided and w_strided:
            #     strided_tconv = nn.ConvTranspose2d(
            #         channels[block_index], channels[block_index + 1],
            #         kernel_size=(h_strided["kernel_size"], w_strided["kernel_size"]),
            #         stride=(h_strided["stride"], w_strided["stride"]),
            #         output_padding=(h_strided.get("output_padding", 0), w_strided.get("output_padding", 0)),
            #         padding=0  # No padding for transpose conv
            #     )
            #     merged_blocks.append(nn.Sequential(OrderedDict([
            #         ("tconv", tconv), 
            #         ("strided_tconv", strided_tconv)
            #     ])))
            # else:
            #     merged_blocks.append(nn.Sequential(OrderedDict([("tconv", tconv)])))
        
        return merged_blocks
    
    def get_expanding_block(self) -> List[TransposeConvBlock]:
        """
        Get the expanding block that transforms input shape to output shape.
        
        Returns:
            List of nn.Sequential blocks that together form the expanding architecture
        """
        self.height_block, self.width_block = self._design_blocks()
        self.merged_blocks = self._merge_blocks()
        
        return self.merged_blocks
