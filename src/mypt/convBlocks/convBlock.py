# the main idea here is to create a convolutional block that can be used to build larger networks.

from typing import List, Optional, OrderedDict, Tuple, Union
import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    A flexible convolutional block that can be configured with multiple layers.
    
    This class allows creating a sequence of convolutional layers with optional 
    batch normalization, activation functions, and pooling layers.
    """
    
    def __build_block_single_activation(self, 
                                        channels: List[int], 
                                        kernel_sizes: List[int], 
                                        stride: List[int], 
                                        padding: List[int], 
                                        use_bn: bool, 
                                        activation: nn.Module, 
                                        activation_params: dict):
        # define the activation layer 
        activation_layer = activation(**activation_params) if activation_params else activation()
        layers = []
        for i in range(len(channels) - 1):
            layers.append((f"conv_{i + 1}", nn.Conv2d(channels[i], channels[i+1], kernel_sizes[i], stride[i], padding[i])))
            if use_bn:
                layers.append((f"bn_{i + 1}", nn.BatchNorm2d(channels[i+1])))

        # at the very end, add the activation layer 
        layers.append((f"activation_{len(channels)}", activation_layer))
        return nn.Sequential(OrderedDict(layers))


    def __build_block_activation_after_each_layer(self, 
                                        channels: List[int], 
                                        kernel_sizes: List[int], 
                                        stride: List[int], 
                                        padding: List[int], 
                                        use_bn: bool, 
                                        activation: nn.Module, 
                                        activation_params: dict):
        # define the activation layer 
        layers = []

        for i in range(len(channels) - 1):
            # add the convolutional layer
            layers.append((f"conv_{i + 1}", nn.Conv2d(channels[i], channels[i+1], kernel_sizes[i], stride[i], padding[i])))
            # add the batch normalization layer
            if use_bn:
                layers.append((f"bn_{i + 1}", nn.BatchNorm2d(channels[i+1])))
            # add the activation layer
            # create a new activation layer each time (it seems that the Sequential module does not support having the same object in the list multiple times)
            activation_layer = activation(**activation_params) if activation_params else activation()
            layers.append((f"activation_{i + 1}", activation_layer))
        
        return nn.Sequential(OrderedDict(layers))


    def __init__(self, 
                 num_conv_layers: int, 
                 channels: Union[List[int], Tuple[int]],
                 kernel_sizes: Union[List[int], int],
                 strides: Optional[Union[List[int], int]] = 1,
                 paddings: Optional[Union[List[int], List[str], str, int]] = 'same',
                 use_bn:bool=True,
                 activation_after_each_layer: bool = True,
                 activation: Optional[nn.Module] = None,
                 activation_params: Optional[dict] = None,
                ):
        """The block can be designed into ways: 

        [conv layer, bn, activation, conv layer, bn, activation ...]
        or 
        [conv layer, bn, conv layer, bn, ... conv layer, bn, activation ]

        Args:
            num_conv_layers (int): _description_
            channels (Union[List[int], Tuple[int]]): _description_
            kernel_sizes (Union[List[int], int]): _description_
            stride (Optional[Union[List[int], int]], optional): _description_. Defaults to 1.
            padding (Optional[Union[List[int], List[str], str, int]], optional): _description_. Defaults to 'same'.
            use_bn (bool, optional): _description_. Defaults to True.
            activation (Optional[nn.Module], optional): _description_. Defaults to None.
            activation_params (Optional[dict], optional): _description_. Defaults to None.

        Raises:
            TypeError: _description_
            ValueError: _description_
        """
        super().__init__()

        if not isinstance(channels, Union[List, Tuple]):
            raise TypeError(f"The 'channels' argument must be a list or a tuple. Found: {type(channels)}")
        
        # make sure the number of channels is the number of conv layers + 1        
        if len(channels) != num_conv_layers + 1:
            raise ValueError(f"The number of channels must be the number of conv layers + 1. Found: {len(channels)} channels for {num_conv_layers} conv layers.")
        
        # convert all the other parameters to lists if needed
        if not isinstance(kernel_sizes, Union[List, Tuple]):
            kernel_sizes = [kernel_sizes] * num_conv_layers
        else:
            if len(kernel_sizes) != num_conv_layers:
                raise ValueError(f"The number of kernel sizes must be the number of conv layers. Found: {len(kernel_sizes)} kernel sizes for {num_conv_layers} conv layers.")


        if not isinstance(strides, Union[List, Tuple]):
            strides = [strides] * num_conv_layers
        else:
            if len(strides) != num_conv_layers:
                raise ValueError(f"The number of strides must be the number of conv layers. Found: {len(strides)} strides for {num_conv_layers} conv layers.")

        if not isinstance(paddings, Union[List, Tuple]):
            paddings = [paddings] * num_conv_layers
        else:
            if len(paddings) != num_conv_layers:
                raise ValueError(f"The number of paddings must be the number of conv layers. Found: {len(paddings)} paddings for {num_conv_layers} conv layers.")

        # set the fields
        self.num_conv_layers = num_conv_layers  
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.use_bn = use_bn
        self.activation_after_each_layer = activation_after_each_layer

        if activation_after_each_layer:
            self.block = self.__build_block_activation_after_each_layer(channels, kernel_sizes, strides, paddings, use_bn, activation, activation_params) 
        else:
            self.block = self.__build_block_single_activation(channels, kernel_sizes, strides, paddings, use_bn, activation, activation_params) 

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output after passing through all layers
        """
        return self.block(x)
    
    
    def children(self):
        return self.block.children()

    def named_children(self):
        return self.block.named_children()

    def parameters(self):
        return self.block.parameters()

    def named_parameters(self):
        return self.block.named_parameters()
    
    def to(self, device: torch.device):
        self.block.to(device)
        return self
    
    def eval(self):
        self.block.eval()
        return self
    
    def train(self):
        self.block.train()