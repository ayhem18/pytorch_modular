import torch

import torch.nn as nn

from torch.nn.parameter import Parameter
from typing import Optional, OrderedDict, Union, Callable, Iterator, Tuple

from mypt.building_blocks.auxiliary.norm_act import NormActBlock
from mypt.building_blocks.mixins.residual_mixins import GeneralResidualMixin

class WideResnetBlock(GeneralResidualMixin):
    """
    A basic residual block for the WideResnet network as described in the paper: 
    https://arxiv.org/abs/1605.07146
    

    Important notes no this block: 

    1. The block uses the pre-activation design (BN+ReLU before convolution) 
    
    2. The output shape of the block is very uniform depending on the value of the stride: 
        * if the stride is 1, then given an input shape of (ic, w, h), the output shape will be (oc, w, h) 
        * if the stride is 2, then given an input shape of (ic, w, h), the output shape will be (oc, (w+1)//2, (h+1)/2)
            if the height and width are both even, then floor((w+1)/2) and floor((h+1)/2) are equal to w / 2 and h / 2 respectively. 

    This seems to be an assumption of the paper, but not really explicitly stated. As of the time of the implementation, these assumptions will be forced
    until finding the time for a more flexible implementation (probably using the adaptive convolutional blocks implemented in the library)

    I understood these details by going though the source code 

    https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/resnet.py 

    also note that in the paper, the output size is divided by 2 after each block 
    (and since there is no pooling layer, the convolutional block must be strided exactly once in each block)
    """
    def __set_norm_act_block(self, 
                            channels:int,
                            norm: Optional[nn.Module] = None, 
                            norm_params: Optional[dict] = None, 
                            activation: Optional[Union[str, Callable]] = None, 
                            activation_params: Optional[dict] = None):
        
        if norm is None:
            norm = nn.BatchNorm2d
        
        if norm_params is None:
            norm_params = {}
        
        if len(norm_params) == 0:
            norm_params["num_features"] = channels

        if activation is None:
            activation = 'relu'

        if activation_params is None:
            activation_params = {}

        return NormActBlock(norm, norm_params, activation, activation_params)


    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1, 
        dropout_rate: float = 0.0,


        # normalization parameters  
        norm1: Optional[nn.Module] = None, # the default normalization layer is BatchNorm2d
        norm1_params: Optional[dict] = None, # the default parameters are num_features = in_channels
        norm2: Optional[nn.Module] = None, # the default normalization layer is BatchNorm2d
        norm2_params: Optional[dict] = None, # the default parameters are num_features = out_channels
       
        # activation parameters
        activation: Optional[Union[str, Callable]] = None,
        activation_params: Optional[dict] = None,

        # whether to use a convolutional layer as the shortcut connection when the input shape is the same as the output shape
        force_residual: bool=False
    ):
        # let's reinforce a few checks before proceeding
        if stride not in [1, 2]:
            raise ValueError(f"Stride must be 1 or 2, got {stride}")

        main_stream_field_name = "_block"

        # can be a one-liner, but let's focus on clarify for now
        if not force_residual and (stride == 1 and in_channels == out_channels):
            # the shortcut connection can be a simple identity if the stride is 1 and the input and output channels are the same 
            residual_stream_field_name = None

        else:
            residual_stream_field_name = "_shortcut"

        
        super().__init__(main_stream_field_name=main_stream_field_name, residual_stream_field_name=residual_stream_field_name)
        
        # Store parameters
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._stride = stride
        self._dropout_rate = dropout_rate
        self._activation = activation
        self._activation_params = activation_params
        self._force_residual = force_residual 

        # First convolution block

        # let's build the dictionary to be passed to the nn.Sequential as the main stream
        main_stream_ordered_dict = OrderedDict()

        # first normalization layer
        main_stream_ordered_dict['norm_act_1'] = self.__set_norm_act_block(self.in_channels, norm1, norm1_params, activation, activation_params)   
        # first convolution layer 
        main_stream_ordered_dict["conv1"] = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
        )

        # dropout layer in between 
        main_stream_ordered_dict["dropout"] = nn.Dropout(dropout_rate) # with a dropout rate of 0.0, the layer works as an identity layer 


        main_stream_ordered_dict["norm_act_2"] = self.__set_norm_act_block(self.out_channels, norm2, norm2_params, activation, activation_params)
        # second convolution layer
        main_stream_ordered_dict["conv2"] = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, # the second convolution layer is always 1-strided 
            padding=1, 
        )

        self._block = nn.Sequential(main_stream_ordered_dict)

        if residual_stream_field_name is None:
            self._shortcut = None 
            return 
        
        # at this point, we know that the input shape is different from the output shape
        # the shortcut connection simply uses the output channels and the given stride with a padding of 1
        self._shortcut = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
        )

    # override general torch.nn.Module methods to use the GeneralResidualMixin methods
    def forward(self, x: torch.Tensor, debug: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the residual block.
        
        Args:
            x: Input tensor
            debug: If True, returns main stream output, residual stream output, and final output
            
        Returns:
            Output tensor or (main_output, residual_output, final_output) if debug=True
        """
        return super().residual_forward(x, debug=debug)
    
    def __call__(self, x: torch.Tensor, debug: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Support for optional debug parameter in __call__
        """
        return self.forward(x, debug=debug)
    
    # Override standard module methods to use residual implementations
    def children(self) -> Iterator[nn.Module]:
        """Returns an iterator over immediate children modules."""
        return super().residual_children()
    
    def named_children(self) -> Iterator[Tuple[str, nn.Module]]:
        """Returns an iterator over immediate children modules, yielding both the name and the module."""
        return super().residual_named_children()
    
    def modules(self) -> Iterator[nn.Module]:
        """Returns an iterator over all modules in the network, recursively."""
        return super().residual_modules()
    
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Returns an iterator over module parameters."""
        return super().residual_parameters(recurse=recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        """Returns an iterator over module parameters, yielding both the name and the parameter."""
        return super().residual_named_parameters(prefix=prefix, recurse=recurse)
    
    def to(self, *args, **kwargs) -> 'WideResnetBlock':
        """Moves and/or casts the parameters and buffers."""
        super().residual_to(*args, **kwargs)
        return self
    
    def train(self, mode: bool = True) -> 'WideResnetBlock':
        """Sets the module in training mode."""
        super().residual_train(mode)
        return self
    
    def eval(self) -> 'WideResnetBlock':
        """Sets the module in evaluation mode."""
        super().residual_eval()
        return self
    
    @property
    def in_channels(self) -> int:
        """Returns the number of input channels."""
        return self._in_channels
    
    @property
    def out_channels(self) -> int:
        """Returns the number of output channels."""
        return self._out_channels
    
    @property
    def stride(self) -> int:
        """Returns the stride used in the block."""
        return self._stride
    
    @property
    def dropout_rate(self) -> float:
        """Returns the dropout rate used in the block."""
        return self._dropout_rate

