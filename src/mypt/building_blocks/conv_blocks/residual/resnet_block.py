from mypt.building_blocks.auxiliary.norm_act import NormActBlock
import torch

import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, OrderedDict, Union, Callable, Iterator, Tuple
from torch.nn.parameter import Parameter

from mypt.building_blocks.mixins.residual_mixins import GeneralResidualMixin
from mypt.building_blocks.auxiliary.normalization.utils import get_normalization
from mypt.building_blocks.auxiliary.activations.activations import get_activation

class WideResnetBlock(GeneralResidualMixin):
    """
    A basic residual block for the WideResnet network as described in the paper: 
    https://arxiv.org/abs/1605.07146
    

    Important notes no this block: 

    1. The block uses the pre-activation design (BN+ReLU before convolution) 
    2. The output shape of the block is very uniform depending on the value of the stride: 
        * if the stride is 1, then given an input shape of (w, h, ic), the output shape will be (w, h, oc) 
        * if the stride is 2, then given an input shape of (w, h, ic), the output shape will be ((w+1)//2, (h+1)/2, oc)
            if the height and width are both even, then floor((w+1)/2) and floor((h+1)/2) are equal to w / 2 and h / 2 respectively. 

    This seems to be an assumption of the paper, but not really explicitly stated. As of the time of the implementation, these assumptions will be forced
    until finding the time for a more flexible implementation (probably using the adaptive convolutional blocks implemented in the library)

    I understood these details by going though the source code 

    https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/resnet.py 

    also note that in the paper, the output size is divided by 2 after each block 
    (and since there is no pooling layer, the convolutional block must be strided exactly once...)
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1, 
        dropout_rate: float = 0.0,


        # normalization parameters  
        norm1: Optional[nn.Module] = None,
        norm1_params: Optional[dict] = None,
        norm2: Optional[nn.Module] = None,
        norm2_params: Optional[dict] = None,
       
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

        # First convolution block

        # let's build the dictionary to be passed to the nn.Sequential as the main stream
        main_stream_ordered_dict = OrderedDict()

        # NOTE: THIS CODE IS EXPERIMENTAL SINCE I NEED TO UNDERSTAND HOW TO HANDLE ALL POPULAR NORMALIZATION LAYERS...  
        # TODO: MAKE THIS MORE FLEXIBLE AND GENERAL

        norm1_params = norm1_params or {}
        norm1_params["in_channels"] = in_channels 

        # first normalization layer
        main_stream_ordered_dict['norm_act_1'] = NormActBlock(norm1, norm1_params, activation, activation_params)   
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

        # second normalization layer
        norm2_params = norm2_params or {}
        norm2_params["in_channels"] = out_channels

        main_stream_ordered_dict["norm_act_2"] = NormActBlock(norm2, norm2_params, activation, activation_params)
        # second convolution layer
        main_stream_ordered_dict["conv2"] = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
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




class ConditionedWideResNetBlock(nn.Module):
    """Conditioned Wide ResNet block that accepts external conditioning information.
    
    This extends the standard WRN block with conditioning capabilities using FiLM
    (Feature-wise Linear Modulation) for adaptive batch normalization.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        cond_channels: int,
        stride: int = 1, 
        dropout_rate: float = 0.0,
        activation: Union[str, Callable] = "relu",
        conditioning_method: str = "film"  # film, concat, or add
    ):
        super().__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_channels = cond_channels
        self.stride = stride
        self.dropout_rate = dropout_rate
        self.conditioning_method = conditioning_method
        
        # Set activation function
        if isinstance(activation, str):
            if activation.lower() == "relu":
                self.activation = nn.ReLU(inplace=True)
            elif activation.lower() == "leaky_relu":
                self.activation = nn.LeakyReLU(0.1, inplace=True)
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        else:
            self.activation = activation
        
        # First convolution block
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            bias=False
        )
        
        # Second convolution block
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=1, 
                    stride=stride, 
                    bias=False
                )
            )
        
        # Conditioning layers
        if conditioning_method == "film":
            # FiLM conditioning generating scale and shift parameters
            self.cond_encoder = nn.Sequential(
                nn.Linear(cond_channels, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, out_channels * 2)  # 2 for scale and shift
            )
        elif conditioning_method == "add":
            # Simple additive conditioning
            self.cond_encoder = nn.Sequential(
                nn.Linear(cond_channels, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, out_channels)
            )
        elif conditioning_method == "concat":
            # For concatenation, we'll adjust the second conv to handle more channels
            self.cond_encoder = nn.Sequential(
                nn.Linear(cond_channels, out_channels),
                nn.ReLU(inplace=True)
            )
            # Adjust second conv for concatenated inputs
            self.conv2 = nn.Conv2d(
                out_channels * 2,  # Double channels due to concatenation
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )
        else:
            raise ValueError(f"Unsupported conditioning method: {conditioning_method}")
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # First block (BN -> ReLU -> Conv)
        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)
        
        # Dropout if specified
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        
        # Apply conditioning based on method
        if self.conditioning_method == "film":
            # FiLM conditioning (scale and shift)
            cond = self.cond_encoder(condition)
            gamma, beta = torch.chunk(cond, 2, dim=1)
            
            # Reshape for broadcasting
            gamma = gamma.view(gamma.size(0), -1, 1, 1)
            beta = beta.view(beta.size(0), -1, 1, 1)
            
            # Apply after first BN in second block
            out = self.bn2(out)
            out = gamma * out + beta
            out = self.activation(out)
            out = self.conv2(out)
            
        elif self.conditioning_method == "add":
            # Additive conditioning
            cond = self.cond_encoder(condition)
            cond = cond.view(cond.size(0), -1, 1, 1)
            
            out = self.bn2(out)
            out = self.activation(out)
            out = out + cond  # Add conditioning
            out = self.conv2(out)
            
        elif self.conditioning_method == "concat":
            # Concatenation conditioning
            cond = self.cond_encoder(condition)
            cond = cond.view(cond.size(0), -1, 1, 1)
            cond = cond.expand(-1, -1, out.size(2), out.size(3))
            
            out = self.bn2(out)
            out = self.activation(out)
            out = torch.cat([out, cond], dim=1)  # Concatenate along channel dimension
            out = self.conv2(out)
        
        # Add shortcut connection
        out += self.shortcut(identity)
        
        return out


class WideResNet(nn.Module):
    """Implementation of Wide Residual Network as described in https://arxiv.org/abs/1605.07146"""
    def __init__(
        self, 
        depth: int, 
        width_factor: int, 
        num_classes: int, 
        input_channels: int = 3,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        
        assert (depth - 4) % 6 == 0, "Depth should be 6n+4"
        n = (depth - 4) // 6
        
        # Compute channel widths
        k = width_factor
        nStages = [16, 16*k, 32*k, 64*k]
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        
        # Residual blocks
        self.layer1 = self._make_layer(BasicBlock, nStages[0], nStages[1], n, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(BasicBlock, nStages[1], nStages[2], n, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(BasicBlock, nStages[2], nStages[3], n, stride=2, dropout_rate=dropout_rate)
        
        # Final BN and classifier
        self.bn = nn.BatchNorm2d(nStages[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nStages[3], num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride, dropout_rate):
        layers = []
        layers.append(block(in_channels, out_channels, stride=stride, dropout_rate=dropout_rate))
        
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride=1, dropout_rate=dropout_rate))
            
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.bn(out)
        out = self.relu(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out