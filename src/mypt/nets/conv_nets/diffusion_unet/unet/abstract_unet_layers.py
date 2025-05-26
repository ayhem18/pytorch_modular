import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple, Union


from mypt.building_blocks.mixins.general import SequentialModuleListMixin


class AbstractDownLayer(SequentialModuleListMixin, torch.nn.Module, ABC):
    """
    Abstract class for downsampling layers.
    
    A single downsampling layer for UNet architecture.
    
    This consists of:
    1. A series of instances of either the CondThreeDimWResBlock or the CondOneDimWResBlock
    2. either a DownCondThreeDimWResBlock or a DownCondOneDimWResBlock
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        cond_dimension: Number of conditioning dimensions
        num_resnet_blocks: Number of residual blocks to use
    """
    def __init__(self, 
            in_channels: int,
            out_channels: Union[int, List[int]],
            cond_dimension: int,
            num_resnet_blocks: int,
            *args, **kwargs):
        
        # initialize the nn.Module first
        nn.Module.__init__(self, *args, **kwargs)
        # initialize the SequentialModuleListMixin  
        SequentialModuleListMixin.__init__(self, "_resnet_blocks")

        # a few sanity checks
        if isinstance(out_channels, int):
            out_channels = [out_channels] * num_resnet_blocks

        if len(out_channels) != num_resnet_blocks:
            raise ValueError(f"The length of out_channels must be equal to the number of resnet blocks. Got {len(out_channels)} and {num_resnet_blocks}.")

        # the number of resnet blocks cannot be less than 2
        if num_resnet_blocks < 2:
            raise ValueError(f"The number of resnet blocks must be at least 2. Got {num_resnet_blocks}.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_dimension = cond_dimension
        self.num_resnet_blocks = num_resnet_blocks
        
        # the sub-classes must implement this attribute
        self._resnet_blocks: nn.ModuleList = None


    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the down layer.
        
        Args:
            x: Input tensor [B, C, H, W]
            condition: Conditioning tensor
            
        Returns:
            List of tensors from each resnet block
        """
        return super().sequential_module_list_forward(x, condition)

    def to(self, *args, **kwargs) -> 'AbstractDownLayer': 
        super().module_list_to(*args, **kwargs)
        return self 
    
    def train(self, mode: bool = True) -> 'AbstractDownLayer':
        super().module_list_train(mode)
        return self
    
    def eval(self) -> 'AbstractDownLayer':
        super().module_list_eval()
        return self
    
    def modules(self) -> Iterator[nn.Module]:
        return super().module_list_modules()

    def parameters(self, recurse: bool = True) -> Iterator[torch.Tensor]:
        return super().module_list_parameters(recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.Tensor]]:
        return super().module_list_named_parameters(prefix, recurse)

    


class AbstractUpLayer(SequentialModuleListMixin, torch.nn.Module, ABC):
    """
    A single upsampling layer for UNet architecture.
    
    This consists of:
    1. A series of instances of either the CondThreeDimWResBlock or the CondOneDimWResBlock
    2. either a UpCondThreeDimWResBlock or a UpCondOneDimWResBlock
        
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        cond_dimension: Number of conditioning dimensions
        num_resnet_blocks: Number of residual blocks to use
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dimension: int,
        num_resnet_blocks: int,
        *args, **kwargs
    ):  
        nn.Module.__init__(self, *args, **kwargs)
        SequentialModuleListMixin.__init__(self, "_resnet_blocks")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_dimension = cond_dimension
        self.num_resnet_blocks = num_resnet_blocks

        # the sub-classes must implement this attribute
        self._resnet_blocks: nn.ModuleList = None

  
    @abstractmethod
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the up layer.
        
        Args:
            x: Input tensor [B, C, H, W]
            condition: Conditioning tensor
            condition: Conditioning tensor
            
        Returns:
            Output tensor
        """
        return super().sequential_module_list_forward(x, condition)
    
    def to(self, *args, **kwargs) -> 'AbstractUpLayer': 
        super().module_list_to(*args, **kwargs)
        return self 
    
    def train(self, mode: bool = True) -> 'AbstractUpLayer':
        super().module_list_train(mode)
        return self
    
    def eval(self) -> 'AbstractUpLayer':
        super().module_list_eval()
        return self 
    
    def modules(self) -> Iterator[nn.Module]:
        return super().module_list_modules()
    
    def parameters(self, recurse: bool = True) -> Iterator[torch.Tensor]:
        return super().module_list_parameters(recurse) 
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.Tensor]]:
        return super().module_list_named_parameters(prefix, recurse)





