import torch
from torch import nn
from typing import List, Tuple, Dict, Optional, Union, Any

from mypt.building_blocks.mixins.general import ModuleListMixin


class UnetSkipConnections(ModuleListMixin):
    """
    A class to manage skip connections between contracting and expanding paths in a UNet.
    
    Skip connections typically connect feature maps of same spatial dimensions between the 
    contracting and expanding paths. This class handles the creation and execution of these
    connections, including any necessary transformations to align feature dimensions.
    """
    
    def __init__(self, connections: Union[List[torch.nn.Module], torch.nn.ModuleList]):
        """
        Initialize the skip connections manager.
        
        Args:
            inner_model_field_name: Field name for the inner module list
        """
        super().__init__("_connections")

        if isinstance(connections, torch.nn.ModuleList):
            self._connections = connections
        else:
            self._connections = nn.ModuleList(connections)

        self.num_connections = len(self._connections)

    
    # def add_connection(self, in_channels: int, out_channels: int, name: Optional[str] = None) -> int:
    #     """
    #     Add a new skip connection that transforms from in_channels to out_channels.
        
    #     Args:
    #         in_channels: Number of input channels from the contracting path
    #         out_channels: Number of output channels needed by the expanding path
    #         name: Optional name for the connection
            
    #     Returns:
    #         Index of the added connection
    #     """
    #     # Create a 1x1 convolution to align channel dimensions if needed
    #     if in_channels != out_channels:
    #         conn = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    #     else:
    #         conn = nn.Identity()
        
    #     # Add the connection to the module list
    #     self._connections.append(conn)
        
    #     # Store the index for the new connection
    #     idx = self.num_connections
    #     self.num_connections += 1
        
    #     return idx
    
    def __getitem__(self, index: int) -> nn.Module:
        """
        Get the skip connection module at the specified index.
        
        Args:
            index: Index of the connection
            
        Returns:
            The module for the specified connection
        """
        if index < 0 or index >= self.num_connections:
            raise IndexError(f"Skip connection index {index} out of range (0-{self.num_connections-1})")
        
        return self._connections[index]
    
    def apply_connection(self, x: torch.Tensor, index: int) -> torch.Tensor:
        """
        Apply the skip connection transformation to the input tensor.
        
        Args:
            x: Input tensor from the contracting path
            index: Index of the connection to apply
            
        Returns:
            Transformed tensor ready for the expanding path
        """
        conn = self[index]
        return conn(x)
    
    def to(self, *args, **kwargs) -> 'UnetSkipConnections':
        """Move module to specified device or dtype"""
        super().module_list_to(*args, **kwargs)
        return self
    
    def train(self, mode: bool = True) -> 'UnetSkipConnections':
        """Set module to training mode"""
        super().module_list_train(mode)
        return self
    
    def eval(self) -> 'UnetSkipConnections':
        """Set module to evaluation mode"""
        super().module_list_eval()
        return self
    
    def parameters(self, recurse: bool = True) -> Any:
        """Return an iterator over module parameters"""
        for module in self._connections:
            for param in module.parameters(recurse=recurse):
                yield param
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Any:
        return super().module_list_named_parameters(prefix, recurse)
    
    def modules(self) -> Any:
        """Return an iterator over modules"""
        return super().module_list_modules()
    
    def __len__(self) -> int:
        return self.num_connections
    