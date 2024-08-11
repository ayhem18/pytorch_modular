"""
This script contains an abstact class of a model trained with the SimClr algorithm.
"""

import torch

from torch import nn
from typing import Tuple, Iterator


class SimClrModel(nn.Module):
    def __init__(self) -> None:
        
        super().__init__()
        
        # the feature extractor or the encoder "f" as denoted in the paper.
        self.fe: nn.Module = None 
        self.ph: nn.Module = None
        self.flatten_layer = nn.Flatten()
        self.model = nn.Sequential(self.fe, self.flatten_layer, self.ph)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # the forward function returns both f(x_i) and g(f(x_i)), any loss object should work with gf(x_i)
        f_xi = self.fe(x)
        return f_xi, self.ph.forward(self.flatten_layer.forward(f_xi))


    def __str__(self):
        return self.model.__str__()
    
    def __repr__(self):
        return self.model.__repr__() 
    
    def children(self) -> Iterator[nn.Module]:
        # not overloading this method will return to an iterator with 2 elements: self.__net and self.model
        return self.model.children()

    def modules(self) -> Iterator[nn.Module]:
        return self.model.modules()
    
    def named_children(self) -> Iterator[Tuple[str, nn.Module]]:
        return self.model.named_children()

    def to(self, *args, **kwargs):
        # self.model = self.model.to(*args, **kwargs)
        self.fe = self.fe.to(*args, **kwargs)
        self.flatten_layer = self.flatten_layer.to(*args, **kwargs)
        self.ph = self.ph.to(*args, **kwargs)
        return self 
