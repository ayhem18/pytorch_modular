import abc
import torch
from torch import nn
from typing import Optional, Dict, Callable

class _BasePooling(nn.Module, abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass


class CLSPool(_BasePooling):
    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # assumes a CLS token is prepended at position 0
        return x[:, 0]


class MeanPool(_BasePooling):
    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if pad_mask is None:
            return x.mean(dim=1)
        
        # pad_mask is expected to be of shape (B, S)
        
        # compute the effective sequence lengths
        # make sure to set 1 as the minimum, not to divide by zero 
        lengths = pad_mask.sum(dim=-1, keepdim=True).clamp(min=1)  
        
        # sum the non-masked values and divide by the effective sequence lengths
        return (x * pad_mask.unsqueeze(-1)).sum(dim=1) / lengths


class MaxPool(_BasePooling):
    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if pad_mask is None:
            return x.max(dim=1).values
        
        # pad_mask is expected to be of shape (B, S)
        # set the masked value to -inf to avoid affecting the max value
        x_masked = x.masked_fill(~pad_mask.bool().unsqueeze(-1), float("-inf"))
        
        # return the max value for each sequence
        return x_masked.max(dim=1).values



class LastTokenPool(_BasePooling):
    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if pad_mask is None:
            return x[:, -1]
        # find the last non-masked token for each sequence
        idx = (pad_mask.sum(dim=1) - 1).long()  
        # return the value of the last non-masked token for each sequence (passing [:] wouldn't work here.)

        # any tensor used for indices should be cast to long !!!
        return x[torch.arange(x.shape[0], dtype=torch.long), idx]


_POOLING_REGISTRY: Dict[str, Callable] = {
    "cls": CLSPool,
    "mean": MeanPool,
    "max": MaxPool,
    # "attention": lambda d_model: AttentionPool(d_model),
    "last": LastTokenPool,
}

