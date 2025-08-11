
import torch, abc

from torch import nn
from typing import Optional

from mypt.building_blocks.mixins.general import NonSequentialModuleMixin



class BaseAttentionAgg(nn.Module, abc.ABC):
    def __init__(self, num_heads: int, value_dim: int, key_dim: int) -> None:
        nn.Module.__init__(self)
        self.value_dim = value_dim
        self.key_dim = key_dim
        self.num_heads = num_heads

    @abc.abstractmethod
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pad_mask: Optional[torch.Tensor] = None, is_causal: bool = False) -> torch.Tensor:
        pass

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pad_mask: Optional[torch.Tensor] = None, is_causal: bool = False) -> torch.Tensor:
        return self.forward(q, k, v, pad_mask, is_causal)


    def _causal_mask(self, batch_size: int, sequence_length: int) -> torch.Tensor:
        """Boolean lower-triangular mask broadcast over heads.
        Shape: (B, num_heads, S, S)"""
        causal = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool))
        return causal.unsqueeze(0).unsqueeze(1).expand(batch_size, self.num_heads, -1, -1)


    def _create_final_mask(self, default_mask: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Combine the default mask with the optional padding mask.
        The default mask can be either a causal mask (lower triangular 2d matrix) or an all-ones mask.

        Args:
            default_mask: (B,1,S,S) bool
        pad_mask:    (B,S) with 1/True keep, 0/False mask (optional)
        Returns: (B,1,S,S) bool mask ready to broadcast across heads.
        """
        if pad_mask is None:
            return default_mask

        pad_bool = pad_mask.to(dtype=torch.bool)
        b, s = pad_bool.shape
        row = pad_bool.unsqueeze(-1).unsqueeze(1).expand(b, self.num_heads, s, s)
        col = pad_bool.unsqueeze(1).unsqueeze(1).expand(b, self.num_heads, s, s)
        return default_mask & row & col
