import abc
import torch

from torch import nn
from __future__ import annotations
from typing import Optional, Dict, Callable

from mypt.transformers.transformer_block import TransformerBlock
from mypt.building_blocks.mixins.general import NonSequentialModuleMixin
from mypt.building_blocks.linear_blocks.fc_blocks import ExponentialFCBlock
from mypt.building_blocks.auxiliary.embeddings.scalar.encoding import PositionalEncoding

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
        idx = pad_mask.sum(dim=1) - 1  
        # return the value of the last non-masked token for each sequence (passing [:] wouldn't work here.)
        return x[torch.arange(x.shape[0]), idx]


_POOLING_REGISTRY: Dict[str, Callable] = {
    "cls": CLSPool,
    "mean": MeanPool,
    "max": MaxPool,
    # "attention": lambda d_model: AttentionPool(d_model),
    "last": LastTokenPool,
}


class TransformerClassifier(NonSequentialModuleMixin, nn.Module):
    """Stack of Transformer blocks + pooling + classification head."""

    def __init__(
        self,
        d_model: int,
        num_transformer_blocks: int,
        num_classification_layers: int,
        num_heads: int,
        value_dim: int,
        key_dim: int,
        num_classes: int,
        pooling: str = "cls",
        dropout: float = 0.1,
    ) -> None: 
        nn.Module.__init__(self)
        NonSequentialModuleMixin.__init__(self, inner_components_fields=['encoder', 'pool', 'head', 'pos_emb'])

        if pooling not in _POOLING_REGISTRY:
            raise ValueError(f"Unknown pooling strategy '{pooling}'.")

        self.pos_emb = PositionalEncoding(d_model)

        blocks = [TransformerBlock(d_model, num_heads, value_dim, key_dim, dropout) for _ in range(num_transformer_blocks)]
        self.encoder = nn.Sequential(*blocks)

        # pooling module
        self.pooling_name = pooling
        self.pool = _POOLING_REGISTRY[pooling]()

        self.head = ExponentialFCBlock(output=num_classes,
                                       in_features=d_model,
                                       num_layers=num_classification_layers,
                                       dropout=dropout,
                                       activation='gelu')

    def forward(self, sequence: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # the first step is to encode the position of the tokens in the sequence
        pos_encoding = self.pos_emb(torch.arange(sequence.shape[1], device=sequence.device))
        
        if pad_mask is not None:
            pos_encoding = pos_encoding.masked_fill(~pad_mask.bool().unsqueeze(-1), 0)
        
        sequence = sequence + pos_encoding
        sequence = self.encoder(sequence, pad_mask)
        sequence = self.pool(sequence, pad_mask)

        return self.head(sequence)


    def get_pooling_type(self) -> str:  # utility accessor
        return self.pooling_name



