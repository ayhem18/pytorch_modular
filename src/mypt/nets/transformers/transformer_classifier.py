import abc
import torch

from torch import nn
from typing import Optional, Dict, Callable

from mypt.building_blocks.mixins.general import NonSequentialModuleMixin
from mypt.building_blocks.linear_blocks.fc_blocks import ExponentialFCBlock
from mypt.building_blocks.transformers.transformer_block import TransformerBlock
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
        NonSequentialModuleMixin.__init__(self, inner_components_fields=['pos_emb', 'encoder', 'pool', 'head'])

        if pooling not in _POOLING_REGISTRY:
            raise ValueError(f"Unknown pooling strategy '{pooling}'.")

        self.pos_emb = PositionalEncoding(d_model)

        self.encoder = nn.ModuleList([TransformerBlock(d_model, num_heads, value_dim, key_dim, dropout) for _ in range(num_transformer_blocks)])

        # pooling module
        self.pooling_name = pooling
        self.pool = _POOLING_REGISTRY[pooling]()

        self.head = ExponentialFCBlock(output=1 if num_classes == 2 else num_classes,
                                       in_features=d_model,
                                       num_layers=num_classification_layers,
                                       dropout=dropout,
                                       activation='gelu',
                                       norm_layer='batchnorm1d' # the input to the head is 2D: batchnorm would do the trick
                                       )

    def forward(self, sequence: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # the first step is to encode the position of the tokens in the sequence
        pos_encoding = self.pos_emb(torch.arange(sequence.shape[1], device=sequence.device))
        
        if pad_mask is not None:
            pos_encoding = pos_encoding.masked_fill(~pad_mask.bool().unsqueeze(-1), 0)
        

        x = sequence + pos_encoding
        # according to the original paper, there is a dropout layer after the position encoding...
        # not sure if it makes much of a difference. Keep it simple for now.
        for block in self.encoder:
            x = block(x, pad_mask)

        pooled_output = self.pool(x, pad_mask)

        result = self.head(pooled_output)

        if torch.isnan(result).any():
            raise ValueError("NaN in result")
        
        return result


    def get_pooling_type(self) -> str:  # utility accessor
        return self.pooling_name


    def __call__(self, sequence: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(sequence, pad_mask)

