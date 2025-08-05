"""
This script implements the standard attention mechanism. This is my custom implementation with the scaled dot-product attention.
"""


import torch
import numpy as np

from typing import Optional

from .base import BaseAttentionAgg
from mypt.losses.auxiliary import MaskSoftmax
from mypt.building_blocks.mixins.general import NonSequentialModuleMixin


class DotProductCustomAttAgg(NonSequentialModuleMixin, BaseAttentionAgg):
    def __init__(self, num_heads: int, value_dim: int, key_dim: int) -> None:
        BaseAttentionAgg.__init__(self, num_heads=num_heads, value_dim=value_dim, key_dim=key_dim)

        NonSequentialModuleMixin.__init__(self, inner_components_fields=["W_o"])

        self.W_o = torch.nn.Linear(value_dim * num_heads, self.d_model) 


    def _key_query_product(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        """
        batch_size, _, sequence_length, _ = q.shape 

        query_key_product = torch.matmul(q, k.permute(0, 1, 3, 2)) / np.sqrt(self.key_dim).item()

        # the result must be of the shape (batch_size, num_heads, sequence_length, sequence_length)
        if query_key_product.shape != (batch_size, self.num_heads, sequence_length, sequence_length):
            raise ValueError(f"The shape of the query_key_product tensor is expected to be (batch_size, num_heads, sequence_length, sequence_length), but got {query_key_product.shape}")

        return query_key_product

    def _compute_weights(self, query_key_product: torch.Tensor, final_mask: torch.Tensor) -> torch.Tensor:
        """Apply the boolean *final_mask* to the scores and return soft-max weights.
        Args:
            query_key_product: (B,H,S,S) raw scaled dot-product scores.
            final_mask:        (B,H,S,S) boolean mask where True=keep.
        """
        return MaskSoftmax()(query_key_product, final_mask, dim=-1)


    def _compute_new_v(self, weights: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        """
        batch_size, _, sequence_length, _ = v.shape

        output = torch.matmul(weights, v)
        
        # the output must be of the shape (batch_size, num_heads, sequence_length, value_dim)
        if output.shape != (batch_size, self.num_heads, sequence_length, self.value_dim):
            raise ValueError(f"The shape of the new_v tensor is expected to be (batch_size, num_heads, sequence_length, value_dim), but got {output.shape}")

        return output


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        """
        batch_size, sequence_length, _, _ = q.shape 

        query_key_product = self._key_query_product(q, k)

        default_mask = self._default_mask(batch_size, sequence_length).to(q.device)

        final_mask = self._create_final_mask(default_mask, mask.to(q.device) if mask is not None else None)

        weights = self._compute_weights(query_key_product, final_mask)

        new_v = self._compute_new_v(weights, v)

        # the new_v must be of the shape (batch_size, num_heads, sequence_length, value_dim)
        if new_v.shape != (batch_size, self.num_heads, sequence_length, self.value_dim):
            raise ValueError(f"The shape of the new_v tensor is expected to be (batch_size, num_heads, sequence_length, value_dim), but got {new_v.shape}")

        output = self.W_o.forward(new_v.permute(0, 2, 1, 3).contiguous().view(batch_size, sequence_length, self.num_heads * self.value_dim))

        if output.shape != (batch_size, sequence_length, self.d_model):
            raise ValueError(f"The shape of the output tensor is expected to be (batch_size, sequence_length, d_model), but got {output.shape}")

        return output





