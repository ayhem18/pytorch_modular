"""
This module is my attempt to efficiently implement the multi-head attention layer: Using a single linear layer to compute the query, key and value matrics.
"""

import torch
import numpy as np

from torch import nn
from typing import Optional

from mypt.building_blocks.mixins.general import NonSequentialModuleMixin
from mypt.losses.auxiliary import MaskSoftmax


class MultiHeadAttentionLayer(NonSequentialModuleMixin, nn.Module):
    def __init__(self, d_model: int, num_heads: int, value_dim: int, key_dim: int) -> None:
        nn.Module.__init__(self)
        NonSequentialModuleMixin.__init__(self, inner_components_fields=['W_q', 'W_k', 'W_v', 'W_o'])

        if (d_model % num_heads != 0):
            raise ValueError(f"d_model must be divisible by num_heads, but got d_model={d_model} and num_heads={num_heads}")


        self.d_model = d_model
        self.num_heads = num_heads


        self.value_dim = value_dim
        self.key_dim = key_dim

        self.W_q = nn.Linear(d_model, key_dim * num_heads)
        self.W_k = nn.Linear(d_model, key_dim * num_heads)
        self.W_v = nn.Linear(d_model, value_dim * num_heads)

        self.W_o = nn.Linear(value_dim * num_heads, d_model) 

    # --------------------------------------------------------------
    # Mask helpers (aligned with SingleHeadAttentionLayer)
    # --------------------------------------------------------------

    def _causal_attention_mask(self, batch_size: int, sequence_length: int) -> torch.Tensor:
        """Boolean lower-triangular mask broadcast over heads.
        Shape: (B, 1, S, S) so it can be broadcast against (B,H,S,S)."""
        causal = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool))
        return causal.unsqueeze(0).unsqueeze(1).expand(batch_size, self.num_heads, -1, -1)

    def create_final_mask(self, causal_mask: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Combine causal and optional padding mask.

        causal_mask: (B,1,S,S) bool
        pad_mask:    (B,S) with 1/True keep, 0/False mask (optional)
        Returns: (B,1,S,S) bool mask ready to broadcast across heads.
        """
        if pad_mask is None:
            return causal_mask

        pad_bool = pad_mask.to(dtype=torch.bool)
        b, s = pad_bool.shape
        row = pad_bool.unsqueeze(-1).unsqueeze(1).expand(b, self.num_heads, s, s)
        col = pad_bool.unsqueeze(1).unsqueeze(1).expand(b, self.num_heads, s, s)
        return causal_mask & row & col

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


    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        """
        batch_size, sequence_length, __ = x.shape 
    
        q = self.W_q.forward(x)
        k = self.W_k.forward(x)
        v = self.W_v.forward(x)

        # create a view for efficiency
        # it is easier to work with (batch_size, num_heads, sequence_length, key_dim)
        q = q.view(batch_size, sequence_length, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, sequence_length, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, sequence_length, self.num_heads, self.value_dim).permute(0, 2, 1, 3)

        query_key_product = self._key_query_product(q, k)

        causal_mask = self._causal_attention_mask(batch_size, sequence_length).to(q.device)
        final_mask = self.create_final_mask(causal_mask, mask.to(q.device) if mask is not None else None)

        weights = self._compute_weights(query_key_product, final_mask)

        new_v = self._compute_new_v(weights, v)

        # the new_v must be of the shape (batch_size, num_heads, sequence_length, value_dim)
        if new_v.shape != (batch_size, self.num_heads, sequence_length, self.value_dim):
            raise ValueError(f"The shape of the new_v tensor is expected to be (batch_size, num_heads, sequence_length, value_dim), but got {new_v.shape}")

        output = self.W_o.forward(new_v.permute(0, 2, 1, 3).contiguous().view(batch_size, sequence_length, self.num_heads * self.value_dim))

        if output.shape != (batch_size, sequence_length, self.d_model):
            raise ValueError(f"The shape of the output tensor is expected to be (batch_size, sequence_length, d_model), but got {output.shape}")

        return output


    def __call__(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(x, mask)

    
