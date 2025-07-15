"""
This file contains my implementation of the attention layer, according to the thorough explanation provided in the 9th chapter of the this amazing Standford resource: 
https://web.stanford.edu/~jurafsky/slp3/ 
"""

import torch 
import numpy as np

from torch import nn
from typing import Optional

from mypt.losses.auxiliary import MaskSoftmax
from mypt.building_blocks.mixins.general import NonSequentialModuleMixin


class SingleHeadAttentionLayer(NonSequentialModuleMixin, nn.Module):
    def __init__(self, input_dimension: int, value_dim: int, key_dim: int) -> None:
        nn.Module.__init__(self)
        NonSequentialModuleMixin.__init__(self, inner_components_fields=['W_q', 'W_k', 'W_v', 'W_o'])

        self.input_dimension = input_dimension
        self.value_dim = value_dim
        self.key_dim = key_dim

        self.W_q = nn.Linear(input_dimension, key_dim) # q_i = W_q * x_i
        self.W_k = nn.Linear(input_dimension, key_dim) # k_i = W_k * x_i
        self.W_v = nn.Linear(input_dimension, value_dim) # v_i = W_v * x_i
        
        self.W_o = nn.Linear(value_dim, input_dimension) # head_i = W_o * v_i


    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------

    def _causal_attention_mask(self, batch_size: int, sequence_length: int) -> torch.Tensor:
        """Return a *boolean* causal (lower-triangular) mask of shape
        (batch_size, sequence_length, sequence_length).

        True  = position is *visible* (allowed to attend)
        False = position is *masked* (disallowed)
        """
        causal = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool))
        return causal.unsqueeze(0).expand(batch_size, -1, -1)

    def create_final_mask(self, causal_mask: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Combine the causal mask with an optional *padding* (sequence) mask.

        Args:
            causal_mask: Bool tensor (B,S,S) from `_causal_attention_mask`.
            pad_mask:    Bool/byte/int tensor (B,S) where 1/True keeps the token,
                         0/False masks it.  May be ``None``.

        Returns:
            Bool tensor (B,S,S) where True indicates *keep*, False indicates mask.
        """
        if pad_mask is None:
            return causal_mask

        # Convert to boolean where True = keep
        pad_mask_bool = pad_mask.to(dtype=torch.bool)

        b, s = pad_mask_bool.shape

        # Expand along rows and columns
        mask_row = pad_mask_bool.unsqueeze(-1).expand(b, s, s)  # queries
        mask_col = pad_mask_bool.unsqueeze(1).expand(b, s, s)   # keys

        final_mask = causal_mask & mask_row & mask_col
        return final_mask

    def _key_query_product(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        """
        batch_size, sequence_length, __ = q.shape 

        # compute the dot product of the query and key matrices
        # the result must be of the shape (batch_size, sequence_length, sequence_length) 
        query_key_product = torch.bmm(q, k.permute(0, 2, 1)) / np.sqrt(self.key_dim).item()
        
        if query_key_product.shape != (batch_size, sequence_length, sequence_length):
            raise ValueError(f"The shape of the query_key_product tensor is expected to be (batch_size, sequence_length, sequence_length), but got {query_key_product.shape}")

        return query_key_product

    def _compute_weights(self, query_key_product: torch.Tensor, final_mask: torch.Tensor) -> torch.Tensor:
        """Apply the boolean *final_mask* to the scores and return soft-max weights.

        Args:
            query_key_product: (B,S,S) raw scaled dot-product scores.
            final_mask:        (B,S,S) boolean mask where True=keep, False=mask.
        """
        return MaskSoftmax().forward(query_key_product, final_mask, dim=-1)

    def _compute_new_v(self, weights: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        can be refactored
        """
        batch_size, sequence_length, __ = v.shape

        output = torch.bmm(weights, v)
        
        # the output must be of the shape (batch_size, sequence_length, value_dim)
        if output.shape != (batch_size, sequence_length, self.value_dim):
            raise ValueError(f"The shape of the new_v tensor is expected to be (batch_size, sequence_length, value_dim), but got {output.shape}")

        return output

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x is expected to be of the shape (batch_size, sequence_length, input_dimension)
        """

        if x.ndim != 3:
            raise ValueError(f"The input tensor is expected to be of the shape (batch_size, sequence_length, input_dimension), but got {x.shape}")

        batch_size, sequence_length, __ = x.shape 

        q = self.W_q.forward(x) # (batch_size, sequence_length, key_dim)
        k = self.W_k.forward(x) # (batch_size, sequence_length, key_dim)
        v = self.W_v.forward(x) # (batch_size, sequence_length, value_dim)

        query_key_product = self._key_query_product(q, k)

        # Build masks -----------------------------------------------------
        causal_mask = self._causal_attention_mask(batch_size, sequence_length).to(q.device)
        # combine the causal mask with the padding mask if provided
        final_mask = self.create_final_mask(causal_mask, mask.to(q.device) if mask is not None else None)

        # Compute weights using the final boolean mask
        weights = self._compute_weights(query_key_product, final_mask)
        
        output = self._compute_new_v(weights, v) 

        return self.W_o.forward(output)

    def __call__(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(x, mask)

