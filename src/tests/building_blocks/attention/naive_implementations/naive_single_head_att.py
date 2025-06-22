import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from typing import Optional

# from mypt.losses.auxiliary import MaskSoftmax


class NaiveSHA(nn.Module):
    def __init__(self, input_dimension: int, value_dim: int, key_dim: int) -> None:
        super().__init__()

        self.input_dimension = input_dimension
        self.value_dim = value_dim
        self.key_dim = key_dim

        self.W_q = nn.Linear(input_dimension, key_dim)  # q_i = W_q * x_i
        self.W_k = nn.Linear(input_dimension, key_dim)  # k_i = W_k * x_i
        self.W_v = nn.Linear(input_dimension, value_dim)  # v_i = W_v * x_i
        
        self.W_o = nn.Linear(value_dim, input_dimension)  # head_i = W_o * v_i
        # self.sm = MaskSoftmax()

    # --------------------------------------------------------------
    # Mask helpers
    # --------------------------------------------------------------

    def _causal_attention_mask(self, batch_size: int, sequence_length: int) -> torch.Tensor:
        """Return boolean lower-triangular mask (B,S,S)."""

        causal = torch.zeros(sequence_length, sequence_length, dtype=torch.bool)

        for i in range(sequence_length):
            for j in range(sequence_length):
                if j > i:
                    causal[i, j] = False
                else:
                    causal[i, j] = True

        return causal.unsqueeze(0).expand(batch_size, -1, -1)


    def create_final_mask(self, causal_mask: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Same semantics as in the vectorised implementation."""
        if pad_mask is None:
            return causal_mask

        pad_mask_bool = pad_mask.to(dtype=torch.bool)

        b, s = pad_mask_bool.shape

        # Expand along rows and columns
        mask_row = pad_mask_bool.unsqueeze(-1).expand(b, s, s)  # queries
        mask_col = pad_mask_bool.unsqueeze(1).expand(b, s, s)   # keys

        final_mask = causal_mask & mask_row & mask_col
        return final_mask
    
    def _key_query_product(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Calculate the dot product between query and key vectors element by element.
        
        Args:
            q: Query tensor of shape (batch_size, sequence_length, key_dim)
            k: Key tensor of shape (batch_size, sequence_length, key_dim)
            
        Returns:
            Dot product tensor of shape (batch_size, sequence_length, sequence_length)
        """
        batch_size, sequence_length, _ = q.shape
        
        # Initialize the output tensor
        result = torch.zeros(batch_size, sequence_length, sequence_length)
        
        # Iterate through each batch
        for b in range(batch_size):
            # Iterate through each query position
            for i in range(sequence_length):
                # Iterate through each key position
                for j in range(sequence_length):
                    # Calculate the dot product between q[b,i] and k[b,j]
                    dot_product = (torch.dot(q[b, i], k[b, j])).item()

                    # Scale by square root of key dimension
                    result[b, i, j] = dot_product / np.sqrt(self.key_dim).item()
        
        return result
        
    def _compute_weights(self, query_key_product: torch.Tensor, final_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply the boolean attention mask and compute softmax weights with explicit loops.
        
        Args:
            query_key_product: Tensor of shape (B, S, S)
            final_mask: Boolean attention mask of shape (B, S, S), where True means keep.
            
        Returns:
            Attention weights tensor of shape (B, S, S)
        """
        batch_size, sequence_length, _ = query_key_product.shape
        
        weights = torch.zeros_like(query_key_product)

        for b in range(batch_size):
            for i in range(sequence_length):
                row_scores = query_key_product[b, i]
                row_mask = final_mask[b, i]
                
                masked_scores = row_scores[row_mask]
                
                if masked_scores.numel() > 0:
                    softmax_scores = F.softmax(masked_scores, dim=0)
                    weights[b, i, row_mask] = softmax_scores

        return weights

    def _compute_new_v(self, weights: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the weighted sum of values with explicit loops.
        
        Args:
            weights: Attention weights tensor of shape (batch_size, sequence_length, sequence_length)
            v: Value tensor of shape (batch_size, sequence_length, value_dim)
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, value_dim)
        """
        batch_size, sequence_length, _ = weights.shape
        
        # Initialize output tensor
        output = torch.zeros(batch_size, sequence_length, self.value_dim)
        
        # Compute weighted sum for each position in each batch
        for b in range(batch_size):
            for i in range(sequence_length):
                output[b, i, :] = weights[b, i, :] @ v[b, :, :]
                 

        return output

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process the input using naive loops instead of vectorized operations.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dimension)
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, input_dimension)
        """
        if x.ndim != 3:
            raise ValueError(f"The input tensor is expected to be of the shape (batch_size, sequence_length, input_dimension), but got {x.shape}")

        batch_size, sequence_length, __ = x.shape 

        # Apply linear transformations to get query, key, and value
        q = self.W_q.forward(x)  # (batch_size, sequence_length, key_dim)
        k = self.W_k.forward(x)  # (batch_size, sequence_length, key_dim)
        v = self.W_v.forward(x)  # (batch_size, sequence_length, value_dim)

        # Compute attention scores
        query_key_product = self._key_query_product(q, k)
        
        # Build masks
        causal_mask = self._causal_attention_mask(batch_size, sequence_length).to(q.device)
        # combine the causal mask with the padding mask if provided 
        final_mask = self.create_final_mask(causal_mask, mask.to(q.device) if mask is not None else None)
        
        # Compute attention weights
        weights = self._compute_weights(query_key_product, final_mask)
        
        # Compute weighted sum of values
        output = self._compute_new_v(weights, v)
        
        # Apply output projection
        final_output = self.W_o.forward(output)
        
        return final_output

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def children(self) -> list[nn.Module]:
        return [self.W_q, self.W_k, self.W_v, self.W_o]

    def named_children(self) -> list[tuple[str, nn.Module]]:
        return [("W_q", self.W_q), ("W_k", self.W_k), ("W_v", self.W_v), ("W_o", self.W_o)]

    def modules(self):
        yield self
        for child in self.children():
            yield from child.modules()

    def parameters(self):
        for child in self.children():
            yield from child.parameters()
