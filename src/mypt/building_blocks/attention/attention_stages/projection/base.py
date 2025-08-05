from __future__ import annotations

import abc
from typing import Tuple, Optional

import torch
from torch import nn


class ProjectionBase(nn.Module, abc.ABC):
    """Abstract base class for the *Projection* stage of attention.

    The role of a Projection module is to map an incoming sequence
    (and optionally an external *context* sequence) into *query*,
    *key* and *value* tensors that will later be consumed by the
    second attention stage (Aggregator / Mixer).

    All subclasses must implement ``forward`` and return a tuple
    ``(q, k, v)`` with shapes::

        q : (B, H, S_q, D_k)
        k : (B, H, S_k, D_k)
        v : (B, H, S_k, D_v)

    where
        B:   batch size
        S_q: length of the *query* sequence (usually == S_k for self-attention)
        S_k: length of the *context/key* sequence
        H:   number of heads (1 for single-head)
        D_k: per-head dimensionality of keys / queries
        D_v: per-head dimensionality of values
    """

    def __init__(self, num_heads: int, key_dim: int, value_dim: int) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)


    def _reshape(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        """Reshape `(B, S, H*dim)` → `(B, H, S, dim)`"""
        B, S, _ = t.shape
        return t.view(B, S, self.num_heads, dim).permute(0, 2, 1, 3)

    
    @abc.abstractmethod 
    def process_query_src(self, query_src: torch.Tensor) -> torch.Tensor:
        """Process the query source sequence.

        Parameters
        ----------
        query_src : torch.Tensor
            Query sequence of shape ``(B, S_q, D_in)``.
        """
        pass

    @abc.abstractmethod 
    def process_key_src(self, key_src: torch.Tensor) -> torch.Tensor:
        """Process the key source sequence.

        Parameters
        ----------
        key_src : torch.Tensor
            Key sequence of shape ``(B, S_k, D_in)``.
        """
        pass


    @abc.abstractmethod 
    def process_value_src(self, value_src: torch.Tensor) -> torch.Tensor:
        """Process the value source sequence.

        Parameters
        ----------
        value_src : torch.Tensor
            Value sequence of shape ``(B, S_k, D_in)``.
        """
        pass


    @abc.abstractmethod
    def forward_impl(
        self,
        query_src: torch.Tensor,
        key_src: torch.Tensor,
        value_src: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Q,K,V projections.

        Parameters
        ----------
        query_src : torch.Tensor
            Query sequence of shape ``(B, S_q, D_in)``.
        key_src : torch.Tensor
            Key sequence of shape ``(B, S_k, D_in)``.
        value_src : torch.Tensor
            Value sequence of shape ``(B, S_k, D_in)``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The three tensors ``(q, k, v)`` with shapes described in the
            class docstring.
        """
        q = self.process_query_src(query_src) # (B, S_q, num_heads, D_k)
        k = self.process_key_src(key_src) # (B, S_k, num_heads, D_k)
        v = self.process_value_src(value_src) # (B, S_k, num_heads, D_v)

        return q, k, v


    @abc.abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        """
        pass

