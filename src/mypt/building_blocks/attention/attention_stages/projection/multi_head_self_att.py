from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from mypt.building_blocks.mixins.general import NonSequentialModuleMixin
from .base import ProjectionBase


class MultiHeadSelfAttentionProjection(NonSequentialModuleMixin, ProjectionBase):
    """Projection stage for standard *multi-head* self-attention.

    Each of Q, K and V is produced by an independent linear layer that
    outputs ``num_heads * dim`` features, which are then reshaped to
    ``(B, H, S, dim)``.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
    ) -> None:
        ProjectionBase.__init__(self, num_heads=num_heads, key_dim=key_dim, value_dim=value_dim)
        NonSequentialModuleMixin.__init__(self, inner_components_fields=["W_q", "W_k", "W_v"])

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )

        self.d_model = d_model

        self.W_q = nn.Linear(d_model, key_dim * num_heads)
        self.W_k = nn.Linear(d_model, key_dim * num_heads)
        self.W_v = nn.Linear(d_model, value_dim * num_heads)

    
    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().forward_impl(x, x, x)

    def process_query_src(self, query_src: torch.Tensor) -> torch.Tensor:
        return super()._reshape(self.W_q(query_src), self.key_dim)

    def process_key_src(self, key_src: torch.Tensor) -> torch.Tensor:
        return super()._reshape(self.W_k(key_src), self.key_dim)

    def process_value_src(self, value_src: torch.Tensor) -> torch.Tensor:
        return super()._reshape(self.W_v(value_src), self.value_dim)


