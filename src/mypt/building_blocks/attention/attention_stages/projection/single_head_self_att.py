from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from mypt.building_blocks.mixins.general import NonSequentialModuleMixin
from .base import ProjectionBase


class SingleHeadSelfAttentionProjection(NonSequentialModuleMixin, ProjectionBase):
    """Projection stage for classic *single-head* self-attention.

    It produces queries, keys and values from the same sequence `x`.
    All three linear projections are distinct but share the same input
    dimensionality.
    """

    def __init__(self, input_dim: int, key_dim: int, value_dim: int) -> None:
        ProjectionBase.__init__(self, num_heads=1, key_dim=key_dim, value_dim=value_dim)
        NonSequentialModuleMixin.__init__(self, inner_components_fields=["W_q", "W_k", "W_v"])

        self.W_q = nn.Linear(input_dim, key_dim)
        self.W_k = nn.Linear(input_dim, key_dim)
        self.W_v = nn.Linear(input_dim, value_dim)

    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute single-head projections according to the new interface."""
        # in self-attention, the query, key and value are the same sequence
        return super().forward_impl(x, x, x)   

    def process_query_src(self, query_src: torch.Tensor) -> torch.Tensor:
        return self.W_q(query_src).unsqueeze(1)

    def process_key_src(self, key_src: torch.Tensor) -> torch.Tensor:
        return self.W_k(key_src).unsqueeze(1)

    def process_value_src(self, value_src: torch.Tensor) -> torch.Tensor:
        return self.W_v(value_src).unsqueeze(1)


