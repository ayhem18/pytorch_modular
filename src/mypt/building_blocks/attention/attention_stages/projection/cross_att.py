import torch
from torch import nn
from typing import Tuple

from .base import ProjectionBase
from mypt.building_blocks.mixins.general import NonSequentialModuleMixin


class CrossAttentionProjection(NonSequentialModuleMixin, ProjectionBase):
    """Projection stage for *cross-attention* (e.g. Transformer decoder).

    Queries are produced from the *query* sequence ``x`` whereas Keys and
    Values are produced from a separate *context* sequence.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
    ) -> None:
        ProjectionBase.__init__(self, num_heads=num_heads, key_dim=key_dim, value_dim=value_dim)
        NonSequentialModuleMixin.__init__(self, inner_components_fields=["W_q", "W_k", "W_v"])

        if query_dim % num_heads != 0:
            raise ValueError(
                f"query_dim ({query_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.query_dim = query_dim
        self.context_dim = context_dim
        self.num_heads = num_heads


        # Separate projection matrices for query and for context (keys/values)
        self.W_q = nn.Linear(query_dim, key_dim * num_heads)
        self.W_k = nn.Linear(context_dim, key_dim * num_heads)
        self.W_v = nn.Linear(context_dim, value_dim * num_heads)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().forward_impl(query, context, context)

    def process_query_src(self, query_src: torch.Tensor) -> torch.Tensor:
        return self._reshape(self.W_q(query_src), self.key_dim)

    def process_key_src(self, key_src: torch.Tensor) -> torch.Tensor:
        return self._reshape(self.W_k(key_src), self.key_dim)

    def process_value_src(self, value_src: torch.Tensor) -> torch.Tensor:
        return self._reshape(self.W_v(value_src), self.value_dim)


