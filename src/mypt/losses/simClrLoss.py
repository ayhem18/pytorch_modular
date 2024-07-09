"""
This script contains the implementation of the Simclr Loss suggested by the paper: "A Simple Framework for Contrastive Learning of Visual Representations"
(https://arxiv.org/abs/2002.05709)
"""

import torch

from torch import nn
from typing import List

from ..similarities import cosineSim


class SimClrLoss(nn.Module):
    _sims = ['cos', 'dot']

    @classmethod
    def _default_build_indices(cls, n:int) -> List[List[int]]:
        # the idea here is to find the indices of the pairs of same images
        return [[i, (i + n) % 2 * n] for i in range(2 * n)]        

    def __init__(self,
                 temperature: float, 
                 similarity: str='cos',
                 build_indices: function = None) -> None:
        if similarity not in self._sims:
            raise NotImplementedError(f"The current implementation supports only a specific set of similarity measures: {self._sims}. Found: {similarity}")
        self.sim = similarity 
        self.temp = temperature
        
        if build_indices is None:
            self._build_indices = self._default_build_indices
        else:
            self._build_indices = build_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # let's first check a couple of things: 
        if x.ndim != 2:
            raise ValueError(f"The current implementation only accepts 2 dimensional input. Found: {x.ndim} -dimensional input.")

        # make sure the number of samples is even
        if len(x) % 2 != 0:
            raise ValueError(f"The number of samples must be even. found: {len(x)} samples")

        N = len(x) // 2
        # this implementation assumes that x[i] and x[i + N] represent the same image (under differnet augmentations)
        
        # step1: calculate the similarities between the different samples
        # the entry [i, j] of 'exp_sims' variable contains the exp(sim(z_i, z_j)) / t
        exp_sims = torch.exp(cosineSim(x, x) / self.temp) 

        pairs_indices = self._build_indices(N)
        # the i-th entry contains exp(sim(x_i, x_{i + N})): the similarity between the two augmentations
        positive_pairs_exp_sims = exp_sims[pairs_indices] 

        # the i-th entry contains sum(exp(sim(x_i, x_k))) for k in [1, 2N] k != i
        exp_sims_sums = torch.sum(exp_sims, dim=1, keepdim=True) - torch.exp(torch.ones(size=(2 * N, 1)) / self.temp)
        
        loss = torch.mean(-torch.log(positive_pairs_exp_sims / exp_sims_sums))

        return loss
