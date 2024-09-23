"""
This script contains the implementation of the Simclr Loss suggested by the paper: "A Simple Framework for Contrastive Learning of Visual Representations"
(https://arxiv.org/abs/2002.05709)
"""

import torch

from torch import nn
from typing import List, Tuple


from ..code_utilities import pytorch_utilities as pu
from ..similarities.cosineSim import CosineSim

class SimClrLoss(nn.Module):
    _sims = ['cos', 'dot']

    @classmethod
    def _default_build_indices(cls, n:int) -> List[List[int]]:
        # this implementation assumes that x[i] and x[i + N] represent the same image (under differnet augmentations)
        return [i for i in range(2 * n)], [(i + n) % (2 * n) for i in range(2 * n)]


    def __init__(self,
                 temperature: float, 
                 debug:bool=False,
                 similarity: str='cos',
                 build_indices=None) -> None:
        # this call is super duper important: inherits the properties of the torch.nn.Module
        # might not be necessary when using pytorch but raises a very confusing error with pytorch Lightning
        super().__init__()

        if similarity not in self._sims:
            raise NotImplementedError(f"The current implementation supports only a specific set of similarity measures: {self._sims}. Found: {similarity}")
        self.sim = similarity 
        self.temp = temperature
        self.debug=debug

        if build_indices is None:
            self._build_indices = self._default_build_indices
        else:
            self._build_indices = build_indices


    def _set_exp_sims(self, x: torch.Tensor) -> torch.Tensor:
        if self.sim == self._sims[0]:
            return torch.exp(CosineSim().forward(x, x) / self.temp)

        return torch.exp((x @ x.T) / self.temp)        


    def forward(self, x: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # let's first check a couple of things: 
        if x.ndim != 2:
            raise ValueError(f"The current implementation only accepts 2 dimensional input. Found: {x.ndim} -dimensional input.")

        # make sure the number of samples is even
        if len(x) % 2 != 0:
            raise ValueError(f"The number of samples must be even. found: {len(x)} samples")

        N = len(x) // 2
        
        # step1: calculate the similarities between the different samples
        # the entry [i, j] of 'exp_sims' variable contains the exp(sim(z_i, z_j)) / t
        exp_sims = self._set_exp_sims(x)

        p1, p2 = self._build_indices(N)
        # the i-th entry contains exp(sim(x_i, x_{i + N})): the similarity between the two augmentations
        positive_pairs_exp_sims = exp_sims[p1, p2]

        # make sure positive_paris_exp_sims is of shape (N, 1)
        if positive_pairs_exp_sims.shape not in [(2 * N,), (2 * N, 1)]:
            raise ValueError(f"Make sure indexing for the positive pairs is correct")

        if positive_pairs_exp_sims.ndim == 1:
            positive_pairs_exp_sims = torch.unsqueeze(positive_pairs_exp_sims, 1)        


        # the i-th entry contains sum(exp(sim(x_i, x_k))) for k in [1, 2N] k != i
        exp_sims_sums = torch.sum(exp_sims, dim=1, keepdim=True) - torch.unsqueeze(torch.diag(exp_sims, diagonal=0), dim=1) 

        if self.debug:
            # for less headache, apply the log operation to get the actual similarities
            return (torch.mean(-torch.log(positive_pairs_exp_sims / exp_sims_sums)), 
                        # positive_pairs_exp_sims contains exp(sim(p1, p2) / temp), to extract the similarity
                        # apply log, then * self.temp 
                        (torch.log(positive_pairs_exp_sims.detach()) * self.temp).cpu(), 
                        
                        # exp_sims_sums is a bit trickier, as it is a sum of exp(sim(item, neg_sample) / temp)
                        # average first, apply log, * self.temp
                        (torch.log(exp_sims_sums.detach() / (2 * N - 1)) * self.temp).cpu()
                    )

        return torch.mean(-torch.log(positive_pairs_exp_sims / exp_sims_sums))



# let's write a naive implementation of the code: non-vectorized for loops implementations
# this class is not meant to be used. It is written solely for testing purposes and to 
# make sure the vectorized efficient implementation above is equivalent to the one introduced in the paper
class _SimClrLossNaive(nn.Module):
    _sims = ['cos', 'dot']

    def __init__(self,
                 temperature: float, 
                 similarity: str='cos') -> None:
        if similarity not in self._sims:
            raise NotImplementedError(f"The current implementation supports only a specific set of similarity measures: {self._sims}. Found: {similarity}")
        self.sim = similarity 
        self.temp = temperature
        

    def _set_exp_sims(self, x: torch.Tensor) -> torch.Tensor:
        if self.sim == self._sims[0]:
            return torch.exp(CosineSim().forward(x, x) / self.temp)

        return torch.exp(x @ x.T / self.temp)        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # let's first check a couple of things: 
        if x.ndim != 2:
            raise ValueError(f"The current implementation only accepts 2 dimensional input. Found: {x.ndim} -dimensional input.")

        # make sure the number of samples is even
        if len(x) % 2 != 0:
            raise ValueError(f"The number of samples must be even. found: {len(x)} samples")

        N = len(x) // 2
        
        # step1: calculate the similarities between the different samples
        # the entry [i, j] of 'exp_sims' variable contains the exp(sim(z_i, z_j)) / t
        exp_sims = self._set_exp_sims(x) 

        loss = 0
        for i in range(2 * N):
            # this is the similarity expression between x_{2k - 1} and x_{2k}
            pos_pair_sim = exp_sims[i, (i + N) % (2 * N)]
            pair_pair_sim = exp_sims[i, i]
            all_sample_sim = exp_sims[i].sum()
            loss += -torch.log(pos_pair_sim / (all_sample_sim - pair_pair_sim))

        # average the loss
        loss = loss / (2 * N)
        return loss
