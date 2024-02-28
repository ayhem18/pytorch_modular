"""
This script contains a simple implementation of the emperical estimate of the Maximum Mean Disprecancy measure as indicated in the 
equation '2' in the paper: "Unsupervised Domain Adaptation with Residual Transfer Networks" (https://arxiv.org/pdf/1602.04433.pdf)
"""

import torch
from torch import nn
from typing import Union

# let's implement this function as a Pytorch Loss


class GaussianMMD(nn.Module):
    def __init__(
                self,
                sigma: float,
                *args, 
                **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if sigma <= 0:
            raise ValueError(f"sigma must be positive. Found: {sigma}")
        self.sigma = sigma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # convert the input to float if needed 
        x, y = x.to(torch.float32), y.to(torch.float32)

        if x.ndim != 2 or y.ndim != 2:
            raise ValueError(f"This function expects the input to be 2 dimensional. Found: x: {x.shape}, y: {y.shape}")
        # the first term to calculate is: 
        # sum_{i=1, ns} sum_{j=1, ns} (k(xi, xj)) / ns^2 where k(xi, xj) = exp(- ||xi - xj || ^ 2 / sigma ) where |xi - xj| ^ 2 = |xi|^2 + |xj|^2 - 2 <xi, xj> 
        x_norm = torch.broadcast_to(input=torch.linalg.norm(x, ord=2, dim=1, keepdim=True) ** 2, size=(x.shape[0], x.shape[0])) 
        y_norm = torch.broadcast_to(input=torch.linalg.norm(y, ord=2, dim=1, keepdim=True) ** 2, size=(y.shape[0], y.shape[0]))
        xx, xy, yy = x @ x.T, x @ y.T, y @ y.T 
         
        kxx = torch.exp(- (x_norm + x_norm.T - 2 * xx) / self.sigma)
        first_term = kxx.mean()
        
        kxy = torch.exp(- (y_norm + y_norm.T - 2 * yy) / self.sigma)
        second_term = kxy.mean()
        
        kyy = torch.exp(- (x_norm + y_norm.T - 2 * xy) / self.sigma)
        third_term = kyy.mean()

        return first_term + second_term - 2 * third_term


# let's make sure things are going as expected

def naive_implementation(x: torch.Tensor, y: torch.Tensor, sigma: float):
    x, y = x.to(torch.float32), y.to(torch.float32)
    nx = len(x) 
    ny = len(y) 

    # first term
    first_term = 0
    for xi in x:
        for xj in x:
            # apply the kernel operation
            first_term += torch.exp(- (torch.linalg.norm(xi - xj) ** 2) / sigma).item()
    first_term = first_term / (nx ** 2)

    second_term = 0
    for yi in y:
        for yj in y:
            # apply the kernel operation
            second_term += torch.exp(- (torch.linalg.norm(yi - yj) ** 2) / sigma).item()
    second_term = second_term / (ny ** 2)

    third_term = 0
    for xi in x:
        for yi in y:
            # apply the kernel operation
            third_term += torch.exp(- (torch.linalg.norm(xi - yi) ** 2) / sigma).item()
    third_term = third_term / (nx * ny)

    return first_term + second_term - 2 * third_term

import random
import numpy as np

if __name__ == '__main__':
    for _ in range(100):
        x = torch.round(torch.randn(size=(4, 10)), decimals=2)
        y = torch.round(torch.randn(size=(4, 10)), decimals=2)
        sigma = round(0.5 + random.random() * 2, 2)

        mmd = GaussianMMD(sigma=sigma).forward(x, y).item()
        naive_mmd = naive_implementation(x, y, sigma=sigma) 

        print(np.abs(mmd - naive_mmd))

        if not np.isclose(mmd, naive_mmd, atol=10 ** -6):
            raise ValueError(f"Please make sure the code is written correctly. naive: {naive_mmd}, vectorized: {mmd}")
