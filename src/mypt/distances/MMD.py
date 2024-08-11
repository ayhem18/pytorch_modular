"""
This script contains a simple implementation of the emperical estimate of the Maximum Mean Disprecancy measure as indicated in the 
equation '2' in the paper: "Unsupervised Domain Adaptation with Residual Transfer Networks" (https://arxiv.org/pdf/1602.04433.pdf)
"""

import torch
from torch import nn
from typing import Union
from tqdm import tqdm
from ..code_utilities import pytorch_utilities as pu
# let's implement this function as a Pytorch Loss

def _zero_diagonals(x: torch.Tensor) -> torch.Tensor: 
    # the main idea here is to set the diagonals to zero
    return x * (torch.ones(x.shape) - torch.eye(*x.shape)).to(pu.get_module_device(x))

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

        if x.shape[1] != y.shape[1]:
            raise ValueError(f"x and y must be of the same dimensions !!. Found: x: {x.shape}, y: {y.shape}")

        # use torch.linalg.vector_norm as its behavior is much more expectable 
        x_norm_squared = torch.broadcast_to(input=torch.linalg.vector_norm(x, dim=1, keepdim=True) ** 2, size=(x.shape[0], x.shape[0]))
        y_norm_squared = torch.broadcast_to(input=torch.linalg.vector_norm(y, dim=1, keepdim=True) ** 2, size=(y.shape[0], y.shape[0]))

        xx, xy, yy = x @ x.T, x @ y.T, y @ y.T
        # we know that the diagonals of the expression x_norm_squared + x_norm_squared.T - 2 * xx are zeros
        # thus to avoid producing the 'torch.inf' value we can set them manually to zero
        kxx = torch.exp(- _zero_diagonals(x_norm_squared + x_norm_squared.T - 2 * xx) / self.sigma)
        if torch.any(torch.logical_or(torch.isinf(kxx), torch.isnan(kxx))):
            raise ValueError(f"inf or nan detected in kxx")
        first_term = kxx.mean()
        
        kyy = torch.exp(- _zero_diagonals(y_norm_squared + y_norm_squared.T - 2 * yy) / self.sigma)
        if torch.any(torch.logical_or(torch.isinf(kyy), torch.isnan(kyy))):
            raise ValueError(f"inf or nan detected in kyy")
        second_term = kyy.mean()
        
        # before proceeding with calculating the intersection
        if x.shape[0] == y.shape[0]:
            kxy = torch.exp(- (x_norm_squared + y_norm_squared.T - 2 * xy) / self.sigma)
            if torch.any(torch.logical_or(torch.isinf(kxy), torch.isnan(kxy))):
                raise ValueError(f"inf or nan detected in kxy")
            third_term = kxy.mean()
        else:
            # the last term should 
            x_norm_y = torch.broadcast_to(input=torch.linalg.vector_norm(x, dim=1, keepdim=True) ** 2, size=(x.shape[0], y.shape[0]))
            y_norm_x = torch.broadcast_to(input=torch.linalg.vector_norm(y, dim=1, keepdim=True) ** 2, size=(y.shape[0], x.shape[0]))

            kxy = torch.exp(- (x_norm_y + y_norm_x.T - 2 * xy) / self.sigma)
            if torch.any(torch.logical_or(torch.isinf(kxy), torch.isnan(kxy))):
                raise ValueError(f"inf or nan detected in kxy")
            third_term = kxy.mean()
        
        return (first_term + second_term - 2 * third_term)

    def __call__(self, x: torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        return self.forward(x, y)


def naive_implementation(x: torch.Tensor, y: torch.Tensor, sigma: float):
    x, y = x.to(torch.float32), y.to(torch.float32)
    nx = len(x) 
    ny = len(y) 

    # first term
    first_term = 0
    for xi in x:
        for xj in x:
            # apply the kernel operation
            first_term += torch.exp(- (torch.linalg.vector_norm(xi - xj) ** 2) / sigma).item()
    first_term = first_term / (nx ** 2)

    second_term = 0
    for yi in y:
        for yj in y:
            # apply the kernel operation
            second_term += torch.exp(- (torch.linalg.vector_norm(yi - yj) ** 2) / sigma).item()
    second_term = second_term / (ny ** 2)

    third_term = 0
    for xi in x:
        for yi in y:
            # apply the kernel operation
            third_term += torch.exp(- (torch.linalg.vector_norm(xi - yi) ** 2) / sigma).item()
    third_term = third_term / (nx * ny)

    return first_term + second_term - 2 * third_term
