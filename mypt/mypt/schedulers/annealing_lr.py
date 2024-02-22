"""
This script contains the implementation of the annealing learning rate scheduling strategy according to the ReverseGrad paper    
"""

import torch
from typing import Union
from torch.optim.lr_scheduler import LambdaLR

class AnnealingLR(LambdaLR):    
    def __init__(self, 
                 optimizer: torch.optim,
                 num_epochs: int, 
                 alpha: Union[float, int],
                 beta: Union[float, int], 
                 verbose:bool=False
                 ):
        # let's make some checks on the passes arguments
        if num_epochs <= 0:
            raise ValueError(f"The number of epochs must be a positive integer. Found: {num_epochs}")

        if beta > 1: 
            raise ValueError(f"The parameter beta must be at most 1. Found: {beta}")

        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.alpha = alpha
        self.beta = beta    

        def _formula(epoch: int): 
            # first calculate 'p'
            p = epoch / self.num_epochs
            lr = 1 / ((1 + self.alpha * p) ** self.beta)
            return lr

        self.scheduler = LambdaLR(optimizer=optimizer, 
                                  lr_lambda=[_formula for _ in optimizer.param_groups], # this is done internally in the scheduler class 
                                  verbose=verbose)

    # the main idea is to override the LambdaLR function by calling those of the self.scheduler field
    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)

    def get_lr(self):
        self.scheduler.get_lr()

    def step(self):
        self.scheduler.step()

from torch import nn

if __name__ == '__main__':
    m = nn.Linear(in_features=10, out_features=1)
    opt = torch.optim.SGD(m.parameters(), lr=10 ** -3)
    s = AnnealingLR(optimizer=opt, num_epochs=20, beta=1, alpha=2)    
    print(opt.param_groups[0]['lr'])
    for _ in range(20):
        opt.step()
        before_lr = opt.param_groups[0]['lr']
        s.step()
        after_lr = opt.param_groups[0]['lr']
        print(f"before and after the scheduling: {before_lr}, {after_lr}") 
        print("#" * 20)
    