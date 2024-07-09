import torch
from typing import Optional

from ..schedulers.annealing_lr import AnnealingLR
from ..CBM.model.loss import CBMLoss, BinaryCBMLoss

def optimizer_str_to_cls(optimizer_str: Optional[str]) -> callable:
    if optimizer_str is None:
        return None
    
    if optimizer_str == 'adam':
        return torch.optim.Adam 
    
    if optimizer_str == 'sgd':
        return torch.optim.SGD

    raise ValueError(f"the following string is not associated with an optimizer class: {optimizer_str}") 


def scheduler_str_to_cls(scheduler_str: Optional[str]) -> callable:
    if scheduler_str is None:
        return None
    
    if scheduler_str == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR

    if scheduler_str == 'annealing':
        return AnnealingLR

    raise ValueError(f"the following string is not associated with an scheduler class: {scheduler_str}") 
    

def loss_str_to_cls(loss_str: Optional[str]) -> callable:
    if loss_str is None:
        return None
    
    if loss_str == 'cross_entropy':
        return CBMLoss

    if loss_str == 'binary':
        return BinaryCBMLoss

    raise ValueError(f"the following string is not associated with an scheduler class: {loss_str}") 
