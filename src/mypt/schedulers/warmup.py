"""
the main aim of this script is to create a learning rate scheduler with warmups
"""

from typing import List, Union
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer

from mypt.code_utils import pytorch_utils as pu
from mypt.schedulers.annealing_lr import AnnealingLR


_DEFAULT_LRS = ['linear',] # more would be added gradually



def _map_str_2_lr_scheduler(optimizer: Optimizer, 
                            lrs_str: str, 
                            num_warmup_epochs: int, 
                            **kwargs) -> LRScheduler: 
    if lrs_str not in _DEFAULT_LRS:
        raise NotImplementedError(f"The library currently supports only the following strings for learning rate schedulers: {_DEFAULT_LRS}")

    if lrs_str == 'linear':
        return LinearLR(optimizer=optimizer, 
                        total_iters=num_warmup_epochs, 
                        start_factor=kwargs.get('start_factor', 0.01),
                        end_factor=kwargs.get('end_factor', 1),
                        )
    

def set_warmup_epochs(
                       optimizer: Optimizer,
                       main_lr_scheduler: LRScheduler, 
                       num_warmup_epochs: int,
                       warmup_lr_scheduler: Union[str, LRScheduler]) -> SequentialLR:

    # get the linear scheduler
    if isinstance(warmup_lr_scheduler, str):
        warmup_lr_scheduler = _map_str_2_lr_scheduler(optimizer=optimizer, 
                                            lrs_str=warmup_lr_scheduler, 
                                            num_warmup_epochs=num_warmup_epochs)

    final_lr_scheduler = SequentialLR(optimizer=optimizer, 
                                schedulers=[warmup_lr_scheduler, main_lr_scheduler], 
                                milestones=[num_warmup_epochs])

    return final_lr_scheduler

