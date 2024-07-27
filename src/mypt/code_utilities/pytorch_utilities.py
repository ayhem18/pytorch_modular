"""
Unlike helper_functionalities.py, this script contains, Pytorch code that is generally used across different
scripts and Deep Learning functionalities
"""

import gc, torch, os, random

import numpy as np

from torch import nn
from typing import Union, List, Optional
from pathlib import Path
from datetime import datetime as d
from torch.optim.optimizer import Optimizer

from .directories_and_files import process_path

HOME = os.path.dirname(os.path.realpath(__file__))

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# set the default device
def get_default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_module_device(module: nn.Module) -> str:
    # this function is mainly inspired by this overflow post:
    # https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently
    if hasattr(module, 'device'):
        return module.device
    device = next(module.parameters()).device
    return device

def __verify_extension(p):
    return os.path.basename(p).endswith('.pt') or os.path.basename(p).endswith('.pth')



def save_checkpoint(model: nn.Module, 
                    optimizer: Optimizer, 
                    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
                    path: Union[str, Path], 
                    **kwargs):

    path = process_path(path, file_ok=True, dir_ok=False, condition=lambda p : os.path.splitext(p)[-1] in ['.pt', '.pnt'], error_message="Make sure the checkpoint has the correct extension")

    ckpnt_dict = {"model_state_dict": model.state_dict(), 
                  "optimizer_state_dict": optimizer.state_dict(), 
                }   

    if lr_scheduler is not None:
        ckpnt_dict["lr_scheduler_state_dict"] = lr_scheduler.state_dict()

    # add any extra keywords to be saved with the checkpoints
    ckpnt_dict.update(kwargs)
    torch.save(ckpnt_dict, path)



def save_model(model: nn.Module, path: Union[str, Path] = None) -> None:
    # the time of saving the model
    now = d.now()
    file_name = "-".join([str(now.month), str(now.day), str(now.hour), str(now.minute)])
    # add the extension
    file_name += '.pt'

    # first check if the path variable is None:
    path = path if path is not None else os.path.join(HOME, file_name)

    # process the path
    path = process_path(path,
                             dir_ok=True,
                             file_ok=True,
                             condition=lambda p: not os.path.isfile(p) or __verify_extension(p),
                             error_message='MAKE SURE THE FILE PASSED IS OF THE CORRECT EXTENSION')

    if os.path.isdir(path):
        path = os.path.join(path, file_name)

    # finally save the model.
    torch.save(model.state_dict(), path)


def load_model(base_model: nn.Module,
               path: Union[str, Path]) -> nn.Module:
    # first process the path
    path = process_path(path,
                             dir_ok=False,
                             file_ok=True,
                             condition=lambda p: not os.path.isfile(p) or __verify_extension(p),
                             error_message='MAKE SURE THE FILE PASSED IS OF THE CORRECT EXTENSION')

    base_model.load_state_dict(torch.load(path))

    return base_model


# let's define functionalities for reproducibility

def seed_everything(seed: int = 69):
    # let's set reproducility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed=seed)
    torch.use_deterministic_algorithms(True, warn_only=True) # certain layers have no deterministic implementation... well need to compromise on this one...
    torch.backends.cudnn.benchmark = False    
    
def set_worker_seed(_, seed: int = 69):
    np.random.seed(seed=seed)
    random.seed(seed)


# let's define functionalities to iterate through models
def iterate(module: nn.Module) -> List[nn.Module]:
    children_nodes = []
    # if the module has no children node
    if len(list(module.children())) == 0:
        return [module]

    for c in module.children():
        children_nodes.extend(iterate(c))        

    return children_nodes
    
