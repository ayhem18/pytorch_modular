"""
Unlike helper_functionalities.py, this script contains, Pytorch code that is generally used across different
scripts and Deep Learning functionalities
"""

import gc, torch, os, random, re
import numpy as np

from torch import nn
from typing import Union, List, Optional
from pathlib import Path
from datetime import datetime as d
from torch.optim.optimizer import Optimizer
from torchvision import transforms as tr
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


def save_checkpoint(model: nn.Module, 
                    optimizer: Optimizer, 
                    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
                    path: Union[str, Path], 
                    **kwargs):

    ckpnt_dict = {"model_state_dict": model.state_dict(), 
                  "optimizer_state_dict": optimizer.state_dict()}   

    if lr_scheduler is not None:
        ckpnt_dict["lr_scheduler_state_dict"] = lr_scheduler.state_dict()

    # add any extra keywords to be saved with the checkpoints
    ckpnt_dict.update(kwargs)
    torch.save(ckpnt_dict, path)

    # make sure the path is correctly created
    path = process_path(save_path=path, dir_ok=False, file_ok=True, 
                        condition=lambda p : os.path.splitext(p)[-1] in ['.pt', '.pnt'], 
                        error_message="Make sure the checkpoint has the correct extension")



# let's define functionalities for reproducibility and random seeds

def seed_everything(seed: int = 69):
    # let's set reproducility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed=seed)
    torch.use_deterministic_algorithms(True, warn_only=True) # certain layers have no deterministic implementation... we'll need to compromise on this one...
    torch.backends.cudnn.benchmark = False 

    # the final step to ensure reproducibility is to set the environment variable: # CUBLAS_WORKSPACE_CONFIG=:16:8
    import warnings
    # first check if the CUBLAS_WORSKPACE_CONFIG variable is set or not
    env_var = os.getenv('CUBLAS_WORKSPACE_CONFIG')
    if env_var is None:
        # the env variable was not set previously
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8' # this is not the only viable value. I cannot remember the other value at the time of writing this code.
    else:
        if env_var not in [':16:8']:
            warnings.warn(message=f"the env variable 'CUBLAS_WORKSPACE_CONFIG' is set to the value {env_var}. setting it to: ':16:8' ")
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'        

    # pytorch lightning keeps issuing this warning every time: 
    # You are using a CUDA device ('NVIDIA A100-SXM4-80GB') that has Tensor Cores. To properly utilize them, 
    # you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. 
    # For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision

    # check if the device is gpu
    if 'cuda' in get_default_device():
        # it does not seem like a bad option
        torch.set_float32_matmul_precision('medium')


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
    

def get_augmentation_name(aug):
    aug_cls_name = re.sub("^[^\s\w]*", "", str(aug.__class__))
    aug_cls_name = re.sub("[^\s\w]*$", "", aug_cls_name)
    return re.split("\W+", aug_cls_name)[-1]


