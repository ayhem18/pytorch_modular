from .base import BaseLogger
from .tensorboard_logger import TensorBoardLogger
from .wandb_logger import WandbLogger

def get_logger(name: str, **kwargs) -> BaseLogger:
    if name == 'tensorboard':
        return TensorBoardLogger(**kwargs)
    elif name == 'wandb':
        return WandbLogger(**kwargs)
    else:
        raise ValueError(f"Unknown logger: {name}") 