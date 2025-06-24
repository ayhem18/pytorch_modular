from .base import BaseLogger
from .wandb_logger import WandbLogger
from .tensorboard_logger import TensorBoardLogger
from .placeholder_logger import PlaceholderLogger

def get_logger(name: str, **kwargs) -> BaseLogger:
    if name == 'tensorboard':
        return TensorBoardLogger(**kwargs)
    elif name == 'wandb':
        return WandbLogger(**kwargs)
    elif name is None or name == 'placeholder':
        return PlaceholderLogger(**kwargs)
    else:
        raise ValueError(f"Unknown logger: {name}") 