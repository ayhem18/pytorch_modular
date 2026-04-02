from .base import BaseLogger

# only import the packages when needed

def get_logger(name: str, **kwargs) -> BaseLogger:
    if name == 'tensorboard':
        try:
            from .tensorboard_logger import TensorBoardLogger
        except ImportError:
            raise ImportError("TensorBoardLogger is not installed. Please install it with `pip install tensorboard`.")
        return TensorBoardLogger(**kwargs)
    elif name == 'wandb':
        try:
            from .wandb_logger import WandbLogger
        except ImportError:
            raise ImportError("WandbLogger is not installed. Please install it with `pip install wandb`.")
        return WandbLogger(**kwargs)
    elif name is None or name == 'placeholder':
        from .placeholder_logger import PlaceholderLogger
        return PlaceholderLogger(**kwargs)
    else:
        raise ValueError(f"Unknown logger: {name}") 