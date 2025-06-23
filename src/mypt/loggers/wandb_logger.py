import wandb
from typing import Dict, Any
from .base import BaseLogger

class WandbLogger(BaseLogger):
    def __init__(self, 
                 log_dir: str, 
                 project: str,
                 run_name: str,
                 **kwargs):
        super().__init__(log_dir)
        # wandb.init expects args like project, entity, name, config
        # We can pass them through kwargs
        wandb.init(dir=log_dir, project=project, name=run_name, **kwargs)

    def log_scalar(self, tag: str, value: float, step: int):
        wandb.log({tag: value}, step=step)

    def log_dict(self, values: Dict[str, float], step: int):
        wandb.log(values, step=step)

    def log_image(self, tag: str, image: Any, step: int, **kwargs):
        """
        Logs an image.
        'image' can be a PIL Image, numpy array, or torch tensor.
        """
        wandb.log({tag: wandb.Image(image, **kwargs)}, step=step)

    def log_table(self, tag: str, data: Any, step: int, **kwargs):
        """
        Logs tabular data.
        'data' can be a pandas DataFrame, or list of lists.
        """
        wandb.log({tag: wandb.Table(data=data, **kwargs)}, step=step)
        
    def log_histogram(self, tag: str, values: Any, step: int):
        wandb.log({tag: wandb.Histogram(values)}, step=step)

    def log_config(self, config: Dict, config_file_name: str = "config.json"):
        """Logs configuration to wandb."""
        # save the configuration
        super().log_config(config, config_file_name)
        wandb.config.update(config)


    def close(self):
        wandb.finish() 