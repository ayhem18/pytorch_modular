import os
import json
import torch
import numpy as np
import pandas as pd

from typing import Dict, Any, List, Union
from torch.utils.tensorboard import SummaryWriter

from .base import BaseLogger

class TensorBoardLogger(BaseLogger):
    def __init__(self, log_dir: str, **kwargs):
        super().__init__(log_dir, **kwargs)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, global_step=step)

    def log_dict(self, values: Dict[str, float], step: int):
        for tag, value in values.items():
            self.log_scalar(tag, value, step)
    
    def log_image(self, tag: str, image: Any, step: int):
        """
        Logs an image.
        Args:
            tag (str): The tag for the image.
            image (np.ndarray or torch.Tensor): The image to log.
            step (int): The step at which to log the image.
            dataformats (str): Image data format specification. Default is 'CHW'.
        """
        # set the dataformats based on the type of the image
        if isinstance(image, torch.Tensor):
            dataformats = 'CHW'
        elif isinstance(image, np.ndarray):
            dataformats = 'HWC'
        else:
            raise TypeError(f"Image must be a torch.Tensor or np.ndarray, got {type(image)}")

        self.writer.add_image(tag, image, step, dataformats=dataformats)
        
    def log_table(self, tag: str, data: Union[pd.DataFrame, List[List]], step: int):
        """
        Logs tabular data as text in Markdown format.
        'data' can be a pandas DataFrame or a list of lists.
        """
        try:
            if isinstance(data, pd.DataFrame):
                text_table = data.to_markdown()
            else: # assuming list of lists
                df = pd.DataFrame(data)
                text_table = df.to_markdown()
        except Exception as e:
            print(f"Error logging table to tensorboard: {e}")
            try:
                text_table = str(data) # fallback
            except TypeError:
                # raise the error again with a more informative message
                raise TypeError(f"The data can be converted to neither a pandas Dataframe, markdown nor a string")

        self.writer.add_text(tag, text_table, global_step=step)

    def log_histogram(self, tag: str, values: Any, step: int):
        self.writer.add_histogram(tag, values, global_step=step)

    def log_config(self, config: Dict, config_file_name: str):
        """Saves configuration to a JSON file in the log directory."""
        super().log_config(config, config_file_name)

    def close(self):
        self.writer.close() 