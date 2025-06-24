import os, json, pickle
from typing import Any, Dict

from mypt.code_utils import directories_and_files as dirf

from .base import BaseLogger

class PlaceholderLogger(BaseLogger):
    """Abstract base class for logging."""

    def __init__(self, log_dir: str):
        """
        Initializes the logger.
        Args:
            log_dir (str): The directory where logs will be stored.
        """
        super().__init__(log_dir)


    def log_scalar(self, tag: str, value: float, step: int):
        """Logs a single scalar value."""
        pass

    def log_dict(self, values: Dict[str, float], step: int):
        """Logs a dictionary of scalar values."""
        pass

    def log_image(self, tag: str, image: Any, step: int, **kwargs):
        """Logs an image."""
        pass

    def log_table(self, tag: str, data: Any, **kwargs):
        """Logs tabular data."""
        pass

    def log_histogram(self, tag: str, values: Any, step: int):
        """Logs a histogram of values."""
        pass
    
    def log_config(self, config: Dict, config_file_name: str):
        """Logs the configuration."""
        super().log_config(config, config_file_name)


    def save(self, object: Any, file_name: str):
        """Saves an object to a file."""
        super().save(object, file_name)


    def close(self):
        """Closes the logger and releases resources."""
        pass
