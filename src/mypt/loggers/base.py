import os, json, pickle
from typing import Any, Dict
from abc import ABC, abstractmethod

from mypt.code_utils import directories_and_files as dirf


class BaseLogger(ABC):
    """Abstract base class for logging."""

    @abstractmethod
    def __init__(self, log_dir: str, **kwargs):
        """
        Initializes the logger.
        Args:
            log_dir (str): The directory where logs will be stored.
        """
        log_dir = dirf.process_path(log_dir, dir_ok=True, file_ok=False, must_exist=False)
        self.log_dir = log_dir

    @abstractmethod
    def log_scalar(self, tag: str, value: float, step: int):
        """Logs a single scalar value."""
        pass

    @abstractmethod
    def log_dict(self, values: Dict[str, float], step: int):
        """Logs a dictionary of scalar values."""
        pass

    @abstractmethod
    def log_image(self, tag: str, image: Any, step: int, **kwargs):
        """Logs an image."""
        pass

    @abstractmethod
    def log_table(self, tag: str, data: Any, **kwargs):
        """Logs tabular data."""
        pass

    @abstractmethod
    def log_histogram(self, tag: str, values: Any, step: int):
        """Logs a histogram of values."""
        pass
    
    @abstractmethod
    def log_config(self, config: Dict, config_file_name: str):
        """Logs the configuration."""
        # the default implementation saves the config to a json file
        if not config_file_name.endswith(".json"):
            config_file_name = f"{config_file_name}.json"

        config_path = os.path.join(self.log_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)


    def save(self, object: Any, file_name: str):
        """Saves an object to a file."""

        # if the object is a dictionary, save it as a json file
        # otherwise, save it with pickle 

        if isinstance(object, dict):
            file_name = f"{file_name}{('.json' if not file_name.endswith('.json') else '')}"

            # save it as a json file
            file_path = os.path.join(self.log_dir, file_name)
            with open(file_path, 'w') as f:
                json.dump(object, f, indent=4)

            return
        
        # the object is not a dictionary; save it with pickle        
        file_name = f"{file_name}{('.obj' if not file_name.endswith('.obj') else '')}"
        file_path = os.path.join(self.log_dir, file_name)
        with open(file_path, 'wb') as f:
            pickle.dump(object, f)
        

    @abstractmethod
    def close(self):
        """Closes the logger and releases resources."""
        pass 