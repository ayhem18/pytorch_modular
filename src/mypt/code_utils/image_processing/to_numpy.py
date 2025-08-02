import os
import torch
import numpy as np

from pathlib import Path
from typing import Union
from PIL import Image



def str_to_np(data: Union[str, Path]) -> np.ndarray:
    """Converts a file path to a NumPy array."""
    if not os.path.exists(data):
        raise FileNotFoundError(f"File not found: {data}")
    
    img = Image.open(data).convert("RGB")
    return np.array(img)

def pt_to_np(data: torch.Tensor) -> np.ndarray:
    """Converts a PyTorch Tensor to a NumPy array."""
    data = data.detach().cpu()
    
    if len(data.shape) == 2:  # Grayscale image: HxW
        return data.numpy()
    elif len(data.shape) == 3:
        if data.shape[0] == 1:  # Grayscale image with channel: 1xHxW
            return data.squeeze(0).numpy()
        elif data.shape[0] == 3:  # RGB image: 3xHxW
            return data.permute(1, 2, 0).numpy()  # Convert to HxWx3
        else:
            raise ValueError(f"Unsupported tensor shape: {data.shape}")
    else:
        raise ValueError(f"Unsupported tensor dimensionality: {len(data.shape)}")

def normalize_np_array_pixel_values(data: np.ndarray) -> np.ndarray:
    """Normalizes NumPy array pixel values to be in the [0, 255] range as uint8."""
    if data.dtype in [np.float32, np.float64]:
        if data.min() >= 0 and data.max() <= 1:
            data = (data * 255)

        if data.min() >= -1 and data.max() <= 1:
            data = (data + 1) / 2.0
        
        return data.astype(np.uint8)
    return data

def normalize_np_array_shape(data: np.ndarray) -> np.ndarray:
    """Normalizes the shape of a NumPy array for visualization."""
    if data.ndim not in [2, 3]:
        raise ValueError(f"Unsupported array dimensionality: {data.ndim}")
    
    if data.ndim == 3 and data.shape[2] == 1: # Grayscale with channel
        return data.squeeze(2)

    return data

def to_displayable_np(data: Union[str, Path, np.ndarray, torch.Tensor, Image.Image]) -> np.ndarray:
    """Converts any supported input into a displayable NumPy array."""
    if isinstance(data, (str, Path)):
        img_np = str_to_np(data)
    elif isinstance(data, Image.Image):
        img_np = np.array(data)
    elif isinstance(data, torch.Tensor):
        img_np = pt_to_np(data)
    elif isinstance(data, np.ndarray):
        img_np = data
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    img_np = normalize_np_array_pixel_values(img_np)
    img_np = normalize_np_array_shape(img_np)
    return img_np
