import os
import cv2
import torch

import numpy as np

from PIL import Image
from pathlib import Path
from typing import Union
from matplotlib import pyplot as plt


def visualize(data: Union[str, Path, np.ndarray, torch.Tensor, Image.Image], 
              window_name: str = "Image", 
              wait_key: int = 0) -> None:
    """
    Visualizes different types of input data.
    
    Args:
        data: Input data to visualize. Can be:
            - str or Path: Path to an image file
            - np.ndarray: NumPy array representing an image
            - torch.Tensor: PyTorch tensor representing an image
        window_name: Name of the window to display the image in
        wait_key: Time in milliseconds to wait for a key event (0 means wait indefinitely)
    """
    # Case 1: Path or string input
    if isinstance(data, (str, Path)):
        if not os.path.exists(data):
            raise FileNotFoundError(f"File not found: {data}")
        
        # Read image as numpy array and recursively call this function
        img = cv2.imread(str(data))
        if img is None:
            raise ValueError(f"Could not read image from path: {data}")
        
        return visualize(img, window_name, wait_key)
    

    if isinstance(data, Image.Image):
        return visualize((np.array(data).copy()), window_name, wait_key)

    # Case 2: PyTorch tensor
    if isinstance(data, torch.Tensor):
        # Ensure tensor is on CPU and detached from computation graph
        data = data.detach().cpu()
        
        # Handle different tensor shapes
        if len(data.shape) == 2:  # Grayscale image: HxW
            img = data.numpy()
        elif len(data.shape) == 3:
            if data.shape[0] == 1:  # Grayscale image with channel: 1xHxW
                img = data.squeeze(0).numpy()
            elif data.shape[0] == 3:  # RGB image: 3xHxW
                img = data.permute(1, 2, 0).numpy()  # Convert to HxWx3
            else:
                raise ValueError(f"Unsupported tensor shape: {data.shape}")
        else:
            raise ValueError(f"Unsupported tensor dimensionality: {len(data.shape)}")
        
        return visualize(img, window_name, wait_key)
    
    if not isinstance(data, np.ndarray):
        raise ValueError(f"Unsupported data type: {type(data)}")

    # Normalize if the data type is float
    if data.dtype in [np.float32, np.float64]:
        if data.max() <= 1.0:
            data = (data * 255).astype(np.uint8)
        else:
            data = data.astype(np.uint8)
    
    # Handle different array shapes
    if len(data.shape) == 2:  # Grayscale image
        pass  # OpenCV can display 2D arrays directly
    elif len(data.shape) == 3:
        if data.shape[2] == 3:  # RGB image
            pass  # OpenCV expects HxWx3 format
        elif data.shape[2] == 1:  # Grayscale image with channel dimension
            data = data.squeeze(2)
        else:
            raise ValueError(f"Unsupported number of channels: {data.shape[2]}")
    else:
        raise ValueError(f"Unsupported array dimensionality: {len(data.shape)}")
    
    # # Display the image
    # cv2.imshow(window_name, data)
    # cv2.waitKey(wait_key)
    # cv2.destroyAllWindows()

    # display with matplotlib
    plt.imshow(data, cmap="gray" if data.ndim == 2 else None)
    plt.title(window_name)
    plt.show()
