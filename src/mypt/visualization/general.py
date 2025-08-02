import math, torch
import numpy as np

from PIL import Image
from pathlib import Path
from typing import Union, Iterable
from matplotlib import pyplot as plt

from mypt.code_utils.image_processing.to_numpy import to_displayable_np 

def visualize(data: Union[str, Path, np.ndarray, torch.Tensor, Image.Image], 
              window_name: str = "Image") -> None:
    """Visualizes different types of input data."""
    img_np = to_displayable_np(data)
    
    plt.imshow(img_np, cmap="gray" if img_np.ndim == 2 else None)
    plt.title(window_name)
    plt.axis('off')
    plt.show()

def _get_grid_shape(n: int) -> tuple[int, int]:
    """
    Calculates a grid shape (rows, cols) for a given number of items 'n'.
    The grid is made to be as square as possible.
    """
    if n <= 0: return (0, 0)
    if n == 1: return (1, 1)
    
    sqrt_n = int(math.sqrt(n))
    rows = 1
    for i in range(sqrt_n, 1, -1):
        if n % i == 0:
            rows = i
            break
            
    cols = n // rows
    return rows, cols

def visualize_grid(images: Iterable[Union[str, Path, np.ndarray, torch.Tensor, Image.Image]], 
                   title: str = "Image Grid") -> None:
    """Visualizes a collection of images in a grid."""
    image_list = list(images)
    n = len(image_list)
    
    if n == 0:
        print("No images to visualize.")
        return
        
    rows, cols = _get_grid_shape(n)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    fig.suptitle(title)
    
    if n == 1:  
        axes = [axes]
    else:
        axes = axes.flatten()
        
    for i, img_data in enumerate(image_list):
        ax = axes[i]
        img_np = to_displayable_np(img_data)
        
        ax.imshow(img_np, cmap="gray" if img_np.ndim == 2 else None)
        ax.axis('off')
        
    for i in range(n, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()
