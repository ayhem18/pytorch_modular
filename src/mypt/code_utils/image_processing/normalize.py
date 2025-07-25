"""
This script contains functions for normalizing images, including converting
from high bit-depth (e.g., 16-bit) to 8-bit for visualization or processing.
"""
import numpy as np
import cv2

def linear_normalize(image: np.ndarray, target_min: int = 0, target_max: int = 255) -> np.ndarray:
    """
    Scales an image to a target range using linear normalization (min-max scaling).
    This is the most common method for converting 16-bit to 8-bit.
    Args:
        image: Input image as a numpy array.
        target_min: The minimum value of the target range.
        target_max: The maximum value of the target range.
    Returns:
        Normalized image as a numpy array of type uint8.
    """
    # Using cv2.normalize is efficient
    return cv2.normalize(image, None, target_min, target_max, cv2.NORM_MINMAX, dtype=cv2.CV_8U) # type: ignore

# def clahe_normalize(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
#     """
#     Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) and then scales to 8-bit.
#     Excellent for enhancing local contrast in images.
#     Args:
#         image: Input image as a numpy array (works best on grayscale).
#         clip_limit: Threshold for contrast limiting.
#         tile_grid_size: Size of the grid for histogram equalization.
#     Returns:
#         Enhanced and normalized 8-bit image.
#     """
#     if image.dtype not in ['uint8', 'uint16']:
#         raise ValueError("CLAHE is often applied to 16-bit or 8-bit images. Ensure input range is appropriate.")

#     if image.ndim > 2:
#         # Applying to each channel if it is a color image
#         if image.dtype == 'uint8':
#             img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#             l, a, b = cv2.split(img_lab)
#             clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
#             cl = clahe.apply(l)
#             limg = cv2.merge((cl,a,b))
#             return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
#         else: # for 16 bit color images
#              print("Warning: CLAHE on color 16-bit images is not standard. Converting to grayscale first is recommended.")
#              return clahe_normalize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))


#     clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
#     # Apply CLAHE
#     if image.dtype == 'uint8':
#         clahe_image = clahe.apply(image)
#     else: # Apply to 16-bit and then normalize
#         clahe_image_16bit = clahe.apply(image)
#         clahe_image = linear_normalize(clahe_image_16bit)

#     return clahe_image


def gamma_normalize(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Applies gamma correction to an image and scales it to 8-bit.
    Gamma < 1.0 brightens shadows; Gamma > 1.0 darkens highlights.
    Args:
        image: Input image as a numpy array.
        gamma: The gamma value for correction.
    Returns:
        Gamma-corrected and normalized 8-bit image.
    """
    if gamma <= 0:
        raise ValueError("Gamma must be positive.")
        
    # Normalize to [0, 1] range first
    img_min, img_max = np.min(image), np.max(image)
    if img_max == img_min:
        raise ValueError("The image has no range of values. There might be an issue with the image.")

    img_norm = (image - img_min) / (img_max - img_min)
    
    # Apply gamma correction
    gamma_corrected = np.power(img_norm, 1.0 / gamma)
    
    # Scale to 8-bit
    normalized_image = (gamma_corrected * 255).astype(np.uint8)
    return normalized_image


def log_normalize(image: np.ndarray) -> np.ndarray:
    """
    Applies logarithmic scaling to an image and normalizes to 8-bit.
    Useful for images with high dynamic range where details in dark regions are important.
    Args:
        image: Input image as a numpy array.
    Returns:
        Log-scaled and normalized 8-bit image.
    """
    # Normalize to [0, max_val] to handle log(0)
    img_min, img_max = np.min(image), np.max(image)
    if img_max == img_min:
        raise ValueError("The image has no range of values. There might be an issue with the image.")

    img_norm = (image - img_min) / (img_max - img_min)

    # Apply log transform, add epsilon to avoid log(0)
    log_transformed = np.log1p(img_norm) # log1p is log(1+x)
    
    # Scale to 8-bit
    normalized_image = linear_normalize(log_transformed)
    return normalized_image


def sigmoid_normalize(image: np.ndarray, gain: float = 10.0, cutoff: float = 0.5) -> np.ndarray:
    """
    Applies sigmoidal contrast adjustment to an image and scales to 8-bit.
    Increases contrast in the mid-tones while compressing shadows and highlights.
    Args:
        image: Input image as a numpy array.
        gain: Controls the steepness of the curve (contrast). Higher is more contrasty.
        cutoff: The center of the transition (mid-point of the dynamic range, 0-1).
    Returns:
        Sigmoid-adjusted and normalized 8-bit image.
    """
    # Normalize to [0, 1] range first
    img_min, img_max = np.min(image), np.max(image)

    if img_max == img_min:
        raise ValueError("The image has no range of values. There might be an issue with the image.")
    
    img_norm = (image - img_min) / (img_max - img_min)
    
    # Apply sigmoid function
    sigmoid_image = 1 / (1 + np.exp(gain * (cutoff - img_norm)))
    
    # Scale to 8-bit
    normalized_image = (sigmoid_image * 255).astype(np.uint8)
    return normalized_image
