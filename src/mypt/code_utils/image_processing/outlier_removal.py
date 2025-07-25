"""
This script contains functions to remove outliers from images
"""

import numpy as np

from functools import partial
from scipy.ndimage import uniform_filter, generic_filter, median_filter 


## LOCAL FILTERS

def local_zscore_filter(image: np.ndarray, ksize: int = 3, threshold: float = 3.0) -> np.ndarray:
    """
    Removes local outliers using the z-score within a local window.
    Pixels with a z-score above the threshold are replaced with the local mean.
    Args:
        image: Input image as a numpy array (2D or 3D).
        ksize: Size of the local window (must be odd).
        threshold: Z-score threshold for outlier detection.
    Returns:
        Filtered image as a numpy array.
    """
    if image.ndim == 2:
        # compute the mean 
        mean = uniform_filter(image.astype(float), ksize)

        # variance = (square - mean) ^ 2 
        # std = sqrt(variance)
        sqr_image = image.astype(float) ** 2
        variance = uniform_filter((sqr_image - mean) ** 2, ksize)
        std = np.sqrt(variance)

        # convert the image to z-score
        z = np.zeros_like(image, dtype=float)
        std_nonzero = std > 0
        z[std_nonzero] = (image[std_nonzero] - mean[std_nonzero]) / std[std_nonzero]
        outlier_mask = np.abs(z) > threshold
        filtered = image.copy()
        filtered[outlier_mask] = mean[outlier_mask].astype(image.dtype)
        return filtered
    else:
        # Apply channel-wise for color images
        filtered = np.stack([
            local_zscore_filter(image[..., c], ksize, threshold) for c in range(image.shape[-1])
        ], axis=-1)
        return filtered

def _mad_filter_func(values, threshold):
    center = values[len(values) // 2]
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return center
    score = np.abs(center - median) / mad
    return median if score > threshold else center

def local_mad_filter(image: np.ndarray, ksize: int = 3, threshold: float = 3.0) -> np.ndarray:
    """
    Removes local outliers using the Median Absolute Deviation (MAD) within a local window.
    Pixels with a MAD score above the threshold are replaced with the local median.
    Args:
        image: Input image as a numpy array (2D or 3D).
        ksize: Size of the local window (must be odd).
        threshold: MAD threshold for outlier detection.
    Returns:
        Filtered image as a numpy array.
    """

    if image.ndim == 2:
        return generic_filter(image, partial(_mad_filter_func, threshold=threshold), size=ksize, mode='reflect')
    else:
        filtered = np.stack([
            local_mad_filter(image[..., c], ksize, threshold) for c in range(image.shape[-1])
        ], axis=-1)
        return filtered

def local_median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Applies a median filter to the image to remove local outliers (salt-and-pepper noise).
    Args:
        image: Input image as a numpy array (2D or 3D).
        ksize: Size of the local window (must be odd).
    Returns:
        Filtered image as a numpy array.
    """
    if image.ndim == 2:
        return median_filter(image, size=ksize, mode='reflect')
    else:
        filtered = np.stack([
            median_filter(image[..., c], size=ksize, mode='reflect') for c in range(image.shape[-1])
        ], axis=-1)
        return filtered



## GLOBAL FILTERS
def percentile_clip(image: np.ndarray, lower_percentile: float = 1.0, upper_percentile: float = 99.0) -> np.ndarray:
    """
    Clips an image's pixel values to a specified percentile range.
    This is a global operation that helps remove extreme outlier pixels.
    Args:
        image: Input image as a numpy array.
        lower_percentile: The lower percentile (0-100) to clip values to.
        upper_percentile: The upper percentile (0-100) to clip values to.
    Returns:
        The clipped image with the same data type as the input.
    """
    lower_val = np.percentile(image, lower_percentile)
    upper_val = np.percentile(image, upper_percentile)
    return np.clip(image, a_min=lower_val, a_max=upper_val)




