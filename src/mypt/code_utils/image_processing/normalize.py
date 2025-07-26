"""
This script contains functions for normalizing images, including converting
from high bit-depth (e.g., 16-bit) to 8-bit for visualization or processing.
"""
import numpy as np
import cv2
from . import format as fmt

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
    res = cv2.normalize(image, None, target_min, target_max, cv2.NORM_MINMAX, dtype=cv2.CV_8U) # type: ignore
    return res.astype(np.uint8) 

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

def _gamma_less_than_one_normalize(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    # Normalize to [0, 1] range first
    # if the vlaue of gamme is too small, 1 / gamme might get too large 
    # and the image might blow up
    img_min, img_max = np.min(image), np.max(image)
    if img_max == img_min:
        raise ValueError("The image has no range of values. There might be an issue with the image.")

    img_norm = (image - img_min) / (img_max - img_min)
    
    # Apply gamma correction
    gamma_corrected = np.power(img_norm, 1.0 / gamma)
    
    # Scale to 8-bit
    normalized_image = linear_normalize(gamma_corrected)
    return normalized_image

def _gamma_greater_than_one_normalize(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    # if gamme is greater than 1, then 1 / gamma will be less than 1. we can just apply the transformation directly 
    # without min-max scaling first
    gammed_corrected = np.power(image, 1.0 / gamma)
    normalized_image = linear_normalize(gammed_corrected)
    return normalized_image


def gamma_normalize(image: np.ndarray, gamma: float = 0.5) -> np.ndarray:
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

    return _gamma_less_than_one_normalize(image, gamma) if gamma < 1.0 else _gamma_greater_than_one_normalize(image, gamma)        


def _rank_order_normalize_2d(img: np.ndarray, count_threshold: int) -> np.ndarray:
    """Applies rank-order normalization to a single channel (grayscale) image."""
    # Get unique pixel values and their counts. np.unique returns sorted unique values.
    unique_vals, counts = np.unique(img, return_counts=True)

    # Filter out infrequent pixel values to create the new "palette".
    palette = unique_vals[counts >= count_threshold]

    if len(palette) == 0:
        # If no values meet the threshold, return a black image.
        return np.zeros_like(img, dtype=np.uint8)

    # Create a lookup table (LUT) that maps an index to a new 8-bit value.
    num_palette_values = len(palette)
    if num_palette_values > 1:
        lut = ((np.arange(num_palette_values) / (num_palette_values - 1)) * 255).astype(np.uint8)
    else:
        lut = np.array([128], dtype=np.uint8)

    # Map original pixel values to their rank in the new palette.
    indices = np.searchsorted(palette, img)
    indices = np.clip(indices, 0, num_palette_values - 1)

    # Apply the lookup table to create the new image.
    res = lut[indices]

    return res



def rank_order_normalize(image: np.ndarray, count_threshold: int = 1) -> np.ndarray:
    """
    Normalizes an image to 8-bit by mapping pixel values based on their rank order,
    after filtering out infrequent values. This enhances contrast for frequent values.
    This is a form of non-linear histogram equalization. For color images, this
    is applied to the L-channel of the LAB color space to avoid color distortion.

    The method works as follows:
    1.  Compute the frequency of each unique pixel value in the image.
    2.  Filter out pixel values that occur fewer times than `count_threshold`.
        The remaining sorted values form a new "palette".
    3.  Let the filtered palette have N values. Each value in the original image is
        mapped to a new 8-bit value based on its rank `i` in the palette,
        calculated as `(i / (N-1)) * 255`.

    Args:
        image: Input image as a numpy array (grayscale or BGR color).
        count_threshold: The minimum number of times a pixel value must occur
                         to be included in the normalization mapping. Values below
                         this threshold will be mapped to the next lowest rank.

    Returns:
        Normalized 8-bit image (grayscale or color, matching input).
    """

    if not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3)):
        raise ValueError("Input image must be grayscale (2D) or 3-channel BGR color (3D).")

    if image.ndim == 2:
        # Grayscale image
        return _rank_order_normalize_2d(image, count_threshold)
    
    # --- Color Image Workflow ---
    
    # 1. Convert BGR to LAB with full dynamic range
    L_channel, a_channel, b_channel = fmt.bgr_to_lab_full_range(image)

    # 2. Apply rank-order normalization to the L-channel.
    # The L-channel is a float in range [0, 100], but the logic works the same.
    # The output will be a uint8 image in range [0, 255].
    normalized_L = _rank_order_normalize_2d(L_channel, count_threshold)

    # 3. Convert the normalized L channel back to a [0, 100] float range
    #    to be compatible with the a* and b* channels for the inverse transform.
    normalized_L_float = normalized_L.astype(float) * (100.0 / 255.0)

    # 4. Convert back to BGR using the full-range pipeline
    # The output is a float BGR image in the [0, 1] range.
    bgr_float = fmt.lab_to_bgr_full_range(normalized_L_float, a_channel, b_channel)

    # 5. Convert the final float image to a standard uint8 BGR image
    return (bgr_float * 255).astype(np.uint8)


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
    # img_min, img_max = np.min(image), np.max(image)
    # if img_max == img_min:
    #     raise ValueError("The image has no range of values. There might be an issue with the image.")

    # img_norm = (image - img_min) / (img_max - img_min)

    # Apply log transform, add epsilon to avoid log(0)
    log_transformed = np.log1p(image) # log1p is log(1+x)
    
    # Scale to 8-bit
    normalized_image = linear_normalize(log_transformed)
    return normalized_image


# def sigmoid_normalize(image: np.ndarray, gain: float = 10.0, cutoff: float = 0.5) -> np.ndarray:
#     """
#     Applies sigmoidal contrast adjustment to an image and scales to 8-bit.
#     Increases contrast in the mid-tones while compressing shadows and highlights.
#     Args:
#         image: Input image as a numpy array.
#         gain: Controls the steepness of the curve (contrast). Higher is more contrasty.
#         cutoff: The center of the transition (mid-point of the dynamic range, 0-1).
#     Returns:
#         Sigmoid-adjusted and normalized 8-bit image.
#     """
#     # Normalize to [0, 1] range first
#     img_min, img_max = np.min(image), np.max(image)

#     if img_max == img_min:
#         raise ValueError("The image has no range of values. There might be an issue with the image.")
    
#     img_norm = (image - img_min) / (img_max - img_min)
    
#     # Apply sigmoid function
#     sigmoid_image = 1 / (1 + np.exp(gain * (cutoff - img_norm)))
    
#     # Scale to 8-bit
#     normalized_image = (sigmoid_image * 255).astype(np.uint8)
#     return normalized_image
