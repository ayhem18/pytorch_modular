"""
This module contains functions to prepare images in different formats (RGB, BGR, grayscale, etc.)
"""

import numpy as np
import cv2


def bgr_to_xyz(bgr_image: np.ndarray) -> np.ndarray:
    """
    Converts a BGR image to the CIE XYZ color space, preserving the full dynamic range.
    This is the first step in converting from a device-dependent color space (BGR)
    to a device-independent one (CIELAB).

    The conversion uses the standard sRGB to XYZ transformation matrix.

    Reference:
        http://www.brucelindbloom.com/index.html?Eqn_RGB_to_XYZ.html

    Args:
        bgr_image: A BGR image as a NumPy array (can be uint8, uint16, or float).

    Returns:
        The image in XYZ color space as a floating-point NumPy array.
    """
    # Normalize the image to a [0, 1] float range based on its data type
    if bgr_image.dtype == 'uint8':
        bgr_float = bgr_image.astype(float) / 255.0
    elif bgr_image.dtype == 'uint16':
        bgr_float = bgr_image.astype(float) / 65535.0
    else:
        bgr_float = bgr_image.astype(float) # Assume it's already [0, 1] if float

    # Standard BGR to XYZ conversion matrix
    # This matrix is derived from the standard sRGB to XYZ conversion formulas.
    bgr_to_xyz_matrix = np.array([
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ]).T # Transpose because we will right-multiply

    # Apply the transformation
    xyz_image = np.dot(bgr_float, bgr_to_xyz_matrix)
    return xyz_image


def xyz_to_lab(xyz_image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts an XYZ image to the CIE L*a*b* color space.
    This space is designed to be perceptually uniform, meaning a change of the
    same amount in a color value should produce a change of about the same
    visual importance.

    Reference:
        http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_Lab.html

    Args:
        xyz_image: An image in XYZ color space as a floating-point NumPy array.

    Returns:
        A tuple containing the L*, a*, and b* channels as separate float arrays.
    """
    # Define the D65 reference white point
    ref_white = np.array([0.95047, 1.00000, 1.08883])
    
    # Normalize by the reference white
    xyz_norm = xyz_image / ref_white

    # Define the non-linear mapping function `f(t)`
    epsilon = (6/29)**3
    f = np.where(xyz_norm > epsilon,
                 np.cbrt(xyz_norm),
                 (xyz_norm / (3 * (6/29)**2)) + (4/29))

    # Calculate L*, a*, b*
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    
    L_channel = (116 * fy) - 16
    a_channel = 500 * (fx - fy)
    b_channel = 200 * (fy - fz)
    
    return L_channel, a_channel, b_channel


def bgr_to_lab_full_range(bgr_image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts a BGR image (uint8 or uint16) to CIE L*a*b* space without losing
    the dynamic range in the L* channel. This is achieved by implementing the
    full BGR -> XYZ -> LAB pipeline with floating-point precision.

    This function orchestrates the calls to `bgr_to_xyz` and `xyz_to_lab`.

    Args:
        bgr_image: A BGR image as a NumPy array.

    Returns:
        A tuple containing the L*, a*, and b* channels as separate float arrays.
        The L* channel will be in the range [0, 100], while a* and b* will be
        in a range approximately [-128, 127].
    """
    xyz_image = bgr_to_xyz(bgr_image)
    L_channel, a_channel, b_channel = xyz_to_lab(xyz_image)
    return L_channel, a_channel, b_channel


def lab_to_xyz(L_channel: np.ndarray, a_channel: np.ndarray, b_channel: np.ndarray) -> np.ndarray:
    """
    Converts L*a*b* channels back to the CIE XYZ color space.

    Reference:
        http://www.brucelindbloom.com/index.html?Eqn_Lab_to_XYZ.html

    Args:
        L_channel: The L* channel as a float array.
        a_channel: The a* channel as a float array.
        b_channel: The b* channel as a float array.

    Returns:
        The image in XYZ color space as a floating-point NumPy array.
    """
    fy = (L_channel + 16) / 116
    fx = a_channel / 500 + fy
    fz = fy - b_channel / 200

    f_xyz = np.stack([fx, fy, fz], axis=-1)
    
    epsilon = 6/29
    
    xyz_norm = np.where(f_xyz > epsilon,
                        f_xyz**3,
                        (f_xyz - 16/116) * 3 * (epsilon**2))

    ref_white = np.array([0.95047, 1.00000, 1.08883])
    xyz_image = xyz_norm * ref_white
    return xyz_image


def xyz_to_bgr(xyz_image: np.ndarray) -> np.ndarray:
    """
    Converts an XYZ image back to the BGR color space.

    Reference:
        http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_RGB.html

    Args:
        xyz_image: The image in XYZ color space as a floating-point NumPy array.

    Returns:
        The BGR image as a float array in the [0, 1] range.
    """
    xyz_to_bgr_matrix = np.linalg.inv(np.array([
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ]).T)

    bgr_float = np.dot(xyz_image, xyz_to_bgr_matrix)
    # Clip to [0, 1] range to handle out-of-gamut colors
    return np.clip(bgr_float, 0, 1)


def lab_to_bgr_full_range(L_channel: np.ndarray, a_channel: np.ndarray, b_channel: np.ndarray) -> np.ndarray:
    """
    Converts L*a*b* channels back to a BGR image using a full-range,
    floating-point precision pipeline.

    Args:
        L_channel: The L* channel, typically [0, 100].
        a_channel: The a* channel, typically [-128, 127].
        b_channel: The b* channel, typically [-128, 127].

    Returns:
        The BGR image as a float array in the [0, 1] range.
    """
    xyz_image = lab_to_xyz(L_channel, a_channel, b_channel)
    bgr_image = xyz_to_bgr(xyz_image)
    return bgr_image
