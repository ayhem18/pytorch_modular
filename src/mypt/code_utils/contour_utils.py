"""
A script for contour utility functions
"""

import numpy as np

from typing import Iterable, List, Tuple, Union
from .bbox_utils import extract_contour_bounding_box

CONTOUR = Iterable[Iterable[int]]

NUM = Union[float, int]



def _squeeze_iterable(x: Iterable) -> List:
    while len(x) == 1:
        x = x[0]

    if isinstance(x, np.ndarray):
        return x.tolist()
    
    if isinstance(x, Tuple):
        return list(x)
    
    return x


def cv2_contour_format(contour: CONTOUR) -> CONTOUR:
    # the first step is to squeeze the iterable
    squeezed_contour = _squeeze_iterable(contour) 

    # check if the length of the contour is 2
    if len(squeezed_contour) == 2:
        if isinstance(squeezed_contour[0], np.num) and isinstance(squeezed_contour[0], np.num):
            # the the contour originally contained only 1 point (a point is a tuple of 2 numbers)
            temp_array = np.array(squeezed_contour)
            res = temp_array[np.newaxis, np.newaxis, :]
            if res.shape != (1, 1, 2):
                raise ValueError(f"The contour has not the correct format. Found: {res}. The expected format is: (1, 1, 2)")
            return res
    
    # at this point, the contour must be a list of iteratebles where each iteratble contains 2 numbers
    further_squeezed_contour = [_squeeze_iterable(c) for c in squeezed_contour] 

    if not all([len(c) == 2 for c in further_squeezed_contour]):
        raise ValueError(f"The contour has not the correct format. Found: {further_squeezed_contour}. The expected format is: a list of iteratebles where each iteratble contains 2 numbers")
    
    # at this point, the contour is a list of iteratebles where each iteratble contains 2 numbers
    # we need to convert it to a numpy array
    res = np.array(further_squeezed_contour)
    res = res[:, np.newaxis, :]

    if res.shape != (len(squeezed_contour), 1, 2):
        raise ValueError(f"The contour has not the correct format. Found: {res}. The expected format is: (len(squeezed_contour), 1, 2)")
    
    return res
        


def interpolate_between_two_points(p1: Tuple[int, int], p2: Tuple[int, int], num_points: int) -> CONTOUR:
    y1, x1 = p1
    y2, x2 = p2
    slope1 = (y2 - y1)  / (num_points + 1)
    slope2 = (x2 - x1)  / (num_points + 1)
    res =  [(int(y1 + slope1 * (i + 1)), int(x1 + slope2 * (i + 1))) for i in range(num_points)]
    return res


def stretch_contour(contour: CONTOUR, coefficient: NUM, y_x: bool = True, x_axis: bool = True):
    if x_axis:
        return stretch_contour_x_axis(contour, coefficient, y_x)
    else:
        return stretch_contour_y_axis(contour, coefficient, y_x)


def stretch_contour_y_axis(contour: CONTOUR, coefficient: NUM, y_x: bool = True):
    # the idea is as follows:
    # 1. rotate the contour over the y = x line
    # 2. stretch the contour with the stretch_contour_x_axis function
    # 3. rotate the contour back
    rotated_contour = [(c2, c1) for c1, c2 in contour]
    stretched_contour = stretch_contour_x_axis(rotated_contour, coefficient, y_x)
    res =  [(c2, c1) for c1, c2 in stretched_contour]
    return res


def stretch_contour_x_axis(contour: CONTOUR, coefficient: NUM, y_x: bool = True):

    if coefficient < 1:
        raise ValueError(f"The coefficient argument must be larger than 1. Found: {coefficient}")

    if coefficient == 1:
        return contour

    # normalize the contour
    if not y_x:
        nc = [(y, x) for x, y in contour]
    else:
        nc = [(y, x) for y, x in contour]

    sorted_contour = sorted(nc, key=lambda p: p[1]) 
    
    # find the middle point
    middle_point = sorted_contour[len(sorted_contour) // 2]
    
    mpx = middle_point[1]

    new_contour = [(y, int(mpx + (x - mpx) * coefficient)) for y, x in sorted_contour]

    # if the minimum value in the new_contour is less than 0, then we need to shift the contour to the right: to avoid negative values
    # compute the minimum value in the new_contour
    min_x_new_coordinate = min(new_contour, key=lambda p: p[1])[1]
    min_y_new_coordinate = min(new_contour, key=lambda p: p[0])[0]

    # if the minimum value is less than 0, then we need to shift the contour to the right
    if min_x_new_coordinate < 0:
        new_contour = [(y, x + abs(min_x_new_coordinate)) for y, x in new_contour]

    # if the minimum value is less than 0, then we need to shift the contour to the right
    if min_y_new_coordinate < 0:
        new_contour = [(y + abs(min_y_new_coordinate), x) for y, x in new_contour]

    assert all([x >= 0 for _, x in new_contour]), f"The new contour contains negative values. Found: {new_contour}"
    assert all([y >= 0 for y, _ in new_contour]), f"The new contour contains negative values. Found: {new_contour}"
    
    # iterate and interpolate
    num_points = int(max(1, coefficient // 2))  
    interpolation_points = [interpolate_between_two_points(new_contour[i], new_contour[i + 1], num_points) for i in range(len(new_contour) - 1)]

    # flatten the list 
    for ip in interpolation_points:
        new_contour.extend(ip)

    # the final step is to sort by x_axis
    new_contour = sorted(new_contour, key=lambda p: p[1])

    x21, _, x22, _ = extract_contour_bounding_box(new_contour)

    x11, _ ,x12, _ = extract_contour_bounding_box(contour)

    assert np.isclose((x22 - x21) / (x12 - x11), coefficient, atol=0.1), f"The stretching did not work as expected found: {(x22 - x21) / (x12 - x11)} expected: {coefficient}"

    return new_contour

