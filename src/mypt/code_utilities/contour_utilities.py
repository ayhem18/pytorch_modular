"""
A script for contour utility functions
"""


from typing import Iterable, Tuple, Union
from itertools import chain
from .bbox_utilities import extract_contour_bounding_box

CONTOUR = Iterable[Iterable[int]]

NUM = Union[float, int]


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
    return [(c2, c1) for c1, c2 in stretched_contour]


def stretch_contour_x_axis(contour: CONTOUR, coefficient: NUM, y_x: bool = True):

    if coefficient <= 1:
        raise ValueError(f"The coefficient argument must be larger than 1. Found: {coefficient}")

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

    # if the minimum value is less than 0, then we need to shift the contour to the right
    if min_x_new_coordinate < 0:
        new_contour = [(y, x + abs(min_x_new_coordinate)) for y, x in new_contour]

    assert all([x >= 0 for _, x in new_contour]), f"The new contour contains negative values. Found: {new_contour}"

    # iterate and interpolate
    num_points = int(max(1, coefficient // 2))  
    interpolation_points = [interpolate_between_two_points(new_contour[i], new_contour[i + 1], num_points) for i in range(len(new_contour) - 1)]

    # flatten the list 
    for ip in interpolation_points:
        new_contour.extend(ip)

    # the final step is to sort by x_axis
    return sorted(new_contour, key=lambda p: p[1])

