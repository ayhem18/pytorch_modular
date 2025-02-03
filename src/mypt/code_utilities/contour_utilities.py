"""
A script for contour utility functions
"""


from typing import Iterable, Tuple
from itertools import chain
from .bbox_utilities import extract_contour_bounding_box
from ..shortcuts import NUM, CONTOUR


def interpolate_between_two_points(p1: Tuple[int, int], p2: Tuple[int, int], num_points: int) -> CONTOUR:
    y1, x1 = p1
    y2, x2 = p2
    slope1 = (y1 - y2 )  / (num_points + 1)
    slope2 = (x1 - x2)  / (num_points + 1)
    return [(int(y1 + slope1 * (i + 1)), int(x1 + slope2 * (i + 1))) for i in range(num_points)]


def stetch_contour(contour: CONTOUR, coefficient: NUM, y_x: bool = True, x_axis=True):

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

    # # extract the min_x and max_x points
    # min_x, max_x = sorted_contour[0][1], sorted_contour[-1][1]

    # org_left_distance = max_x - middle_point
    # org_right_distance = middle_point - min_x

    # new_left_distance = org_left_distance * coefficient
    # new_right_distance = org_right_distance * coefficient

    # # compute the new edges
    # new_min_x, new_max_x = int(middle_point - new_right_distance), int(middle_point + new_left_distance)

    new_contour = [(y, int(middle_point + (x - middle_point) * coefficient)) if x <= middle_point else 
                    (y, int(middle_point + (middle_point - x) * coefficient)) for y, x in sorted_contour]
    
    
    # iterate and interpolate
    num_points = int(max(1, coefficient // 2))  
    interpolation_points = [interpolate_between_two_points(new_contour[i], new_contour[i + 1], num_points) for i in range(len(new_contour) - 1)]

    # flatten the list 
    for ip in interpolation_points:
        new_contour.extend(ip)

    # the final step is to sort by x_axis
    return sorted(new_contour, key=lambda p: p[1])

