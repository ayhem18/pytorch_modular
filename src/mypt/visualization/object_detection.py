"""
This script contains functionalities to visualize images with bounding boxes
"""

import cv2 as cv
import numpy as np

from typing import List, Optional, Tuple, Union
from mypt.code_utils.bbox_utils import convert_bbox_annotation, _verify_pascal_voc_format, OBJ_DETECT_ANN_TYPE, PASCAL_VOC


def draw_single_bounding_box(image: np.ndarray, 
                            y_max: Union[float, int], 
                            y_min: Union[float, int], 
                            x_max: Union[float, int], 
                            x_min: Union[float, int],
                            color: Optional[Union[str, Tuple[int]]] = None,
                            thickness: int = 3) -> np.ndarray:
    if color is None:
        color = (0, 255, 0)
    
    # create a bounbding box in the pascal_voc format
    ann = [x_min, y_min, x_max, y_max]
    # make sure the dimensions make sense
    x_min, y_min, x_max, y_max = _verify_pascal_voc_format(ann, img_shape=image.shape, normalize=False)

    # copy the image as we will be drawing on top of it
    img_copy = image.copy()
    # draw a rectangle
    cv.rectangle(img_copy, (x_min, y_min), (x_max, y_max), color, thickness)

    return img_copy    


def draw_multi_bbox(image: np.ndarray, bboxes: List[OBJ_DETECT_ANN_TYPE], bbox_format: str = PASCAL_VOC) -> None:
    if len(image.shape) != 3:
        raise ValueError(f"Make sure to pass a 3 dimensional image. Otherwise, the results are unpredictable")
    
    # make a copy
    img_copy = image.copy()

    # convert the bounding boxes to the pascal_voc format
    bboxes = [convert_bbox_annotation(b, 
                                      img_shape=image.shape,
                                      current_format=bbox_format, 
                                      target_format=PASCAL_VOC) for b in bboxes]

    for b in bboxes:
        x_min, y_min, x_max, y_max = b
        cv.rectangle(img_copy, (x_min, y_min), (x_max, y_max), color=(0, 255,0), thickness=3)

    return img_copy
    
