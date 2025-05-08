"""
This script contains some functionalities helpful for the segmentation task
"""

import numpy as np

from PIL import Image
from pathlib import Path
from collections import deque
from typing import List, Optional, Tuple, Union, Dict

from .bbox_utils import OBJ_DETECT_ANN_TYPE


def process_segmentation_mask(segmentation_mask: Union[np.ndarray, str, Path],
                              pixelsMapping: Dict = None,
                              ) -> Optional[Tuple[List[Tuple[int]], List[int]]]:

    if isinstance(segmentation_mask, (str, Path)):
        segmentation_mask = np.asarray(Image.open(segmentation_mask))

    if segmentation_mask.ndim not in [2, 3]:
        raise ValueError(f"The segmentation is expected to be either 2 or 3 dimensional.")

    if segmentation_mask.ndim == 3 and pixelsMapping is None:
        raise TypeError("if the mask is 3 dimensional, the pixelMapping is required")


    if pixelsMapping is not None and 0 in pixelsMapping.values():
        raise ValueError("the value 0 is reserved for background !!. Make sure it is not used as a value in the mapping configuration !!!")


    # make sure the segmentation mask is not degenerate: has the same value across the entire mask
    # first flatten it to 2d
    if segmentation_mask.ndim == 3:
        _sm2d = np.sum(segmentation_mask, axis=-1)
    else:
        _sm2d = segmentation_mask.copy()

    if np.all(_sm2d == _sm2d[0, 0]):
        raise ValueError(f"The mask is degenerate. There is only one pixel value across the entire mask")

    # convert the segmentation_mask to 2D
    sm2d = np.zeros(shape=(segmentation_mask.shape[:2]))

    structure_pixels = set()

    for i in range(segmentation_mask.shape[0]):
        for j in range(segmentation_mask.shape[1]):
            if segmentation_mask.ndim == 3:
                sm2d[i][j] = pixelsMapping.get(tuple(segmentation_mask[i,j,:].tolist()), 0)
            else: 
                sm2d[i][j] = segmentation_mask[i][j] if pixelsMapping is None else pixelsMapping.get(sm2d[i][j], 0)

            if sm2d[i][j] != 0:
                structure_pixels.add((i, j))

    assert len(structure_pixels) != 0, "the extract of structure pixels gone wrong..."

    # components to save the structures    
    components = []
    comp_classes = []

    visited_pixels = set()
    
    for iy, ix in structure_pixels:
        if (iy, ix) in visited_pixels:
            continue
        
        current_component = set()

        # set it as visited 
        visited_pixels.add((iy, ix))

        # the pixel was not visited before, apply bsf
        queue = deque([(iy, ix)])

        while len(queue) != 0:
            y, x = queue.pop()

            current_component.add((y, x))
            # possible neighbors (mind the boundaries)
            possible_neighbors = [(y + i, x + j) for i in range(-1, 2) for j in range(-1, 2) if (0 <= (y + i) < sm2d.shape[0] and 0 <= (x + j) < sm2d.shape[1])]

            # filter the neighbors: must be non-visited and of the same color
            next = [n for n in possible_neighbors if (n not in visited_pixels and sm2d[n[0]][n[1]] == sm2d[y][x])]

            # set each of the pixels as visited and add them to the queue
            for n in next:
                visited_pixels.add(n)
                queue.append(n)


        # the component is fully traversed at this point: add it to the list of components
        components.append(list(current_component))
        # extract the segmentation_mask associated with the component 
        comp_classes.append(sm2d[iy][ix])

    return components, comp_classes


