"""
This script contains the code to prepare the data for training on the object localization task
"""

import os
import xml.etree.ElementTree as ET

from typing import Union, Tuple, List
from pathlib import Path

from mypt.code_utilities import directories_and_files as dirf


def extract_annotation_from_xml(xml_file_path: Union[str, Path]) -> Tuple[Tuple[int, int], List[int]]:
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    size = {
        'width': int(root.find('size').find('width').text),
        'height': int(root.find('size').find('height').text),
        'depth': int(root.find('size').find('depth').text)
    }

    # find the 
    x_min = int(root.find('object').find('bndbox').find('xmin').text)
    x_max = int(root.find('object').find('bndbox').find('xmax').text)
    y_min = int(root.find('object').find('bndbox').find('ymin').text)
    y_max = int(root.find('object').find('bndbox').find('ymax').text)

    # extract the classs, bounding box and image shape
    return root.find('object').find('name').text.lower(), [x_min, y_min, x_max, y_max], (size['width'], size['height']) 
