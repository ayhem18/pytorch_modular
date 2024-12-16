## testing the conversion of obj detection annotation formats

import random
import numpy as np

from typing import Tuple
from tqdm import tqdm

from mypt.code_utilities import bbox_utilities as au

################################## utility functions ################################## 
def _random_yolo_ann(img_shape=None):
    x_c = 0.05 + round(0.8 * random.random(), 4)
    y_c = 0.05 + round(0.8 * random.random(), 4)

    min_dim_x = min(1 - x_c, x_c)
    min_dim_y = min(1 - y_c, y_c)

    w = (min_dim_x * random.random() - 0.02)
    h = (min_dim_y * random.random() - 0.02)

    # this is okay since x_c and y_c range between [0.05 and 0.85]
    h = round(max(h, 0.01) * 2, 4)
    w = round(max(w, 0.01) * 2, 4)

    return [x_c, y_c, w, h]

def _random_coco_ann(img_shape: Tuple[int, int]):
    y, x = img_shape
    min_x, min_y = random.randint(0, int(0.8 * x)), random.randint(0, int(0.8 * y))
    w, h = int((x - min_x) * (random.random() + 0.0001)), int((y - min_y) * (random.random() + 0.0001))
    # make sure the width and the height are at least 1
    w, h= max(w, 1), max(h, 1)
    return [min_x, min_y, w, h]

def _random_pascal_voc_ann(img_shape: Tuple[int, int]):
    y, x = img_shape
    min_x = random.randint(0, int(x * 0.75))
    min_y = random.randint(0, int(y * 0.75))

    max_x = random.randint(min_x + 2, x)
    max_y = random.randint(min_y + 2, y)

    return [min_x, min_y, max_x, max_y]

def _random_albumentation_ann(img_shape=None):
    min_x = 0.01 + 0.8 * random.random()
    min_y = 0.01 + 0.8 * random.random()
    max_x = min_x + random.random() * (1 - min_x - 0.005)
    max_y = min_y + random.random() * (1 - min_y - 0.005)

    max_x = max(max_x, min_x + 2 / img_shape[1])
    max_y = max(max_y, min_y + 2 / img_shape[0])

    return [min_x, min_y, max_x, max_y]


__format_random_generation = {au.COCO: _random_coco_ann, 
                              au.ALBUMENTATION: _random_albumentation_ann, 
                              au.PASCAL_VOC: _random_pascal_voc_ann, 
                              au.YOLO: _random_yolo_ann}


################################# test conversion #################################
def _test_conversion_single_format(format: str, num_tests:int = 5 * 10 ** 4):
    if format not in au.OBJ_DETECT_ANN_FORMATS:
        raise NotImplementedError(f"Currently supporting only the following formats: {au.OBJ_DETECT_ANN_FORMATS}")

    other_formats = [fr for fr in au.OBJ_DETECT_ANN_FORMATS if fr != format]

    for _ in tqdm(range(num_tests), desc=f'passing tests with the {format} format'):
        img_shape = (random.randint(50, 400), random.randint(50, 400))
        random_ann = __format_random_generation[format](img_shape=img_shape)

        # verify the format
        au.verify_object_detection_ann_format(annotation=random_ann, 
                                              img_shape=img_shape, 
                                              current_format=format)

        for of in other_formats:
            # convert 
            of_ann = au.convert_annotations(annotation=random_ann, current_format=format, target_format=of, img_shape=img_shape)

            au.verify_object_detection_ann_format(annotation=of_ann, img_shape=img_shape, current_format=of)

            # convert back
            or_ann_converted = au.convert_annotations(annotation=of_ann, current_format=of, target_format=format, img_shape=img_shape)


            if format in [au.COCO, au.PASCAL_VOC]:
                # since those de
                all([abs(v1 - v2) <= 1 for v1, v2 in zip(random_ann, or_ann_converted)]), "Make sure the conversion is correct"
            else:
                max_dim = max(img_shape)
                all([abs(v1 - v2) <= 2 / max_dim for v1, v2 in zip(random_ann, or_ann_converted)]), "Make sure the conversion is correct"


def _test_conversion(num_tests:int = 5 * 10 ** 4):
    # set the seed for reproducibility
    from mypt.code_utilities import pytorch_utilities as pu
    pu.seed_everything(seed=0)
    for f in au.OBJ_DETECT_ANN_FORMATS:
        _test_conversion_single_format(format=f, num_tests=num_tests)


if __name__ == '__main__':
    _test_conversion()
