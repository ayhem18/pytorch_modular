"""
This script contains the code to prepare the data for training on the object localization task
"""

import os
import albumentations as A
import numpy as np

from typing import Union, List, Tuple
from pathlib import Path
from functools import partial

from mypt.code_utilities import annotation_utilites as au
from mypt.data.datasets.obj_detect_loc.object_localization import ObjectLocalizationDs

from src.data_preparation import extract_annotation_from_xml, get_annotation_file_path

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def img_2_annotation(file_path: Union[str, Path], ann_folder: Union[str, Path]):
    ann_file = os.path.join(ann_folder, get_annotation_file_path(file_path))

    if not os.path.exists(ann_file):
        # this means that the sample is a background sample
        return [
            ['backgroun'], 
            [au.DEFAULT_BBOX_BY_FORMAT[au.PASCAL_VOC]]
                ]    

    cls, bbox, img_shape = extract_annotation_from_xml(ann_file)
    if len(bbox) != 1:
        bbox = [bbox]

    if not isinstance(cls, (List, Tuple)):
        cls = [cls]

    return cls, bbox, img_shape

if __name__ == '__main__':
    train_data = os.path.join(SCRIPT_DIR, 'data', 'plants_ds_train')
    ann_folder = os.path.join(SCRIPT_DIR, 'data', 'plants_ds_annotations')
    output_shape = (227, 227)

    samples = [os.path.join(train_data, f) for f in os.listdir(train_data)] 

    ds = ObjectLocalizationDs(root_dir=train_data, 
                              img_augs=[A.RandomSizedBBoxSafeCrop(*output_shape, p=1), A.ColorJitter(p=1)],
                              
                              output_shape=output_shape,
                              compact=True,

                              image2annotation=partial(img_2_annotation, ann_folder=ann_folder),

                              target_format='albumentations',
                              current_format='pascal_voc',
                              background_label='background', 
                              )
    
    import matplotlib.pyplot as plt
    import cv2 as cv    
    import random

    n = len(ds)

    indices = random.sample(range(n), 10)
    
    for i in indices:
        x, y = ds[i]
        
        obj_indicator, bbox, c = y[0].item(), y[1:5].tolist(), y[5:].tolist()

        x = (np.moveaxis(x.numpy(), 0,-1) * 255).astype(np.uint8)
        x = np.ascontiguousarray(x)

        pascal_voc_bbox =au.convert_bbox_annotation(annotation=bbox, current_format=au.ALBUMENTATIONS, target_format=au.PASCAL_VOC, img_shape=x.shape[:2])

        if obj_indicator:
            xmin, ymin, xmax, ymax = pascal_voc_bbox
            cv.rectangle(x, (xmin, ymin), (xmax, ymax), (0, 255,0), 1)    

        plt.title(f'class : {c}')
        plt.imshow(x)
        plt.show()

        # b_img = convert_img_2_background(img, bbox)
        # plt.imshow(b_img)
        # plt.show()


