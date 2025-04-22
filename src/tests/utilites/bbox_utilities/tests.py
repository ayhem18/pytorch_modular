"""
This script contains the code to test bbox_utility functions.
"""
import os
import numpy as np
from mypt.code_utils.bbox_utils import bounding_boxes_intersect, PASCAL_VOC

def test_positive_intersection():
    # using the bounding boxes to generate the image samples
    img = np.zeros(shape=(500, 500, 3))
    box1 = [100, 100, 250, 200]
    box2 = [200, 50, 300, 150]
    assert bounding_boxes_intersect(bbox1=box1, bbox2=box2, img_shape=img.shape, bbox1_format=PASCAL_VOC, bbox2_format=PASCAL_VOC)


    img = np.zeros(shape=(500, 500, 3))
    box1 = [100, 100, 250, 200]
    box2 = [150, 50, 300, 350]
    assert bounding_boxes_intersect(bbox1=box1, bbox2=box2, img_shape=img.shape, bbox1_format=PASCAL_VOC, bbox2_format=PASCAL_VOC)


    img = np.zeros(shape=(500, 500, 3))
    box1 = [0, 0, 400, 400]
    box2 = [100, 100, 200, 200]
    assert bounding_boxes_intersect(bbox1=box1, bbox2=box2, img_shape=img.shape, bbox1_format=PASCAL_VOC, bbox2_format=PASCAL_VOC)

    img = np.zeros(shape=(500, 500, 3))
    box1 = [0, 0, 300, 300]
    box2 = [100, 100, 200, 400]
    assert bounding_boxes_intersect(bbox1=box1, bbox2=box2, img_shape=img.shape, bbox1_format=PASCAL_VOC, bbox2_format=PASCAL_VOC)

def test_negative_intersection():
    img = np.zeros(shape=(500, 500, 3))
    box1 = [100, 100, 250, 200]
    box2 = [150, 0, 200, 50]
    assert not bounding_boxes_intersect(bbox1=box1, bbox2=box2, img_shape=img.shape, bbox1_format=PASCAL_VOC, bbox2_format=PASCAL_VOC)

    # 2nd test case
    img = np.zeros(shape=(500, 500, 3))
    box1 = [100, 100, 250, 200]
    box2 = [0, 150, 100, 200]
    assert not bounding_boxes_intersect(bbox1=box1, bbox2=box2, img_shape=img.shape, bbox1_format=PASCAL_VOC, bbox2_format=PASCAL_VOC)

    # 3rd test case
    img = np.zeros(shape=(500, 500, 3))
    box1 = [0, 0, 400, 400]
    box2 = [400, 400, 500, 500]
    assert not bounding_boxes_intersect(bbox1=box1, bbox2=box2, img_shape=img.shape, bbox1_format=PASCAL_VOC, bbox2_format=PASCAL_VOC)

    # 4th test case
    img = np.zeros(shape=(500, 500, 3))
    box1 = [200, 200, 300, 300]
    box2 = [250, 310, 270, 340]
    assert not bounding_boxes_intersect(bbox1=box1, bbox2=box2, img_shape=img.shape, bbox1_format=PASCAL_VOC, bbox2_format=PASCAL_VOC)



if __name__ == '__main__':
    # the intersection code works for the positive case
    test_positive_intersection()
    test_positive_intersection()
