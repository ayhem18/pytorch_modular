"""
This script contains the code to generate visual samples used to test the box_utilities code
"""

import os
import numpy as np

from PIL import Image

from mypt.code_utilities import directories_and_files as dirf
from mypt.visualization.object_detection import draw_multi_bbox

SCRITP_DIR = os.path.dirname(os.path.realpath(__file__))
IMAGES_DIR = os.path.join(SCRITP_DIR, 'bbox_test_images')


def generate_intersection_samples():
    intersection_dir = dirf.process_path(os.path.join(IMAGES_DIR, 'intersection', 'intersection_samples'), dir_ok=True, file_ok=False, must_exist=False)
    # this function generates and save images where the boxes intersect
    
    # first test case
    img = np.zeros(shape=(500, 500, 3))
    box1 = [100, 100, 250, 200]
    box2 = [200, 50, 300, 150]

    img = draw_multi_bbox(img, [box1, box2]).astype(np.uint8)    
    Image.fromarray(img).save(os.path.join(intersection_dir, 'test_case1.png'))
    

    # 2nd test case
    img = np.zeros(shape=(500, 500, 3))
    box1 = [100, 100, 250, 200]
    box2 = [150, 50, 300, 350]

    img = draw_multi_bbox(img, [box1, box2]).astype(np.uint8)    
    Image.fromarray(img).save(os.path.join(intersection_dir, 'test_case2.png'))

    # 3rd test case
    img = np.zeros(shape=(500, 500, 3))
    box1 = [0, 0, 400, 400]
    box2 = [100, 100, 200, 200]
    img = draw_multi_bbox(img, [box1, box2]).astype(np.uint8)    
    Image.fromarray(img).save(os.path.join(intersection_dir, 'test_case3.png'))

    # 4th test case
    img = np.zeros(shape=(500, 500, 3))
    box1 = [0, 0, 300, 300]
    box2 = [100, 100, 200, 400]
    img = draw_multi_bbox(img, [box1, box2]).astype(np.uint8)    
    Image.fromarray(img).save(os.path.join(intersection_dir, 'test_case4.png'))

def generate_no_intersection_samples():
    no_intersection_dir = dirf.process_path(os.path.join(IMAGES_DIR, 'intersection', 'no_intersection_samples'), dir_ok=True, file_ok=False, must_exist=False)
    # this function generates and save images where the boxes intersect
    
    # first test case
    img = np.zeros(shape=(500, 500, 3))
    box1 = [100, 100, 250, 200]
    box2 = [150, 0, 200, 50]

    img = draw_multi_bbox(img, [box1, box2]).astype(np.uint8)    
    Image.fromarray(img).save(os.path.join(no_intersection_dir, 'test_case1.png'))
    

    # 2nd test case
    img = np.zeros(shape=(500, 500, 3))
    box1 = [100, 100, 250, 200]
    box2 = [0, 150, 100, 200]

    img = draw_multi_bbox(img, [box1, box2]).astype(np.uint8)    
    Image.fromarray(img).save(os.path.join(no_intersection_dir, 'test_case2.png'))



    # 3rd test case
    img = np.zeros(shape=(500, 500, 3))
    box1 = [0, 0, 400, 400]
    box2 = [400, 400, 500, 500]
    img = draw_multi_bbox(img, [box1, box2]).astype(np.uint8)    
    Image.fromarray(img).save(os.path.join(no_intersection_dir, 'test_case3.png'))

    # 4th test case
    img = np.zeros(shape=(500, 500, 3))
    box1 = [200, 200, 300, 300]
    box2 = [250, 310, 270, 340]
    img = draw_multi_bbox(img, [box1, box2]).astype(np.uint8)    
    Image.fromarray(img).save(os.path.join(no_intersection_dir, 'test_case4.png'))
    
    pass

if __name__ == '__main__':
    generate_intersection_samples()
    generate_no_intersection_samples()
