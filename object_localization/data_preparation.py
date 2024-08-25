"""
This script contains the code to prepare the data for training on the object localization task
"""

import os, random, numpy as np
import xml.etree.ElementTree as ET

from typing import Union, Tuple, List
from pathlib import Path

from mypt.code_utilities import directories_and_files as dirf
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


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
    return [root.find('object').find('name').text.lower()], [x_min, y_min, x_max, y_max], (size['width'], size['height']) 


def prepare_data(data_folder: Union[str, Path], ann_folder: Union[str, Path]=None):
    data_folder = dirf.process_path(data_folder, dir_ok=True, file_ok=False, 
                                    condition=lambda d: all([os.path.splitext(f)[-1].lower() in ['.xml'] + dirf.IMAGE_EXTENSIONS for f in os.listdir(d)]), # basically make sure that all files are either images of .xml files
                                    error_message=f'Make sure the directory contains only .xml or image data'
                                    )
    if ann_folder is None:
        ann_folder = os.path.join(Path(data_folder).parent, f'{os.path.basename(data_folder)}_annotations')
        ann_folder = dirf.process_path(ann_folder, dir_ok=True, file_ok=False)
        
    else:
        ann_folder = dirf.process_path(ann_folder, 
                                       dir_ok=True, 
                                       file_ok=False, 
                                       condition=lambda d: (len(os.listdir(d))) == 0, 
                                       error_message=f"The annotation dir must be empty ")

    # move the .xml files to annotation folder
    dirf.copy_directories(src_dir=data_folder, 
                          des_dir=ann_folder, 
                          copy=False, # setting copy to False will move the files 
                          filter_directories=lambda p: os.path.splitext(p)[-1].lower() == ".xml" #move onle .xml files
                          )
    # everything should be done
    return ann_folder


def get_annotation_file_path(sample_file_path: Union[str, Path]) -> Union[str, Path]:
    file_name = os.path.basename(sample_file_path)
    file_name, _ = os.path.splitext(file_name)
    return f'{file_name}.xml'    


def convert_img_2_background(img: np.ndarray, bbox:List[int]) -> np.ndarray:
    x_min, y_min, x_max, y_max = bbox
    b = img[y_min : y_max + 1, x_min: x_max + 1, :]
    padded_b = np.pad(b, ((y_min, img.shape[0] - y_max), (x_min, img.shape[1] - x_max)))

    img_b = img - padded_b

    sums = np.sum(img_b, axis=2)
    area = (img.shape[0] * img.shape[1]) - ((x_max - x_min + 1) * (y_max - y_min + 1))
    values = sums / area
    
    img[x_min: x_max + 1, y_min : y_max + 1, :] = values
    return img

def add_background_cls(data_folder: Union[str, Path],
                       ann_folder: Union[str, Path], 
                       num_samples:int, 
                       seed:int=0):
    data_folder = dirf.process_path(data_folder, dir_ok=True, file_ok=False, 
                                    condition=lambda d: all([os.path.splitext(f)[-1].lower() in dirf.IMAGE_EXTENSIONS for f in os.listdir(d)]), # basically make sure that all files are either images of .xml files
                                    error_message=f'Make sure the directory contains only image data'
                                    )

    ann_folder = dirf.process_path(data_folder, dir_ok=True, file_ok=False, 
                                    condition=lambda d: all([os.path.splitext(f)[-1].lower() == '.xml' for f in os.listdir(d)]), # basically make sure that all files are either images of .xml files
                                    error_message=f'Make sure the directory contains only xml files'
                                    )
    
    # iterate through each img in the datafolder
    # 1. extract the bounding boxes
    # 2. calculate the media pixel value outside of the bounding box for each channel
    # 3. set the pixel values within the bounding box to the estimated values in step 2

    random.seed(seed)
    files = [os.path.join(data_folder, f) for f in os.listdir(data_folder)]
    
    num_samples = int(round(len(files) * num_samples)) if isinstance(num_samples, float) else num_samples

    files = random.sample(files, num_samples)
    
    for index, f in enumerate(files, start=1):
        _, ext = os.path.splitext(f)
        img = np.asarray(Image.open(f))
        ann_path = os.path.join(ann_folder, get_annotation_file_path(f))
        _, bbox, _ = extract_annotation_from_xml(ann_path)
        bg_img = convert_img_2_background(img, bbox)                
        # save the image 
        Image.fromarray(bg_img).save(os.path.join(data_folder, f'background_{index}.{ext}'))    


def split_data_train_val(data_folder: Union[str, Path], val_split:float=0.15):
    train_dir = os.path.join(Path(data_folder).parent, f'{os.path.basename(data_folder)}_train')
    val_dir = os.path.join(Path(data_folder).parent, f'{os.path.basename(data_folder)}_val')

    dirf.directory_partition(src_dir=data_folder, des_dir=val_dir, copy=False, portion=val_split)
    dirf.copy_directories(src_dir=data_folder, des_dir=train_dir, copy=True) # set copy to False to save disk space


if __name__ == '__main__':
    folder = os.path.join(SCRIPT_DIR, 'plants_ds')    
    # data preparation is taken across 3 steps:

    # move the annotation to a different folder
    # add some background images
    # split the data into train and validation splits

    img = np.random.randint(0, 255, (7, 7, 3))
    print(img)
    bbox = [2, 2, 4, 4]
    new_img = convert_img_2_background(img, bbox)
