"""
This script contains an implementation of an Object Detection Dataset
"""

import itertools

from pathlib import Path
from typing import Union, Optional, Dict, Tuple, List
from torch.utils.data import Dataset


from ...code_utilities import directories_and_files as dirf, annotation_utilites as au



# let's define a common datatype
my_iter = Union[Tuple, List]

class ObjectDetectionDs(Dataset):
    __supported_formats = ['pascal_voc', 'albumentations', 'coco', 'yolo']
    # the supported formats can be found in the following page of the albumentations documentation:
    # https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    
    
    @classmethod
    def _verify_single_annotation(cls, annotation: my_iter, label_type: Union[str, type] = None) -> str:
        if not isinstance(annotation, (Tuple, List)) or len(annotation) != 2:
            raise ValueError(f"Make sure each annotation is either a list of a tuple of length 2. Found type: {type(annotation)} and length: {len(annotation)}")

        # the first element should represent the class labels 
        # the second element should represent the bounding boxes

        if len(annotation[0]) != len(annotation[1]):
            raise ValueError(f"The length of the class labels must match the lenght of the bounding boxes. Found: {len(annotation[0])} and {len(annotation[1])}")

        cls, ann = annotation

        # proceed with checking the class labels
        if label_type is None:
            for c in cls:
                if not (isinstance(c, (str, int)) or isinstance(c, type(cls[0]))):
                    raise ValueError(f"The class labels must of types {int} or {str}. Found {c} of type {type(c)} + make sure all class labels are of the same type")
            
            label_type = type(cls[0]) 
        
        else:
            if isinstance(label_type, str):
                label_type = eval(label_type)

            for c in cls:
                if not isinstance(c, label_type):
                    raise ValueError(f"make sure all class labels are of the same type")

        au.verify_object_detection_annotation(annotation=ann)

        return label_type

    @classmethod
    def __convert2yolo(cls, 
                       img_shape: Tuple[int, int], 
                       ann: my_iter, 
                       current_format: str) -> my_iter:
        img_w, img_h = img_shape
        



    @classmethod
    def _convert_annotations(cls, 
                             ann: my_iter, 
                             current_format: str, 
                             target_format: str) -> my_iter:
        pass


    def __init__(self,
                 root_dir: Union[str, Path],

                 img_annotations: Optional[Dict],
                 img2ann_dict: Optional[Dict], 
                 read_ann: Optional[callable],
                 
                 current_format: Optional[str],
                 des_format: str,
                 convert: Union[bool, callable]=True,

                 add_background_label:bool=False,
                 image_extensions: Optional[List[str]]=None
                ) -> None:
        # init the parent class
        super().__init__()

        if image_extensions is None:
            image_extensions = dirf.IMAGE_EXTENSIONS 

        root_dir = dirf.process_path(root_dir, 
                        dir_ok=True, 
                        file_ok=False,
                        condition=lambda d: dirf.image_directory(d, image_extensions=image_extensions),
                        error_message=f'the directory is expected to contain only image files')

        
        ######################### annotation verification #########################
        if img_annotations is None and (img2ann_dict is None or read_ann is None):
            raise ValueError(f"Make sure to pass either 'img_annotations' argument or both of 'img2ann_dict' and 'read_ann'")
        
        
