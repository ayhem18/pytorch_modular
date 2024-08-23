import os
import numpy as np

from pathlib import Path
from typing import Union, Optional, Dict, Tuple, List
from torch.utils.data import Dataset
from PIL import Image

from abc  import ABC, abstractmethod

from mypt.code_utilities import directories_and_files as dirf, annotation_utilites as au


# let's define a common datatype
my_iter = Union[Tuple, List]

class ObjectDataset(Dataset, ABC):
    """
    This dataset was designed as an abstract dataset for the object localization and detection tasks 
    """
    
    # the supported formats can be found in the following page of the albumentations documentation:
    # https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    __supported_formats = au.OBJ_DETECT_ANN_FORMATS
    
    
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
            
        ann = [au.verify_object_detection_annotation(a) for a in ann]
        return ann, label_type

    @classmethod
    def load_sample(cls, sample_path: Union[str, Path]):
        # make sure the path is absolute
        if not os.path.isabs(sample_path):
            raise ValueError(f"The loader is expecting the sample path to be absolute.\nFound:{sample_path}")

        # this code is copied from the DatasetFolder python file
        with open(sample_path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
    

    def __set_img_annotations(self, 
                               img_annotations: Dict,
                               current_format: str, 
                               convert: callable):

        # if not isinstance(img_annotations, Dict):
        #     raise TypeError(f"the prepared image annotations must of the type Dict. Found: {type(img_annotations)}")

        # read the image files
        img_files = sorted([os.path.join(self.root_dir, img) for img in os.listdir(self.root_dir)])

        # make sure the keys match the image files
        keys = sorted([k if os.path.isabs(k) else os.path.join(self.root_dir, k) for k in list(img_annotations.keys())])
        
        if keys != img_files:
            raise ValueError(f"Please make sure to pass an annotation to all files in the root directory.")

        # verify the annotations
        _, label_type = self._verify_single_annotation(img_annotations.items()[0][1], label_type=None)
        
        for key, ann in img_annotations.items():
            flattened_ann, _ = self._verify_single_annotation(annotation=ann, label_type=label_type)
            img_annotations[key] = flattened_ann

        # annotations verified !! final step: convert to the target format
        for img_path, img_ann in img_annotations.items():
            
            # load the image shape
            img_shape = (np.asarray(self.load_sample(img_path)).shape)[:2], # self.load_sample return a PIL.Image, convert to numpy array whose shape would be [w, h, 3]
            cls_ann, bbox_ann = img_ann

            if current_format is not None:
                bbox_ann = [np.asarray(au.convert_annotations(annotation=b, 
                                                    current_format=current_format,
                                                    target_format=self.target_format) 
                                                    for b in bbox_ann)]
            else:
                try:
                    bbox_ann = [convert(b, img_shape=img_shape) for b in bbox_ann]
                except:
                    try:
                        bbox_ann = [convert(b, img_shape=img_shape) for b in bbox_ann]
                    except:
                        raise ValueError(f"the 'convert' callable should accept only the bounding box as an input or bbox + the shape of the image as a keyword argument: 'img_shape'")

            img_annotations[img_path] = [cls_ann, bbox_ann]

        return img_annotations


    def __read_img_annotations(self, 
                                img_path_2_img_ann: Dict,
                                read_ann: callable, 
                                current_format: str, 
                                convert: callable):
        # read the image files
        img_files = sorted([os.path.join(self.root_dir, img) for img in os.listdir(self.root_dir)])

        # make sure the keys match the image files
        keys = sorted([k if os.path.isabs(k) else os.path.join(self.root_dir, k) for k in list(img_path_2_img_ann.keys())])
        
        if keys != img_files:
            raise ValueError(f"Please make sure to pass an annotation to all files in the root directory.")

        # build a map between the image paths and their corresponding 
        img_anns = {img_path: read_ann(ann) for img_path, ann in img_path_2_img_ann.items()}

        return self.__set_img_annotations(img_annotations=img_anns, current_format=current_format, convert=convert)


    def __init__(self,
                 root_dir: Union[str, Path],

                 img_annotations: Optional[Dict],
                 img2ann_dict: Optional[Dict], 
                 read_ann: Optional[callable],
                 
                 target_format: str,
                 current_format: Optional[str],
                 convert: Optional[callable]=None,

                 add_background_label:bool=False,
                 image_extensions: Optional[List[str]]=None
                ) -> None:
        # init the parent class
        super().__init__()

        if image_extensions is None:
            image_extensions = dirf.IMAGE_EXTENSIONS 

        self.root_dir = dirf.process_path(root_dir, 
                        dir_ok=True, 
                        file_ok=False,
                        condition=lambda d: dirf.image_directory(d, image_extensions=image_extensions),
                        error_message=f'the directory is expected to contain only image files')

        ######################### annotation formats #########################
        # the target format must be supported as it will be passed to the augmentations 
        if not target_format in self.__supported_formats:
            raise NotImplementedError(f"The dataset supports only the following formats: {self.__supported_formats}. Found target format: {target_format}") 

        if not current_format in self.__supported_formats and convert is None:
            raise ValueError(f"either the 'current_format' argument or the 'convert' argument must be specified to convert the annotations to the target format")
        
        self.target_format = target_format

        ######################### annotation verification #########################
        if img_annotations is None and (img2ann_dict is None or read_ann is None):
            raise ValueError(f"Make sure to pass either 'img_annotations' argument or both of 'img2ann_dict' and 'read_ann'")

        if img_annotations is not None:
            self.annotations = self.__set_img_annotations(img_annotations, 
                                                          current_format, 
                                                          convert)
        else:
            self.annotations = self.__read_img_annotations()

        self.idx2sample_path = dict(enumerate(sorted([os.path.join(self.root_dir, img) for img in os.listdir(self.root_dir)])))

        self.str_cls_2_index_cls = None 

        # if the labels are passed as string, they need to be converted to integer indices
        _, label_type = self._verify_single_annotation(self.annotations.items()[0][1], label_type=None)

        if label_type in [str, 'str']:
            all_classes = set()    
            for _, v in self.annotations:
                cls_ann, _ = v
                all_classes.update([c.lower() for c in cls_ann])
            
            self.str_cls_2_index_cls = dict([(c, i) for i, c in enumerate(sorted(list(all_classes)), start=1)])

            # convert to indices
            for k, v in self.annotations:
                cls_ann, bbox_ann = v
                cls_ann = [self.str_cls_2_index_cls[c] for c in cls_ann]
                
                if add_background_label and len(cls_ann) == 0:
                    cls_ann = [0]
                    bbox_ann = [[0, 0, 0, 0]]
                    
                self.annotations[k] = cls_ann, bbox_ann

            return 

        # at this point classes were already saved as indices
        all_classes = set()    
        for _, v in self.annotations:
            cls_ann, _ = v
            all_classes.update([c.lower() for c in cls_ann])

        for k, v in self.annotations:
            if add_background_label and len(cls_ann) == 0:
                cls_ann = [max(all_classes) + 1 if 0 in all_classes else 0]
                bbox_ann = [[0, 0, 0, 0]]
                self.annotations[k] = cls_ann, bbox_ann
            
    
    @abstractmethod
    def __getitem__(self, index):
        pass
