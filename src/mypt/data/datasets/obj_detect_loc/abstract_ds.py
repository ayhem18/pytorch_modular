import os, warnings
import numpy as np

from pathlib import Path
from typing import Union, Optional, Dict, Tuple, List
from torch.utils.data import Dataset
from PIL import Image
from copy import copy

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
    def _verify_single_label(cls, annotation: my_iter, label_type: Optional[Union[str, type]]=None) -> str:
        if not isinstance(annotation, (Tuple, List)) or len(annotation) != 2:
            raise ValueError(f"Make sure each annotation is either a list of a tuple of length 2. Found type: {type(annotation)} and length: {len(annotation)}")

        # the first element should represent the class labels 
        # the second element should represent the bounding boxes
        if not (isinstance(annotation[0], (Tuple, List)) and isinstance(annotation[1], (Tuple, List))): 
            raise TypeError(f"The class and bounding boxes must be passed as iterables of the same length. Found: classes as {type(annotation[0])}, bboxes as {type(annotation[1])}")
            
        if len(annotation[0]) != len(annotation[1]):
            raise ValueError(f"The class and bounding boxes must be passed as iterables of the same length. Found: {len(annotation[0])} and {len(annotation[1])}")

        cls, ann = annotation

        # proceed with checking the class labels
        if label_type is None:
            for c in cls:
                if not (isinstance(c, (str, int)) or isinstance(c, type(cls[0]))):
                    raise ValueError(f"The class labels must of types {int} or {str}. Found {c} of type {type(c)} + make sure all class labels are of the same type")
            try:
                label_type = type(cls[0]) 
            except KeyError:
                raise ValueError(f"The labels cannot be empty. Found annotation with at least on empty label, cls: {cls}, bbox: {ann}")
        else:
            if isinstance(label_type, str):
                label_type = eval(label_type)

            for c in cls:
                if not isinstance(c, label_type):
                    raise ValueError(f"make sure all class labels are of the same type")
            
        ann = [au.verify_object_detection_bbox(a) for a in ann]
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
    

    def __verify_img_annotation_map(self, image2annotation: Union[Dict, callable]):
        """
        The keys have to be sample file paths
        The values can be either: 
            1. a tuple of two iterables: image classes, bounding boxes
            2. a tuple of three iterables: image classes, bounding boxes, and iage 
            3. a callable that returns the first option
            4. a callable that returns the second option
            
        This functions verifies the provided mapping satisfies the requirements
        Args:
            image2annotation (Dict): a dictionary that maps a sample file path to its corresponding annotation
        """        

        if not (isinstance(image2annotation, Dict) or callable(image2annotation)):
            raise TypeError(f"the dataset expects the image 2 annotation mapping to be either a Dict or a callable object. Found: {type(image2annotation)}")

        is_callable = False

        if isinstance(image2annotation, Dict):
            # read the image files
            img_files = sorted([os.path.join(self.root_dir, img) for img in os.listdir(self.root_dir)])

            # make sure the keys match the image files
            keys = sorted([k if os.path.isabs(k) else os.path.join(self.root_dir, k) for k in list(image2annotation.keys())])
                    
            if keys != img_files:
                raise ValueError(f"Please make sure to pass an annotation to all files in the root directory.")

            k, val = image2annotation.items()[0]

        else:
            is_callable = True
            k = os.path.join(self.root_dir, os.listdir(self.root_dir)[0])
            try:
                val = image2annotation(k)
            except Exception as e:
                raise ValueError(f"calling the callable with a sample file path raised the following error: {e}")

        ann = copy(val)

        if  not isinstance(ann, (Tuple, List)):
            raise TypeError(f"The img2ann includes a callable object that does not return a tuple or a list. The callable returns an object of type: {type(ann)}")
            
        if len(ann) not in [2, 3]:
            raise ValueError(f"The img2ann mapping returns an iterable of length different from 2 and 3: {ann}")
        
        self._verify_single_label(annotation=val[:2])

        if len(val) == 2:
            # first raise a warning letting the user know that passing only the annotation would require laoding the samples to extract their shapes
            # which is preferably avoided
            warnings.warn(message=f"not passing the image shape requires loading all the samples !!")    

        return is_callable


    def __set_img_annotations(self, 
                               image2annotation: Union[Dict, callable],
                               current_format: Optional[str], 
                               convert: callable):

        # let's verify that the mapping corresponds to the expected format
        is_callable = self.__verify_img_annotation_map(image2annotation=image2annotation)

        # copy the mapping to avoid modifying the input
        if is_callable:
            # build a dictionary between the sample path and its annotation
            img_files = [os.path.join(self.root_dir, img) for img in os.listdir(self.root_dir)]
            img_annotations = {f: image2annotation(f) for f in img_files}
        else:
            img_annotations = image2annotation.copy()

        # since dict.items() method is not subscriptable, to get a random element, one needs to convert the entire dictionary either to a list or an iterator
        # which is computationally inefficient
        
        for key, annotation in img_annotations.items():
            _, cls_label_type = self._verify_single_label(annotation=annotation[:2])
            break # breaking after one iteration, as only the one label type is needed
        
        for key, annotation in img_annotations.items():
            flattened_ann, _ = self._verify_single_label(annotation=annotation[:2], label_type=cls_label_type)
            img_annotations[key] = [annotation[0], flattened_ann] + list(annotation[2:])

        # annotations verified !! final step: convert to the target format
        for img_path, img_ann in img_annotations.items():
            
            if len(img_ann) == 3:
                # load the image shape  
                cls_ann, bbox_ann, img_shape = img_ann
            else:
                img_shape = (np.asarray(self.load_sample(img_path)).shape)[:2] # self.load_sample return a PIL.Image, convert to numpy array of shape [w, h, 3]
                cls_ann, bbox_ann = img_ann

            if current_format is not None:
                # the bbox annotation supposedly contains the bounding box for each object in the image
                # convert each bounding box to the target format
                bbox_ann = [au.convert_bbox_annotation(annotation=b, 
                                                    current_format=current_format,
                                                    target_format=self.target_format, 
                                                    img_shape=img_shape) 
                                                    for b in bbox_ann]
            else:
                try:
                    bbox_ann = [convert(b, img_shape=img_shape) for b in bbox_ann]
                except:
                    try:
                        bbox_ann = [convert(b) for b in bbox_ann]
                    except:
                        raise ValueError(f"the 'convert' callable should accept only the bounding box as an input or bbox + the shape of the image as a keyword argument: 'img_shape'")

            img_annotations[img_path] = [cls_ann, bbox_ann]

        return img_annotations


    def __set_indices(self):
        self.all_classes = set()
        # save all the classes in the data
        for _, v in self.annotations.items():
            cls_ann, _ = v
            self.all_classes.update([c.lower() for c in cls_ann])

        self.cls_2_cls_index = dict([(c, i) for i, c in enumerate(sorted(list([c for c in self.all_classes if c != self.background_label])), start=0)])
        self.cls_2_cls_index[self.background_label] = len(self.cls_2_cls_index)
        # revert the mapping
        self.cls_index_2_cls = {v: k for k, v in self.cls_2_cls_index.items()}

        self.all_classes = sorted([k for k, _ in self.cls_2_cls_index.items()], key=self.cls_2_cls_index.get)
        assert self.all_classes[-1] == self.background_label, "make sure the background label is placed at the last position in the self.all_classes list"

    def __init__(self,
                 root_dir: Union[str, Path],

                 image2annotation: Union[Dict, callable],
        
                 target_format: str,
                 current_format: Optional[str],
                 background_label:Union[int, str],

                 convert: Optional[callable]=None,
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

        self.data_count = len(os.listdir(root_dir))

        ######################### annotation formats #########################
        # the target format must be supported as it will be passed to the augmentations 
        if not target_format in self.__supported_formats:
            raise NotImplementedError(f"The dataset supports only the following formats: {self.__supported_formats}. Found target format: {target_format}") 

        if not current_format in self.__supported_formats and convert is None:
            raise ValueError(f"either the 'current_format' argument or the 'convert' argument must be specified to convert the annotations to the target format")
        
        self.target_format = target_format

        ######################### annotation verification #########################
        self.annotations = self.__set_img_annotations(image2annotation, 
                                                        current_format, 
                                                        convert)

        ######################### class conversion #########################        
        self.idx2sample_path = dict(enumerate(sorted([os.path.join(self.root_dir, img) for img in os.listdir(self.root_dir)])))

        # a dictionary that maps the original classes to their numerical indices (add more flexibility to the class)
        self.cls_2_cls_index = None     
        self.cls_index_2_cls = None # mapping the index to the original class

        # if the user passes the background label, then save it, otherwise it will be automatically deduced
        self.background_label = background_label 
        # a set of all class indices
        self.all_classes = None
        self.__set_indices()
    
    def __len__(self) -> int:
        return self.data_count

    @abstractmethod
    def __getitem__(self, index):
        pass
