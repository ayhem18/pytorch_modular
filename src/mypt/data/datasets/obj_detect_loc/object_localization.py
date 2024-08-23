"""
This script contains an implementation of a dataset class designed for the object Localization task (an image can have at most one object of interest)
"""

import torch
import albumentations as A


from torchvision import transforms as tr
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple


from .abstract_ds import ObjectDataset

class ObjectDetectionDs(ObjectDataset):

    def __init__(self,
                 root_dir: Union[str, Path],
                 
                 img_augs: List,
                 output_shape: Tuple[int, int],

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
        super().__init__(root_dir=root_dir,
                    img_annotations=img_annotations,
                    img2ann_dict=img2ann_dict, 
                    read_ann=read_ann,
                    
                    target_format=target_format,
                    current_format=current_format,
                    convert=convert,

                    add_background_label=add_background_label,
                    image_extensions=image_extensions
                    )   

        self.augmentations = img_augs

        if len(img_augs) > 0:
            if not isinstance(img_augs[-1], (tr.Resize, A.Resize)):
                self.augmentations.append(A.Resize(output_shape[0], output_shape[1]))
            elif isinstance(img_augs[-1], tr.Resize):
                self.augmentations.pop()
                self.augmentations.append(A.Resize(output_shape[0], output_shape[1]))
            else:
                self.augmentations[-1] = A.Resize(output_shape[0], output_shape[1])

        # add a ToTensor transformation
        self.augmentations.append(tr.ToTensor())

        self.final_aug = A.Compose(self.augmentations, A.BboxParams(format=target_format, label_fields='cls_labels'))        


    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # load the sample
        sample_path = self.idx2sample_path[index]
        sample = self.load_sample(sample_path)
        # fetch the bounding boxes and the class labels
        sample_cls, sample_bbox = self.annotations[sample_path]
        
        # pass the sample through the final augmentation
        transform = self.final_aug(sample, bboxes=sample_bbox, cls_labels=sample_cls)

        # fetch the labels after augmentations
        img, cls_labels, bboxes = transform['image'], transform['cls_labels'], transform['bboxes']

        # make sure to convert the data to torch Tensors
        if not isinstance(cls_labels, torch.Tensor):
            cls_labels = torch.Tensor(cls_labels)
            bboxes = torch.from_numpy(bboxes)

        return img, cls_labels, bboxes            
