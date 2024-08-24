"""
This script contains an implementation of a dataset class designed for the object Localization task (an image can have at most one object of interest)
"""

import torch
import albumentations as A, numpy as np

from torchvision import transforms as tr
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple


from .abstract_ds import ObjectDataset

class ObjectDetectionDs(ObjectDataset):

    def __init__(self,
                 root_dir: Union[str, Path],
                 
                 img_augs: List,
                 output_shape: Tuple[int, int],

                 compact: bool,

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

        if isinstance(self.augmentations[-1], tr.ToTensor):
            self.augmentations.pop()
        
        self.final_aug = A.Compose(self.augmentations, A.BboxParams(format=target_format, label_fields='cls_labels'))        

        # this field determines whether to returns 
        self.compact = compact 

    def __getitem__(self, index) -> Union[
                                          Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], 
                                          Tuple[torch.Tensor, torch.Tensor]
                                        ]:
        # load the sample
        sample_path = self.idx2sample_path[index]
        sample = self.load_sample(sample_path)
        # fetch the bounding boxes and the class labels
        sample_cls, sample_bbox = self.annotations[sample_path]

        if len(sample_cls) > 1 or len(sample_bbox) > 1:
            raise ValueError(f"found a sample with more than one cls or bounding box !!!")

        # pass the sample through the final augmentation
        transform = self.final_aug(sample, bboxes=sample_bbox, cls_labels=sample_cls)

        # fetch the labels after augmentations
        img, cls_labels, bboxes = transform['image'], transform['cls_labels'], transform['bboxes']

        # convert the image to a torch tensor anyway
        img = tr.ToTensor()(img)

        # after applying the augmentation and adjusting the bounding boxes accordingly
        # return the final labels: depends on self.compact
        
        # first the object indicator: a boolean flat indicating whether there is an object of interest on the image or not
        object_indicator = int(cls_labels[0] != self.background_cls_index)
    
        if self.compact:
            # one hot encode the label
            cls_label_index = self.cls_2_cls_index[cls_labels[0]]
            cls_label_one_hot = [int(i == cls_label_index) for i in len(self.all_classes)]
            # concatenate everything together
            final_label = [object_indicator] + (bboxes.tolist() if isinstance(bboxes, np.ndarray) else bboxes) + cls_label_one_hot
            # convert the compact label to tensor tensor
            final_output =  img, torch.Tensor(final_label)
            return final_output

        # self.compact set to False implies that the object indicator, the bboxes and the cls labels will be returned as 3 seperated values
        return img, torch.tensor(object_indicator), torch.tensor(bboxes), torch.tensor(cls_labels[0])  

