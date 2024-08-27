"""
This script contains an implementation of a dataset class designed for the object Localization task (an image can have at most one object of interest)
"""

import torch
import albumentations as A, numpy as np

from torchvision import transforms as tr
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple

from mypt.code_utilities import pytorch_utilities as pu
from .abstract_ds import ObjectDataset

class ObjectLocalizationDs(ObjectDataset):

    def __init__(self,
                 root_dir: Union[str, Path],
                 
                 img_augs: List,
                 output_shape: Tuple[int, int],

                 compact: bool,

                 image2annotation: Union[Dict, callable],
        
                 target_format: str,
                 current_format: Optional[str],

                 background_label:Union[int, str],
                 
                 convert: Optional[callable]=None,
                 image_extensions: Optional[List[str]]=None,
                 seed:int=69,
                ) -> None:

        # init the parent class
        super().__init__(root_dir=root_dir,
                    image2annotation=image2annotation,                    
                    target_format=target_format,
                    current_format=current_format,
                    convert=convert,

                    background_label=background_label,
                    image_extensions=image_extensions
                    )   

        pu.seed_everything(seed=seed)

        self.augmentations = img_augs

        if len(img_augs) > 0:
            if not isinstance(img_augs[-1], (tr.Resize, A.Resize)):
                self.augmentations.append(A.Resize(output_shape[0], output_shape[1]))
            elif isinstance(img_augs[-1], tr.Resize):
                self.augmentations.pop()
                self.augmentations.append(A.Resize(output_shape[0], output_shape[1]))
            else:
                self.augmentations[-1] = A.Resize(output_shape[0], output_shape[1])

        else:
            self.augmentations = [A.Resize(*output_shape, p=1)]

        if isinstance(self.augmentations[-1], tr.ToTensor):
            self.augmentations.pop()

        self.final_aug = A.Compose(self.augmentations, A.BboxParams(format=target_format, 
                                                                    label_fields=['cls_labels'] # make sure to pass a list !!!
                                                                    ))        

        # this field determines whether to returns 
        self.compact = compact 

    def __getitem__(self, index) -> Union[
                                          Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], 
                                          Tuple[torch.Tensor, torch.Tensor]
                                        ]:
        # load the sample
        sample_path = self.idx2sample_path[index]
        img = np.asarray(self.load_sample(sample_path).copy()) # albumentations requires the input to a numpy array
        # fetch the bounding boxes and the class labels
        cls_labels, bboxes = self.annotations[sample_path]

        if len(cls_labels) > 1 or len(bboxes) > 1:
            raise ValueError(f"found a sample with more than one cls or bounding box !!!")

        # pass the sample through the final augmentation
        transform = self.final_aug(image=img, bboxes=bboxes, cls_labels=cls_labels)

        # fetch the labels after augmentations
        img, cls_labels, bboxes = transform['image'], transform['cls_labels'][0], transform['bboxes'][0]

        # convert the image to a torch tensor 
        img = tr.ToTensor()(img.copy())
        
        # first the object indicator: a boolean flat indicating whether there is an object of interest on the image or not
        object_indicator = int(cls_labels != self.background_label)
        cls_label_index = self.cls_2_cls_index[cls_labels]
    
        if self.compact:
            # one-hot encode the label
            cls_label_one_hot = [int(i == cls_label_index) for i in range(len(self.all_classes) - 1)]
            
            assert sum(cls_label_one_hot) == int(object_indicator), "Make sure the label is all zeros for background cls and with only one value for an object of interest"

            # concatenate everything together
            final_label = [object_indicator] + (bboxes.tolist() if isinstance(bboxes, np.ndarray) else list(bboxes)) + cls_label_one_hot
            
            return img, torch.Tensor(final_label)

        # self.compact set to False implies that the object indicator, the bboxes and the cls labels will be returned as 3 seperated values
        return img, torch.tensor(object_indicator), torch.tensor(bboxes), torch.tensor(cls_label_index)  

