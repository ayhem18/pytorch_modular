import os
import torch
import numpy as np

import albumentations as A


from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as tr
from typing import List, Tuple, Union, Optional

import mypt.code_utils.directories_and_files as dirf

from mypt.shortcuts import P


class SemanticSegmentationDS(Dataset):

    """
    Dataset class for semantic segmentation tasks.
    
    Args:
        data_dir (str): Directory containing the input images
        mask_dir (str): Directory containing the segmentation masks
        transforms (list, optional): List of albumentations transforms to apply. Defaults to None.
    """
    
    def __init__(self, 
                 data_dir: P, 
                 mask_dir: P, 
                 binary_mask: bool,
                 transforms: Optional[Union[List[A.BasicTransform], A.Compose]]=None,
                 ):
        
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.transforms = transforms if transforms is not None else tr.Compose([tr.ToTensor()])
        self.binary_mask = binary_mask

        if isinstance(self.transforms, List):
            # make sure all transforms are of type A.BasicTransform
            for t in self.transforms:
                if not isinstance(t, A.BasicTransform):
                    raise TypeError(f"All transforms must be of type A.BasicTransform. Found: {type(t)}")
            
            self.transforms = A.Compose(self.transforms)
        

        # Get all image filenames
        self.images = sorted([os.path.join(data_dir, img_name) for img_name in os.listdir(data_dir)
                             if os.path.splitext(img_name)[-1] in dirf.IMAGE_EXTENSIONS])
        
        # Get all mask filenames
        self.masks = sorted([os.path.join(mask_dir, mask_name) for mask_name in os.listdir(mask_dir)
                            if os.path.splitext(mask_name)[-1] in dirf.IMAGE_EXTENSIONS])
        
        # Verify that we have the same number of images and masks
        assert len(self.images) == len(self.masks), "Number of images and masks should be the same"


    @classmethod
    def _load_sample(cls, path: P) -> Image.Image:
        with open(path, "rb") as f:
            return Image.open(f).convert("RGB") 


    def __len__(self):
        # return len(self.images)
        return 10 # a small comment added for debugging purposes

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image and mask
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        # load the image and the mask        
        image, mask = self._load_sample(img_path), self._load_sample(mask_path)
        
        # Apply transformations
        if isinstance(self.transforms, tr.Compose):
            image, mask = self.transforms(image), self.transforms(mask)
        else:
            # the Albumentations transforms expect numpy arrays
            transformed = self.transforms(image=np.asarray(image).copy(), mask=np.asarray(mask).copy())
            image = transformed["image"]
            mask = transformed["mask"]
        
        if self.binary_mask:
            mask = (mask > 0.5).to(torch.float32)
        
        # make sure the image of type torch.float32
        image = image.to(torch.float32)
        
        return image, mask
