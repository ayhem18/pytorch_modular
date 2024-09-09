import torch, numpy as np


import torchvision.transforms as tr

from pathlib import Path 
from typing import Union, List, Tuple, Optional
from bisect import bisect
from tqdm import tqdm

from torchvision.datasets import Food101

from mypt.data.datasets.parallel_augmentation.parallel_aug_abstract import AbstractParallelAugsDs

class Food101Wrapper(AbstractParallelAugsDs):
    def __init__(self, 
                 root_dir: Union[str, Path], 
                output_shape: Tuple[int, int],
                augs_per_sample: int,
                sampled_data_augs:List,
                uniform_augs_before: List,
                uniform_augs_after: List,
                train:bool=True,
                samples_per_cls: Optional[int] = None,
                classification_mode: bool=False) -> None:
        
        if classification_mode and len(sampled_data_augs) > 0:
            raise ValueError(f"In classification mode, there should be no sampled data augmentations...")

        super().__init__(output_shape=output_shape, 
                         augs_per_sample=augs_per_sample,
                         sampled_data_augs=sampled_data_augs,                         
                         uniform_augs_before=uniform_augs_before,
                         uniform_augs_after=uniform_augs_after)
        
        # add all of the uniform data augmentations
        ds_transform = [tr.ToTensor(), tr.Resize(size=output_shape)] + uniform_augs_before + uniform_augs_after

        self._ds = Food101(root=root_dir,     
                         split='train' if train else 'test',
                         transform=tr.Compose(ds_transform) if classification_mode else None, 
                         download=True)
        self.train = train

        self.samples_per_cls_map = None

        if samples_per_cls is not None:
            self._len = 101 * samples_per_cls
            self.__set_samples_per_cls(samples_per_cls) 
        else:
            self._len = len(self._ds)

        self.classification_mode = classification_mode


    def __set_samples_per_cls(self, samples_per_cls: int):        
        # instead of a all rounded function that works for all cases, we will suppose, for the sake of efficiency
        # that each class has exacty 750 images in the training dataset, and 250 in the validation 
        # the other implementation can be found in the "ds_wrapper_.py" script
        if self.train:
            mapping = {samples_per_cls * i: 750 * i for i in range(101)}
        else:
            mapping = {samples_per_cls * i: 250 * i for i in range(101)}         

        self.samples_per_cls_map = mapping


    def __get_item_default_(self, index:int) -> Tuple[torch.Tensor, torch.Tensor]:
        # extract the path to the sample (using the map between the index and the sample path !!!)
        if self.samples_per_cls_map is not None:            
            # extract the indices of first samples of each class in the wrapper
            cls_first_sample_indices = sorted(list(self.samples_per_cls_map.keys()))

            index_in_first_sample_list = bisect(cls_first_sample_indices, index) - 1

            cls_index_first_sample_wrapper = cls_first_sample_indices[index_in_first_sample_list]

            cls_index_first_sample_original = self.samples_per_cls_map[cls_index_first_sample_wrapper]

            final_index = cls_index_first_sample_original + (index - cls_index_first_sample_wrapper)
            
            sample_image:torch.Tensor = self._ds[final_index][0]
        else:
            sample_image = self._ds[index][0]   

        augs1, augs2 = self._set_augmentations()
        s1, s2 = augs1(sample_image), augs2(sample_image) 

        return s1, s2


    def __get_item_cls_(self, index:int) -> Tuple[torch.Tensor, int]:
        # extract the path to the sample (using the map between the index and the sample path !!!)
        if self.samples_per_cls_map is not None:

            # extract the indices of first samples of each class in the wrapper
            cls_first_sample_indices = sorted(list(self.samples_per_cls_map.keys()))

            index_in_first_sample_list = bisect(cls_first_sample_indices, index) - 1

            cls_index_first_sample_wrapper = cls_first_sample_indices[index_in_first_sample_list]

            cls_index_first_sample_original = self.samples_per_cls_map[cls_index_first_sample_wrapper]

            final_index = cls_index_first_sample_original + (index - cls_index_first_sample_wrapper)

            return self._ds[final_index]
        
        item = self._ds[index]
        return item

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, torch.Tensor], 
                                               Tuple[torch.Tensor, List[int]]
                                               ]:
        if self.classification_mode:
            return self.__get_item_cls_(index)

        return self.__get_item_default_(index)

    def __len__(self) -> int:
        return self._len
