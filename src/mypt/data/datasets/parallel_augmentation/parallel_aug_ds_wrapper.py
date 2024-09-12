"""
This script contains the implementation of a dataset object functioning as a wrapper a dataset accessible through Pytorch
"""
import torch, numpy as np


import torchvision.transforms as tr

from pathlib import Path 
from typing import Union, List, Tuple, Optional
from bisect import bisect
from tqdm import tqdm

from torch.utils.data import Dataset
from torchvision.datasets import Food101, Imagenette

from mypt.data.datasets.parallel_augmentation.parallel_aug_abstract import AbstractParallelAugsDs


class ParallelAugWrapperDS(AbstractParallelAugsDs):
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

        self.root_dir = root_dir
        self._ds: Dataset = None
        self.train = train
        self.samples_per_cls_map = None
        self.classification_mode = classification_mode


    # @abstractmethod
    def _set_samples_per_cls(self, samples_per_cls: int):
        # iterate through the dataset
        current_cls = None
        
        last_pointer = 0
        cls_count = 0

        mapping = {0: 0}
        
        for i in tqdm(range(len(self._ds)), desc="iterating through the dataset to set the samples per each class"):
        # for i in tqdm(range(3150), desc="iterating through the dataset to set the samples per each class"):
            _, c = self._ds[i]
            
            if current_cls is None:
                current_cls = c
            
            if current_cls == c:
                cls_count += 1    

                if cls_count == samples_per_cls:
                    last_pointer = last_pointer + cls_count
                    

            else:
                if cls_count <= samples_per_cls:
                    last_pointer = last_pointer + cls_count

                mapping[last_pointer] = i 

                cls_count = 1              
                current_cls = c          

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
        if self._len <=  0:
            raise ValueError(f"Make sure to set the 'self._len' attribute. The exact number has to be overriden by the child class")

        return self._len


class Food101Wrapper(ParallelAugWrapperDS):
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

        super().__init__(
                root_dir=root_dir,
                output_shape=output_shape,
                augs_per_sample=augs_per_sample,
                sampled_data_augs=sampled_data_augs,
                uniform_augs_before=uniform_augs_before,
                uniform_augs_after=uniform_augs_after,
                train=train,
                samples_per_cls=samples_per_cls,
                classification_mode=classification_mode)

        if samples_per_cls is not None:
            self._set_samples_per_cls(samples_per_cls)

        ds_transform = [tr.ToTensor(), tr.Resize(size=output_shape)] + uniform_augs_before + uniform_augs_after

        self._ds = Food101(root=root_dir,     
                         split='train' if train else 'test',
                         transform=tr.Compose(ds_transform) if classification_mode else None, 
                         download=True)
        
        if samples_per_cls is not None:
            self._len = 101 * samples_per_cls
        else:
            self._len = len(self._ds)
        
    def _set_samples_per_cls(self, samples_per_cls: int):        
        # instead of a all rounded function that works for all cases, we will suppose, for the sake of efficiency
        # that each class has exacty 750 images in the training dataset, and 250 in the validation 
        # the other implementation can be found in the "ds_wrapper_.py" script
        if self.train:
            mapping = {samples_per_cls * i: 750 * i for i in range(101)}
        else:
            mapping = {samples_per_cls * i: 250 * i for i in range(101)}         

        self.samples_per_cls_map = mapping


class ImagenetterWrapper(ParallelAugWrapperDS):
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

        super().__init__(
                root_dir=root_dir,
                output_shape=output_shape,
                augs_per_sample=augs_per_sample,
                sampled_data_augs=sampled_data_augs,
                uniform_augs_before=uniform_augs_before,
                uniform_augs_after=uniform_augs_after,
                train=train,
                samples_per_cls=samples_per_cls,
                classification_mode=classification_mode)

        if samples_per_cls is not None:
            self._set_samples_per_cls(samples_per_cls)

        ds_transform = [tr.ToTensor(), tr.Resize(size=output_shape)] + uniform_augs_before + uniform_augs_after

        self._ds = Imagenette(root=self.root_dir,     
                         split='train' if train else 'test',
                         transform=tr.Compose(ds_transform) if classification_mode else None, 
                         download=True,
                         size='full')
        
        if samples_per_cls is not None:
            self._len = 10 * samples_per_cls
        else:
            self._len = len(self._ds)
    
    # the number of samples varies per class, we we will use the parent function
    
