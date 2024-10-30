"""
This script contains the implementation of a dataset object functioning as a wrapper a dataset accessible through Pytorch
"""
import torch

from pathlib import Path 
from typing import Union, List, Tuple, Optional

from torch.utils.data import Dataset
from torchvision.datasets import Food101, Imagenette

from mypt.data.datasets.parallel_augmentation.parallel_aug_abstract import AbstractParallelAugsDs
from mypt.data.datasets.mixins.cls_ds_wrapper import ClassificationDsWrapper

class ParallelAugWrapperDS(ClassificationDsWrapper, AbstractParallelAugsDs):
    def __init__(self, 
                root_dir: Union[str, Path], 
                output_shape: Tuple[int, int],
                augs_per_sample: int,
                sampled_data_augs:List,
                uniform_augs_before: List,
                uniform_augs_after: List,
                train:bool=True) -> None:

        super().__init__(output_shape=output_shape, 
                         augs_per_sample=augs_per_sample,
                         sampled_data_augs=sampled_data_augs,                         
                         uniform_augs_before=uniform_augs_before,
                         uniform_augs_after=uniform_augs_after)

        self.root_dir = root_dir
        self._ds: Dataset = None
        self._len:int = None
        self.train = train
        self.samples_per_cls_map = {} # initialize the samples_per_cls_map to an empty dictionary


    def __get_item_default_(self, index:int) -> Tuple[torch.Tensor, torch.Tensor]:
        # extract the path to the sample (using the map between the index and the sample path !!!) 
        if len(self.samples_per_cls_map) > 0:            
            final_index = self._find_final_index(index)

            sample_image:torch.Tensor = self._ds[final_index][0]
        else:
            sample_image:torch.Tensor = self._ds[index][0]   

        augs1, augs2 = self._set_augmentations()
        s1, s2 = augs1(sample_image), augs2(sample_image) 

        return s1, s2


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
                samples_per_cls: Optional[int] = None) -> None:

        super().__init__(
                root_dir=root_dir,
                output_shape=output_shape,
                augs_per_sample=augs_per_sample,
                sampled_data_augs=sampled_data_augs,
                uniform_augs_before=uniform_augs_before,
                uniform_augs_after=uniform_augs_after,
                train=train)


        self._ds = Food101(root=root_dir,     
                         split='train' if train else 'test',
                         transform=None,#tr.Compose(ds_transform) if classification_mode else None, 
                         download=True)

        # call the self._set_samples_per_cls method after setting the self._ds field
        if samples_per_cls is not None:
            self.samples_per_cls_map = self._set_samples_per_cls(samples_per_cls)
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

        return mapping


class ImagenetterWrapper(ParallelAugWrapperDS):
    def __init__(self, 
                root_dir: Union[str, Path], 
                output_shape: Tuple[int, int],
                augs_per_sample: int,
                sampled_data_augs:List,
                uniform_augs_before: List,
                uniform_augs_after: List,
                train:bool=True,
                samples_per_cls: Optional[int] = None) -> None:

        super().__init__(
                root_dir=root_dir,
                output_shape=output_shape,
                augs_per_sample=augs_per_sample,
                sampled_data_augs=sampled_data_augs,
                uniform_augs_before=uniform_augs_before,
                uniform_augs_after=uniform_augs_after,
                train=train)

        # for some reason, setting the download parameter to True raises an error if the directory already exists
        # wrap the self._ds field in a try and catch statment to cover all cases (setting the download argument with whether the directly exists or not is not enough as certain files might be missing...)
        try:
            self._ds = Imagenette(root=self.root_dir,     
                            split='train' if train else 'val', # the split argument must either 'train' or 'val'
                            transform=None,
                            download=True,  
                            size='full')
        except RuntimeError as e:
            if 'dataset not found' not in str(e).lower():
                self._ds = Imagenette(root=self.root_dir,     
                                split='train' if train else 'val',
                                transform=None, 
                                download=False,  
                                size='full')
            else:
                raise e
            
        if samples_per_cls is not None:
            self.samples_per_cls_map = self._set_samples_per_cls(samples_per_cls)
            self._len = 10 * samples_per_cls
        else:
            self._len = len(self._ds)
    
    def _set_samples_per_cls(self, samples_per_cls: int):
        # the number of samples varies per class: use the parent function
        return super()._set_samples_per_cls(samples_per_cls=samples_per_cls)

