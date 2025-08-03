"""
This script contains an implementation of a generic Dataset Object that returns all images in a given directory
without imposing a specific structure.
"""
import os, torch
import torchvision.transforms as tr

from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, Union, List, Tuple, Dict, Optional
from torchvision.datasets import Food101, Imagenette, MNIST


from mypt.shortcuts import P
from mypt.code_utils import directories_and_files as dirf
from mypt.data.datasets.mixins.cls_ds_wrapper import ClassificationDsWrapperMixin


class GenericFolderDS(Dataset):
    @classmethod
    def load_sample(cls, sample_path: P):
        # make sure the path is absolute
        if not os.path.isabs(sample_path):
            raise ValueError(f"The loader is expecting the sample path to be absolute.\nFound:{sample_path}")

        # this code is copied from the DatasetFolder python file
        with open(sample_path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def _set_image_transforms(self, image_transforms: List) -> List:
        # add to.Tensor to the transforms if it is not already in the list
        if not any([isinstance(t, tr.ToTensor) for t in image_transforms]):
            image_transforms.append(tr.ToTensor())

        return image_transforms

    def _set_item_transforms(self, item_transforms: Optional[Callable[[torch.Tensor], torch.Tensor]]) -> Callable[[torch.Tensor], torch.Tensor]:
        if item_transforms is None:
            return lambda x: x
        
        # make sure item_transforms is callable  
        if not callable(item_transforms):
            raise ValueError(f"The item_transforms must be a callable that accepts a torch.Tensor and returns a torch.Tensor. Found: {item_transforms}")

        # make sure the callable accepts a torch.Tensor and returns a torch.Tensor 
        dummy_tensor = torch.zeros((3, 32, 32)) # a dummy tensor to test the transform
        
        try:
            result = item_transforms(dummy_tensor)
        except Exception as e:
            raise ValueError(f"The item_transforms must accept a torch.Tensor as input. Error: {str(e)}")
        
        if not isinstance(result, torch.Tensor):
            raise ValueError(f"The item_transforms must return a torch.Tensor. Got {type(result).__name__} instead.")

        return item_transforms


    def __init__(self, 
                 root: P,
                 image_transforms: List,
                 item_transforms: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 image_extensions: Optional[Union[List[str], Tuple[str]]] = None):
    
        super().__init__()

        # set the `allowed` / `accepted` extensions
        if image_extensions is None:
            image_extensions = dirf.IMAGE_EXTENSIONS
        
        self.image_extensions = image_extensions

        # make sure the dataset can represents an image dataset
        self.root = dirf.process_path(root, 
                                      must_exist=True, # the directory must already exist
                                      dir_ok=True, 
                                      file_ok=False,
                                      # the directory can contain only image files
                                      condition=lambda d: dirf.dir_contains_only_types(d, valid_extensions=self.image_extensions) 
                                      )     

        self.image_transforms = self._set_image_transforms(image_transforms)

        # after setting the image_transforms, it is time to set the item_transforms: applied to the item (torch.Tensor) produced by the image_transforms 
        # one possible use case is to convert the 
        self.item_transforms = self._set_item_transforms(item_transforms)

        # create a path from indices to path samples
        self.idx2path : Dict = {}
        self.data_count :int = None # a variable to save the number of items in the dataset

        # make sure to call the build_index2path method
        self.build_index2path()


    def build_index2path(self):
        counter = 0
        for r, _, files in os.walk(self.root):
            for f in files:
                file_path = os.path.join(r, f)
                self.idx2path[counter] = file_path 
                counter += 1
        # at this point every number maps to a sample path
        self.data_count = counter    

    def __getitem__(self, index:int) -> torch.Tensor:
        # load the image
        sample = self.load_sample(self.idx2path[index])
        # pass it through the passed transforms
        try:
            compound_tr = tr.Compose(self.image_transforms)
            return compound_tr(sample)
        except Exception:
            # the fall-back approach, call each transformation sequentially
            for t in self.image_transforms:
                sample = t(sample)
        
        sample = self.item_transforms(sample)
        
        return sample

    def __len__(self)->int:
        if self.data_count is None or self.data_count == 0:
            raise ValueError(f"Make sure to set the self.data_count attribute correctly to retrive the size of the dataset in O(1) time !!!")
        return self.data_count



class GenericDsWrapper(ClassificationDsWrapperMixin, Dataset):
    def __init__(self, 
                 root_dir: P,
                 train:bool, 
                 augmentations: Optional[List] = None): 

        self.root_dir = root_dir
        self._ds: Dataset = None
        self._len:int = None

        if augmentations is None:
            # make sure to return tensors and not PIL images
            augmentations = [tr.ToTensor()]

        if not any([isinstance(a, tr.ToTensor) for a in augmentations]):
            augmentations.append(tr.ToTensor()) # append the to.Tensor() transformations at the end            

        self.augmentations = augmentations # augmentations to apply to the images
        self.train = train
        self.samples_per_cls_map = {} # initialize the samples_per_cls_map to an empty dictionary

        self.ds_transform = tr.Compose(augmentations)

    def __getitem__(self, index:int):
        if len(self.samples_per_cls_map) > 0:            
            final_index = self._find_final_index(index)
            sample_image:torch.Tensor = self._ds[final_index][0]
        else:
            sample_image:torch.Tensor = self._ds[index][0]   

        return self.ds_transform(sample_image)


    def __len__(self) -> int:
        if self._len is None or self._len <=  0:
            raise ValueError(f"Make sure to set the 'self._len' attribute. The exact number has to be overriden by the child class")

        return self._len



class Food101GenericWrapper(GenericDsWrapper):
    def __init__(self, 
                root_dir: P, 
                augmentations: List,
                train,
                samples_per_cls: Optional[int] = None) -> None:

        super().__init__(
                root_dir=root_dir,
                augmentations=augmentations,
                train=train)

        self._ds = Food101(root=root_dir,     
                         split='train' if train else 'test',
                         transform=None,
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


class ImagenetteGenericWrapper(GenericDsWrapper):
    def __init__(self, 
                root_dir: P, 
                augmentations: List,
                train,
                samples_per_cls: Optional[int] = None) -> None:

        super().__init__(
                root_dir=root_dir,
                augmentations=augmentations,
                train=train)

        # for some reason, setting the download parameter to True raises an error if the directory already exists
        # wrap the self._ds field in a try and catch statment to cover all cases (setting the download argument with whether the directly exists or not is not enough as certain files might be missing...)
        try:
            self._ds = Imagenette(root=self.root_dir,     
                            split='train' if train else 'val', 
                            transform=None,
                            download=True,  
                            size='full')
        except RuntimeError as e:
            if 'dataset not found' not in str(e).lower():
                self._ds = Imagenette(root=self.root_dir,     
                                split='train' if train else 'val',
                                # setting the transform method in the self._ds object leads to significant degradation in performane
                                # setting transform to None, while setting applying the transformation later in the __getitem__ method should lead to much better performance
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

    def __getitem__(self, index) -> torch.Tensor:
        # apply the augmentations on the original image by calling the parent __getitem__ method 
        return self.ds_transform(super().__getitem__(index))



class MnistGenericWrapper(GenericDsWrapper):
    def __init__(self, 
                 root_dir: P, 
                 augmentations: List,
                 train,
                 samples_per_cls: Optional[int] = None):
        
        super().__init__(root_dir=root_dir, augmentations=augmentations, train=train)
        
        self._ds = MNIST(root=root_dir, 
                        train=train, 
                        download=not os.path.exists(root_dir),
                        transform=None
                        )

        if samples_per_cls is not None:
            self._samples_per_cls_map = self._set_samples_per_cls(samples_per_cls)
            self._len = 10 * samples_per_cls
        else:
            self._len = len(self._ds)


    def _set_samples_per_cls(self, samples_per_cls: int):
        # the number of samples per class is 6000 in the training set and 1000 in the test set
        if self.train:
            mapping = {samples_per_cls * i: 6000 * i for i in range(10)}
        else:
            mapping = {samples_per_cls * i: 1000 * i for i in range(10)}         

        return mapping

    