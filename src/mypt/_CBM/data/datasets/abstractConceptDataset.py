"""
This script contains the parent class of different Dataset objects to efficiently load data to Concept Bottleneck Models
"""

import os
import torch
import shutil

import torchvision.transforms as tr

from torch.utils.data import Dataset
from typing import Union, List, Dict, Tuple
from pathlib import Path
from PIL import Image
from abc import ABC, abstractclassmethod

from ....code_utilities import directories_and_files as dirf

class AbstractConceptDataset(ABC, Dataset):
    """This class implements the shared functionality between all concrete dataset objects
    1. loading samples: given a path to an image, return a PIL object 
    2. find classes: creating a consistent (across systems) mapping between the class names and numerical indices 
    3. _sample_to_concept_label: Concepts labels are precalculated and saved in the directory with the samples: for efficient training, knowing the path
    to the sample should allow us to find the path to its concept label in O(1) time
    4. idx2path: map an index to a sample path in O(1) time: the function __get__item(self, index) must be fast for efficient 
    """
    concept_label_ending = 'concept_label'

    @classmethod
    def load_sample(cls, sample_path: Union[str, Path]):
        # make sure the path is absolute
        if not os.path.isabs(sample_path):
            raise ValueError(f"The loader is expecting the sample path to be absolute.\nFound:{sample_path}")

        # this code is copied from the DatasetFolder python file
        with open(sample_path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def find_classes(self) -> Tuple[List[str], Dict[str, int]]:
        """
        This function iterates through the 'self.root' directory returning a list of classes
        and mapping each class to a number

        Returns: the list of classes and a string-to-index map
        """
        
        index_counter = 0
        cls_index_map = {}
        classes = []

        # sort the folder names alphabetically to ensure reproducibility across different systems
        folder_names = sorted(os.listdir(self.root))
        for cls in folder_names:
            # make sure to ignore folders that end with the concept label ending
            if not cls.endswith(self.concept_label_ending):
                cls_index_map[cls] = index_counter
                index_counter += 1
                classes.append(cls)
        return classes, cls_index_map

    def _sample_to_concept_label(self, sample_path: str, cls_label: str = None) -> str:
        """
        Each sample is associated with an embedding representing the distance between the picture
        and the given concepts. Such embedding is saved in the disk to avoid going through the inference process
        each time during training.

        Args:
            sample_path: the path to the sample
            cls_label: the label of the given sample. passed for debugging purposes

        Returns: a path where the corresponding label tensor should be saved

        """

        # make sure the path to the sample is absolute
        if not os.path.isabs(sample_path):
            raise ValueError(f"The path is expected to be absolute. Found: {sample_path}")

        extracted_label = os.path.basename(Path(sample_path).parent)

        # this is used mainly for debugging purposes
        if cls_label is not None and extracted_label != cls_label:
            raise ValueError(f"expected the class labels to match. Given: {cls_label}, Retrieved: {extracted_label}")

        # extract the name of the file
        sample_name, _ = os.path.splitext(os.path.basename(sample_path))
        # build the path to the tensor label
        tensor_path = os.path.join(self.root, f'{extracted_label}_{self.concept_label_ending}', f'{sample_name}.pt')
        return tensor_path

    @abstractclassmethod
    def _prepare_labels(self,
                        batch_size: int, 
                        debug: bool = False) -> None:
        pass

    def _cls_to_range(self) -> Dict[str, Tuple[int, int]]:
        """
        This method builds the tools needed to efficiently map a numerical index to a unique sample absolute path
        """
        # build a mapping from classes to range of indices
        range_min = 0
        cls_range_map = {}
        
        # the main idea is to map the class name to a range [a, a + number of samples of that class - 1], where 'a' is the
        # total number of samples encountered so far.
        for cls in self.classes:
            folder_files = len(os.listdir(os.path.join(self.root, cls)))
            cls_range_map[cls] = (range_min, range_min + folder_files - 1)
            range_min += folder_files

        return cls_range_map

    def idx2path(self, index: int) -> Tuple[Union[str, Path], str]:
        # first step is to locate the path to the corresponding sample
        sample_path, sample_cls = None, None
        for cls, indices_range in self.index_to_class.items():
            if indices_range[0] <= index <= indices_range[1]:
                sample_cls = cls
                cls_dir = os.path.join(self.root, cls)
                sample_path = os.path.join(cls_dir, sorted(os.listdir(cls_dir))[index - indices_range[0]])
                break
        
        if sample_path is None or sample_cls is None:
            raise ValueError(f"Make sure each index maps to a sample")
        
        return sample_path, sample_cls

    def __getitem__(self, index: int):
        # extract the path to the sample or the class
        sample_path, sample_cls = self.idx2path(index)

        sample_image = self.load_sample(sample_path)
        # apply the transformation
        sample_image = self.image_transform(sample_image) if self.image_transform is not None else sample_image
        # convert the class as 'str' to an index
        cls_label = self.class_to_idx[sample_cls]

        # retrieve the concept label
        concept_label = torch.load(self._sample_to_concept_label(sample_path))

        return sample_image, concept_label, cls_label

    def __len__(self) -> int:
        if self.data_count == 0:
            raise ValueError(f"Please make sure to update the self.data_count field in the constructor to have the length of the dataset precomputed !!!!. Found: {self.data_count}")
        return self.data_count

    def __init__(self, 
                 root: Union[str, Path],
                 image_transform: None,
                 remove_existing: bool = False,
                 ) -> None:
        
        if image_transform is None:
            # the main idea here is to resize the images to a small size to be able to stack them into a single tensor
            image_transform = tr.Compose([tr.Resize(size=(224, 224)), tr.ToTensor()])

        self.root = dirf.process_path(root,
                                           file_ok=False,
                                           dir_ok=True,
                                           # make sure that all the sub files are indeed directories
                                           condition=lambda x: all([os.path.isdir(os.path.join(x, p))
                                                                    for p in os.listdir(x)]),
                                           error_message=f'The root directory is expected to have only directories as'
                                                         f' inner files.')
        
        self.image_transform = image_transform        
        self.data_count = 0

        # remove existing concept labels if needed
        if remove_existing:
            for inner_dir in os.listdir(root):
                if inner_dir.endswith(self.concept_label_ending):
                    shutil.rmtree(os.path.join(root, inner_dir))

        # build the class to index map following Pytorch API
        self.classes, self.class_to_idx = self.find_classes()

        # build a mapping between classes and the associated range of indices
        self.index_to_class = self._cls_to_range()

        # let's count the number of items in the dataset
        self.data_count = sum([len(os.listdir(os.path.join(self.root, c))) for c in self.classes])

