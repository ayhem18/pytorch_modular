"""
This script contains an implementation of a selective image folder dataset that only loads images 
from a directory that are in a provided set of file names. This is useful for selective loading
of images in a classification dataset.
"""
from collections import defaultdict
import os
import warnings
import torch
from PIL import Image
import torchvision.transforms as tr
from torch.utils.data import Dataset
from typing import Iterable, Set, Dict, Callable, Optional, List, Tuple

from mypt.shortcuts import P
from mypt.code_utils import directories_and_files as dirf


class SelectiveImageFolderDS(Dataset):
    """
    A dataset class that loads only specific files from an image classification folder structure.
    The folder structure is expected to be:
    root/
        class1/
            img1.jpg
            img2.jpg
            ...
        class2/
            img1.jpg
            ...
        ...
    
    The dataset will only load files whose names are in the provided set of filenames.
    """
    @classmethod
    def load_sample(cls, sample_path: P):
        # make sure the path is absolute
        if not os.path.isabs(sample_path):
            raise ValueError(f"The loader is expecting the sample path to be absolute.\nFound:{sample_path}")

        # load image similar to torchvision's ImageFolder
        with open(sample_path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")


    def _process_filenames(self, filenames: Iterable[str]) -> Set[str]:        
        filenames = set(filenames)
        found_abs_paths = any(os.path.isabs(filename) for filename in filenames)
        if found_abs_paths:
            warnings.warn("The provided set of filenames contains absolute paths. They are converted to relative paths !!!")

        filenames = set([os.path.basename(filename) for filename in filenames])
        return filenames

    def __init__(self, 
                 root: P,
                 filenames: Set[str],
                 transforms: List,
                 is_class_dir: Optional[Callable[[str], bool]] = None,
                 image_extensions: Optional[Tuple[str, ...]] = None
                 ):
        """
        Initialize the selective image folder dataset.
        
        Args:
            root: Path to the root directory containing class subdirectories
            filenames: Set of filenames to include in the dataset
            transforms: List of transformations to apply to images
            is_class_dir: Optional function to determine if a subdirectory is a class directory.
                          Default is to consider all subdirectories as class directories.
            image_extensions: Optional tuple of valid image extensions. Default is to use 
                             dirf.IMAGE_EXTENSIONS
        """
        super().__init__()

        # Set default values
        if image_extensions is None:
            image_extensions = dirf.IMAGE_EXTENSIONS
        
        self.image_extensions = image_extensions

        # Validate root directory
        self.root = dirf.process_path(root, 
                                     must_exist=True, 
                                     dir_ok=True, 
                                     file_ok=False)

        # Store parameters
        if len(filenames) == 0:
            raise ValueError("The provided set of filenames is empty. Please provide a non-empty set of filenames.")
        
        self.filenames = self._process_filenames(filenames)
        self.transforms = transforms
        self.is_class_dir = is_class_dir or (lambda _: True)  # Default: consider all subdirectories as class dirs
        
        # Data structures for O(1) indexing
        self.idx2path: Dict[int, str] = {}  # Maps index to file path
        self.idx2class: Dict[int, int] = {}  # Maps index to class label
        self.filename2idx: Dict[str, int] = {}  # Maps filename to index
        self.classes: List[str] = []  # List of class names
        self.class_to_idx: Dict[str, int] = {}  # Maps class name to class index
        
        # Build the mappings
        self.build_dataset_mappings()
        
    def build_dataset_mappings(self):
        """Build the mappings between indices, files, and classes for O(1) access."""
        # First scan to identify valid classes
        potential_class_dirs = [d for d in os.listdir(self.root) 
                              if os.path.isdir(os.path.join(self.root, d)) and self.is_class_dir(d)]
        
        if len(potential_class_dirs) == 0:
            raise ValueError("No valid class directories found in the dataset. Please check the dataset folder and/or your filter condition.")
        
        # sorting ensure reproducibility
        self.classes = sorted(potential_class_dirs)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.class_to_indices = defaultdict(lambda: [])

        # Validate file existence and uniqueness
        found_files = set()
        
        # Scan the directory structure
        idx = 0
        for class_name in self.classes:
            class_dir = os.path.join(self.root, class_name)
            class_idx = self.class_to_idx[class_name]
                
            # sort the files to ensure reproducibility
            # and filter non-image files.
            class_files_names = sorted([f for f in os.listdir(class_dir) if os.path.splitext(f.lower())[-1] in self.image_extensions])

            for filename in class_files_names:
                # Skip if not in the requested set of filenames
                if filename not in self.filenames:
                    continue
                
                # Check for duplicate filenames
                if filename in found_files:
                    raise ValueError(f"Duplicate filename detected: {filename}")
                
                found_files.add(filename)
                
                # Add to mappings
                file_path = os.path.join(class_dir, filename)
                # the sample index to the sample path
                self.idx2path[idx] = file_path
                # the sample index to the class index
                self.idx2class[idx] = class_idx
                # the sample file name to the sample index
                self.filename2idx[filename] = idx

                # save the indices of a single class in a dictionary                    
                self.class_to_indices[class_idx].append(idx)
                
                idx += 1
        
        # Check if all requested files were found
        missing_files = self.filenames - found_files
        
        if len(missing_files) > 0:
            raise ValueError(f"The following files were not found in the directory: {missing_files}")
            
        # Set the data count
        self.data_count = idx
        
        # Verify we have at least one file
        if self.data_count == 0:
            raise ValueError("No valid files found in the dataset")
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Return the image and its class label for the given index.
        
        Args:
            index: Index of the image to retrieve
            
        Returns:
            Tuple of (image, class_label)
        """
        if index < 0 or index >= self.data_count:
            raise IndexError(f"Index {index} out of range for dataset with {self.data_count} items")
            
        # Load the image
        sample = self.load_sample(self.idx2path[index])
        class_idx = self.idx2class[index]
        
        # Apply transforms
        try:
            compound_tr = tr.Compose(self.transforms)
            sample = compound_tr(sample)
        except Exception:
            # Fallback approach, call each transformation sequentially
            for t in self.transforms:
                sample = t(sample)
        
        return sample, class_idx
    
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return self.data_count
    
    