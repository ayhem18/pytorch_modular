"""
This script contains an abstract base class for datasets that use concept labels 
alongside traditional class labels.
"""
import os
import torch
from abc import ABC, abstractmethod
from typing import Set, Dict, Callable, Optional, List, Tuple, Any

from mypt.shortcuts import P
from mypt.code_utils import directories_and_files as dirf
from mypt.data.datasets.selective_image_folder import SelectiveImageFolderDS

class AbstractConceptDataset(ABC):
    """
    Abstract base class for datasets that use concept labels alongside traditional class labels.
    This class wraps SelectiveImageFolderDS and adds functionality for concept labels.
    
    The concept labels are expected to be stored in a separate directory structure that mirrors
    the class structure of the original dataset, with file names matching the original samples
    plus a suffix.
    """
    
    def __init__(self,
                 root: P,
                 filenames: Set[str],
                 transforms: List,
                 label_dir: P,
                 label_suffix: str = 'concept_label',
                 is_class_dir: Optional[Callable[[str], bool]] = None,
                 image_extensions: Optional[Tuple[str, ...]] = None
                 ):
        """
        Initialize the abstract concept dataset.
        
        Args:
            root: Path to the root directory containing class subdirectories with images
            filenames: Set of filenames to include in the dataset
            transforms: List of transformations to apply to images
            label_dir: Directory where concept labels will be stored/loaded
            label_suffix: Suffix to add to the sample filename when looking for concept labels
            is_class_dir: Optional function to determine if a subdirectory is a class directory
            image_extensions: Optional tuple of valid image extensions
        """
        # Create the inner SelectiveImageFolderDS instance
        self._ds = SelectiveImageFolderDS(
            root=root,
            filenames=filenames,
            transforms=transforms,
            is_class_dir=is_class_dir,
            image_extensions=image_extensions
        )
        

        # Store the label directory and suffix
        self.label_dir = dirf.process_path(label_dir,
                                         must_exist=False,  # We'll create it if it doesn't exist
                                         dir_ok=True,
                                         file_ok=False)
        
        self.label_suffix = label_suffix
        
        # Create the label directory if it doesn't exist
        if not os.path.exists(self.label_dir):
            os.makedirs(self.label_dir, exist_ok=True)
            
        # Create class subdirectories in the label directory
        for class_name in self._ds.classes:
            class_label_dir = os.path.join(self.label_dir, class_name)
            if not os.path.exists(class_label_dir):
                os.makedirs(class_label_dir, exist_ok=True)
        
    @abstractmethod
    def _prepare_labels(self) -> None:
        """
        Abstract method to prepare concept labels for all samples in the dataset.
        This should be implemented by child classes to generate concept labels
        and save them to the label_dir with the appropriate structure.
        """
        pass
    
    def get_concept_label_path(self, sample_path: str) -> str:
        """
        Get the path to the concept label for a given sample.
        
        Args:
            sample_path: Path to the sample image
            
        Returns:
            Path to the corresponding concept label file
        """
        # Make sure the sample path is absolute
        if not os.path.isabs(sample_path):
            raise ValueError(f"The sample path must be absolute. Found: {sample_path}")
            
        # Extract the class name from the sample path   
        # The class is the name of the parent directory of the sample
        class_name = os.path.basename(os.path.dirname(sample_path))
        
        # Extract the sample filename without extension
        sample_filename = os.path.splitext(os.path.basename(sample_path))[0]
        
        # Build the path to the concept label
        concept_label_path = os.path.join(
            self.label_dir,
            class_name,
            f"{sample_filename}_{self.label_suffix}.pt"
        )
        
        return concept_label_path
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a sample from the dataset along with its concept label and class label.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            Tuple of (image, concept_label, class_label)
        """
        # Get the sample and class label from the inner dataset
        sample, class_label = self._ds[index]
        
        # Get the path to the sample
        sample_path = self._ds.idx2path[index]
        
        # Get the concept label path
        concept_label_path = self.get_concept_label_path(sample_path)
        
        # Load the concept label
        try:
            concept_label = torch.load(concept_label_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Concept label not found at {concept_label_path}. "
                f"Make sure to call _prepare_labels() to generate concept labels."
            )
        
        return sample, concept_label, class_label
    
    def __len__(self) -> int:
        """
        Return the number of items in the dataset.
        
        Returns:
            The dataset size
        """
        return len(self._ds)
        
    @property
    def classes(self) -> List[str]:
        """Get the list of class names."""
        return self._ds.classes
    
    @property
    def class_to_idx(self) -> Dict[str, int]:
        """Get the mapping from class names to indices."""
        return self._ds.class_to_idx 