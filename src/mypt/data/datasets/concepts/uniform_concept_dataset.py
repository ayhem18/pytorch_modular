"""
This script contains functionalities designed to efficiently load data for Concept Bottleneck Models
"""

import os, torch, json
import torchvision.transforms as tr

from tqdm import tqdm
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Union


from mypt.shortcuts import P
from ....code_utils.pytorch_utils import cleanup
from .abstract_concept_dataset import AbstractConceptDataset
from .concepts_generation.Clip_label_generation import ClipLabelGenerator


class UniformConceptDataset(AbstractConceptDataset):

    def _validate_concepts(self, concepts: Union[Dict[str, List[str]], P]) -> Dict[str, List[str]]:
        """
        This function validates the concepts dictionary.
        """
        # the major difference between this class and ConceptDataset class is the concepts constraints
        # each class must be associated with the exact same number of concepts
        if isinstance(concepts, (Path, str)):
            with open(concepts, 'r') as f:
                concepts = json.load(f)
        
        if not isinstance(concepts, Dict): 
            raise TypeError(f"{type(self)} expects concepts to be a dictionary object. Found: {concepts}")

        # make sure each class is associated with the same number of concepts
        num_cs = None
        for k, v in concepts.items():
            if not (isinstance(k, str) and isinstance(v, List)):
                raise TypeError(f"Expecting the class to be string and the concepts to be lists of strings. Found: class {k}, concepts: {v}") 
            num_cs = len(v) if num_cs is None else num_cs
            if num_cs != len(v):
                raise ValueError((f"{type(self)} expects each concept to be associated with the same number of concepts." 
                                 f"Found classes with different number of concepts. {len(list(concepts.items())[0])}: {num_cs}, {k}: {len(v)}"))

        return concepts


    def __init__(self,
                 root: P,
                 concepts: Union[Dict[str, List[str]], P],
                 filenames: Iterable[str],  
                 label_dir: P,  
                 transforms: Union[tr.Compose, List],
                 label_generation_batch_size: int = 512,
                 label_generator: Optional[ClipLabelGenerator] = None,
                 remove_existing: bool = True, 
                 image_extensions: Optional[List[str]] = None,
                 label_suffix: str = '_concept_label',
                 ):
        """
        Args:
            root: the root directory
            concepts: a list / dictionary of concepts used for all classes
            image_transform: transformation applied on a given image
            remove_existing: whether to remove already-existing concept directories
        """
        super().__init__(root=root,
                         filenames=filenames,
                         label_dir=label_dir,
                         transforms=transforms,
                         remove_existing=remove_existing,
                         image_extensions=image_extensions,
                         label_suffix=label_suffix)

        self.concepts = self._validate_concepts(concepts)

        # create the label generator
        self.label_generator = label_generator if label_generator is not None else ClipLabelGenerator()
        
        # same as concepts, concepts_features should be a dictionary mapping each class to the overall embeddings of the 
        self.concepts_features = {}

        for cls, cs in self.concepts.items():
            self.concepts_features[cls] = self.label_generator.encode_concepts(cs)

        _labels_ready = False
        while not _labels_ready:
            try:
                self._prepare_labels(batch_size=label_generation_batch_size)
                _labels_ready = True
            except (MemoryError, torch.cuda.OutOfMemoryError):
                label_generation_batch_size /= 2 

    
    def _prepare_labels(self, batch_size: int) -> None:
        """
        It is important to keep in mind that
        This function iterates through the 'self.root' directory

        """
        # start by freeing up any available GPU memory
        cleanup()

        for class_name in self._ds.classes:
            
            class_indices = self._ds.class_to_indices[self._ds.class_to_idx[class_name]]

            for i in range(0, len(class_indices), batch_size):
                batch_indices = class_indices[i:min(i + batch_size, len(class_indices))]

                batch_file_paths = [self._ds.idx2path[idx] for idx in batch_indices]
            
                batch_concept_labels = super()._verify_full_batch_label_generation(batch_file_paths)

                if batch_concept_labels is None:
                    # this means that these samples have been encountered before: no need to generate concept labels
                    continue 
                
                batch_labels = self.label_generator.generate_image_label(batch_file_paths, self.concepts_features[class_name])

                if batch_labels.shape != (len(batch_file_paths), len(self.concepts_features[class_name])):
                    raise ValueError(f"The labels are expected to be of the shape: {(len(batch_file_paths), len(self.concepts_features[class_name]))}. Found: {batch_labels.shape}")

                for label_tensor, label_file_path in zip(batch_labels, batch_concept_labels):
                    # save the concept label
                    torch.save(label_tensor, label_file_path)

        