"""
This script contains functionalities designed to efficiently load data for Concept Bottleneck Models
"""

import os, torch, itertools, json

import torchvision.transforms as tr

from tqdm import tqdm
from time import sleep
from pathlib import Path
from typing import Union, List, Dict, Iterable, Optional

from mypt.shortcuts import P
from ....code_utils.pytorch_utils import cleanup
from .abstract_concept_dataset import AbstractConceptDataset
from .concepts_generation.Clip_label_generation import ClipLabelGenerator


class BinaryConceptDataset(AbstractConceptDataset):
    def _validate_concepts(self, concepts: Union[Dict[str, List[str]], P]) -> Dict[str, List[str]]:
        """
        This function validates the concepts dictionary.
        """
        
        # handle the case where the 'concepts' are passed as json file
        if isinstance(concepts, (str, Path)):
            with open(concepts, 'r') as f:
                concepts = json.load(f)

        if isinstance(concepts, Dict):
            concepts = list(itertools.chain(*concepts.values()))
        
        return list(set(concepts))

    def __init__(self,
                 root: P,
                 concepts: Union[Dict[str, List[str]], P],
                 filenames: Iterable[str],
                 label_dir: P,
                 top_k: int = 1,
                 label_generation_batch_size: int = 512,
                 transforms: Optional[tr.Compose] = None,
                 remove_existing: bool = True,
                 image_extensions: Optional[List[str]] = None,
                 label_suffix: str = '_concept_label',
                 ):
        """
        Args:
            root: the root directory
            concepts: a list / dictionary of concepts used for all classes
            transforms: transformation applied on a given image
            remove_existing: whether to remove already-existing concept directories
            label_generation_batch_size: the batch size used to generate the concept labels
            label_dir: the directory where the concept labels are stored
            similarity: the similarity metric used to generate the concept labels
            top_k: the number of most similar concepts to be used for thresholding
            label_generation_batch_size: the batch size used to generate the concept labels
            transforms: transformation applied on a given image
            remove_existing: whether to remove already-existing concept directories
            image_extensions: the extensions of the images to be considered
            label_suffix: the suffix of the concept labels
        """
        
        super().__init__(root, 
                         filenames=filenames,
                         transforms=transforms, 
                         label_dir=label_dir,
                         remove_existing=remove_existing,
                         image_extensions=image_extensions,
                         label_suffix=label_suffix)
        
        # validate the concepts
        self.concepts = self._validate_concepts(concepts)

        # create the label generator
        self.clip_generator = ClipLabelGenerator(similarity_as_cosine=False)
        self.concepts_features = self.clip_generator.encode_concepts(concepts=self.concepts)


        self.top_k = top_k 

        self.concepts_thresholds: torch.Tensor = None

        # let's compute the thresholds for each concept
        t_ready = False
        batch_size = label_generation_batch_size
        while not t_ready:
            try:
                self._compute_thresholds(concepts_encoded=self.concepts_features, 
                                        batch_size=batch_size)
                t_ready = True
            except (MemoryError, torch.cuda.OutOfMemoryError):
                # empty the cache
                cleanup()
                sleep(15)
                batch_size = int(batch_size / 1.2)
                # make sure the batch_size is at least 32
                batch_size = max(batch_size, 32)
                
        # having computed the thresholds, we can now generate the concept labels
        _labels_ready = False
        batch_size = label_generation_batch_size
        while not _labels_ready:
            try:
                self._prepare_labels(batch_size=batch_size)
                _labels_ready = True
            except (MemoryError, torch.cuda.OutOfMemoryError):
                # empty the cache
                cleanup()
                sleep(15)
                batch_size = int(batch_size / 1.2)
                # make sure the batch_size is at least 32
                batch_size = max(batch_size, 32)
            
            
    def _class_concept_similarities(self, 
                                    cls_name: str,
                                    concepts_encoded: torch.Tensor, 
                                    batch_size: int) -> torch.Tensor :
        """_summary_

        Args:
            cls_name (str): The class name
            concept_feature (torch.Tensor): an encoding of a concept
            batch_size (int): the size of the batch to be used if the samples cannot be load to the memory at once

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: the similarities between all the samples and the encoding of the given concept: batched or unbatched
        """

        class_indices = self._ds.class_to_indices[self._ds.class_to_idx[cls_name]]
        class_images = [self._ds.idx2path[idx] for idx in class_indices]

        num_concepts, _ = concepts_encoded.shape

        try:
            # try to pass the data at once
            class_concept_sims = self.clip_generator.generate_image_label(images=class_images, 
                                                                          concepts_features=concepts_encoded, 
                                                                          apply_softmax=False)
            if class_concept_sims.shape != (len(class_images), num_concepts): 
                raise ValueError(f"The similarities between the class samples and the concept should be of shape: {(len(class_images),  num_concepts)}. Found: {class_concept_sims.shape}")

            return class_concept_sims

        except (MemoryError, torch.cuda.OutOfMemoryError):
            # empty the cache
            cleanup()

            # iterate with a batch size
            class_concept_sims = torch.cat([
                                            self.clip_generator.generate_image_label(images=class_images[i : i + batch_size], 
                                                                                     concepts_features=concepts_encoded, 
                                                                                     apply_softmax=False) 
                                            for i in range(0, len(class_images), batch_size)
                                            ], dim=0) 
            if class_concept_sims.shape != (len(class_images), num_concepts): 
                raise ValueError(f"The similarities between the class samples and the concept should be of shape: {(len(class_images), num_concepts)}. Found: {class_concept_sims.shape}")

            return class_concept_sims

    def _compute_thresholds(self, 
                            concepts_encoded: torch.Tensor, 
                            batch_size: int) -> torch.Tensor:
        # iterate through the classes
        top_k_similarities = torch.stack([torch.topk(self._class_concept_similarities(cls_name=cls_name, 
                                                                                 concepts_encoded=concepts_encoded, 
                                                                                 batch_size=batch_size), 
                                                                                 k=self.top_k, 
                                                                                 dim=0, 
                                                                                 largest=True)[0]
                                                for cls_name in tqdm(self.classes, desc='class concept similarities for each class')
                                                ], 
                                        dim=0)
        
        # make sure the shape is as expected
        if top_k_similarities.shape != (len(self.classes), self.top_k, concepts_encoded.shape[0]):
            raise ValueError(f"The matrix of similarities between a concept and all classes is of the wrong shape: Expected:"
                             f"{len(self.classes), self.top_k, concepts_encoded.shape[0]}. Found: {top_k_similarities.shape}")

        # torch.min returns a couple with the values and the indices when called using the 'dim' keyword argument
        thresholds=  torch.min(top_k_similarities[:, self.top_k - 1, :], dim=0)[0]
        # make sure to unsqueeze if needed
        thresholds = thresholds.unsqueeze(dim=0) if thresholds.ndim == 1 else thresholds

        if thresholds.shape != (1, concepts_encoded.shape[0]):
            raise ValueError(f"The thresholding has not been performed successfully !! Expected shape: {(1, concepts_encoded.shape[0])} else ")

        self.concepts_thresholds = thresholds
    
    def _generate_concept_labels(self, images: List[Union[str, Path]]) -> torch.Tensor:
        # first make sure to get the similarity between the concepts
        # and the images 
        image_concept_similarities = self.clip_generator.generate_image_label(images=images, 
                                                                              concepts_features=self.concepts_features, 
                                                                              apply_softmax=False)
        
        if image_concept_similarities.shape != (len(images), len(self.concepts)):
            raise ValueError(f"Make sure the similarities between the images and the concepts are calculated correctly." 
                             f"Expected: {(len(images), len(self.concepts))} .Found: {image_concept_similarities.shape}") 

        thresholds = torch.cat([self.concepts_thresholds for _ in range(len(images))], dim=0)
        assert thresholds.shape == image_concept_similarities.shape, "The thresholds and the similarities should be of the exact same shape."
        concept_label = ((image_concept_similarities - thresholds) >= 0).to(torch.float32)         
        return concept_label

    def _prepare_labels(self, batch_size: int) -> None:
        """
        It is important to keep in mind that
        This function iterates through the 'self.root' directory

        """
        # start by freeing up any previously occupied GPU memory
        cleanup()

        for class_name in tqdm(self.classes, desc='concept labels for each class: representation 3'):
            # at this point we know the current folder is a class folder
            class_indices = self._ds.class_to_indices[self._ds.class_to_idx[class_name]]
            class_images = [self._ds.idx2path[idx] for idx in class_indices]

            # create the concept labels in batches
            for i in range(0, len(class_images), batch_size):
                # extract the paths in the current batch
                batch_file_paths = class_images[i:min(i + batch_size, len(class_images))]

                batch_concept_labels_path = self._verify_full_batch_label_generation(batch_file_paths=batch_file_paths)

                if batch_concept_labels_path is None:
                    # this means the labels for this batch have already been computed... no need to repeat the process.
                    continue

                batch_labels = self._generate_concept_labels(batch_file_paths)

                # make sure the labels are indeed binary
                if not torch.all(torch.logical_or(input=(batch_labels == 1), other=(batch_labels == 0))):
                    raise ValueError(f"the concept label should contain the values 1 or 0.")

                if batch_labels.shape != (len(batch_file_paths), len(self.concepts)):
                    raise ValueError(f"The labels are expected to be of the shape: {(len(batch_file_paths), len(self.concepts))}. Found: {batch_labels.shape}")

                for label_tensor, label_file_path in zip(batch_labels, batch_concept_labels_path):
                    # save the concept label
                    torch.save(label_tensor, label_file_path)

        