"""
This script contains functionalities designed to efficiently load data for Concept Bottleneck Models
"""

import os, torch, itertools, json

import torchvision.transforms as tr

from typing import Union, List, Dict
from pathlib import Path
from tqdm import tqdm
from time import sleep

from .abstractConceptDataset import AbstractConceptDataset
from ..Clip_label_generation import ClipLabelGenerator
from ....code_utilities import pytorch_utilities as pu

class BinaryConceptDataset(AbstractConceptDataset):
    def __init__(self,
                 root: Union[str, Path],
                 concepts: Union[Dict[str, List[str]], List[str]],
                 similarity: str,
                 top_k: int = 1,
                 label_generation_batch_size: int = 512,
                 image_transform: tr = None,
                 remove_existing: bool = True):
        """
        Args:
            root: the root directory
            concepts: a list / dictionary of concepts used for all classes
            image_transform: transformation applied on a given image
            remove_existing: whether to remove already-existing concept directories
        """
        super().__init__(root, 
                         image_transform=image_transform, 
                         remove_existing=remove_existing)
        
        # handle the case where the 'concepts' are passed as json file
        if isinstance(concepts, (str, Path)):
            with open(concepts, 'r') as f:
                concepts = json.load(f)

        if isinstance(concepts, Dict):
            concepts = list(itertools.chain(*concepts.values()))

        # remove duplicate elements
        self.concepts = list(set(concepts))

        if similarity not in ['cosine', 'dot']: 
            raise ValueError(f"Please make sure the argument 'similarity' is in {['cosine', 'dot']}. Found: {similarity}")
        self.sim = similarity

        # create the label generator
        self.clip_generator = ClipLabelGenerator(similarity_as_cosine=self.sim == 'cosine')
        self.concepts_features = self.clip_generator.encode_concepts(concepts=self.concepts)

        # save the number of initial folders in 'root'
        initial_num_folders = len(os.listdir(self.root))

        self.top_k = top_k 

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
                pu.cleanup()
                sleep(15)
                batch_size = int(batch_size / 1.2)
                # make sure the batch_size is at least 32
                batch_size = max(batch_size, 32)
                
        _labels_ready = False
        batch_size = label_generation_batch_size
        while not _labels_ready:
            try:
                self._prepare_labels(batch_size=batch_size)
                _labels_ready = True
            except (MemoryError, torch.cuda.OutOfMemoryError):
                # empty the cache
                pu.cleanup()
                sleep(15)
                batch_size = int(batch_size / 1.2)
                # make sure the batch_size is at least 32
                batch_size = max(batch_size, 32)
            
        # at this point the number of directories should have doubled
        assert not remove_existing or len(os.listdir(root)) == 2 * initial_num_folders, \
            "The number of directories is not doubled !!!"


    def _class_concept_similarities(self, cls_name: str,
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
        
        class_path = os.path.join(self.root, cls_name)
        class_images = sorted(os.listdir(class_path))
        class_images = [os.path.join(class_path, i) for i in class_images]

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
            pu.cleanup()

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
        pu.cleanup()

        # extract

        class_names = [f for f in sorted(os.listdir(self.root)) if f in self.classes]

        for folder_name in tqdm(class_names, desc='concept labels for each class: representation 3'):
            # at this point we know the current folder is a class folder
            cls_path = os.path.join(self.root, folder_name)
            # extract the paths to all the images
            image_paths = [os.path.join(self.root, folder_name, file) for file in sorted(os.listdir(cls_path))]
            
            # create the label folder for the current class if needed
            cls_label_folder_path = os.path.join(self.root, f'{folder_name}_{self.concept_label_ending}')
            os.makedirs(cls_label_folder_path, exist_ok=True)

            # create the concept labels in batches
            for i in range(0, len(image_paths), batch_size):
                # extract the paths in the current batch
                batch_file_paths = image_paths[i:i + batch_size]
                # extract the paths to the corresponding concept labels
                batch_concept_labels_path = [self._sample_to_concept_label(sample_path) for sample_path in batch_file_paths]

                # at this point, either all of them should be present or none of them
                concept_labels_present = [os.path.exists(bclp) for bclp in batch_concept_labels_path]

                # we should make sure that either all files in the batch are associated with a concept label or none of them:
                # this ensures reproducibility
                if any(concept_labels_present) and not all(concept_labels_present):
                    raise ValueError(f"Some files in the batch have concept labels and some do not. Please make sure the code is reproducible!!!")
                
                if all(concept_labels_present):
                    # this means that these samples have been encountered before: no need to generate concept labels
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

        