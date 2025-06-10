"""
Tests for the AbstractConceptDataset class using unittest.
"""
import os
import torch
import shutil
import random
import unittest
import tempfile

import numpy as np
import torchvision.transforms as tr

from PIL import Image
from typing import Set, List, Tuple, Optional

from mypt.shortcuts import P
from mypt.code_utils import directories_and_files as dirf
from mypt.data.datasets.selective_image_folder import SelectiveImageFolderDS
from mypt.data.datasets.concepts.abstract_concept_dataset import AbstractConceptDataset

class ToyConceptsDS(AbstractConceptDataset):
    """
    Concrete implementation of AbstractConceptDataset for testing.
    """
    
    def _prepare_labels(self) -> None:
        """
        Generate simple concept labels for testing.
        Each concept label is a tensor with 5 random values.
        """
        # Iterate through all samples in the dataset
        for idx in range(len(self._ds)):
            # Get the sample path
            sample_path = self._ds.idx2path[idx]
                        
            # Generate a concept vector (5 values between 0 and 1)  
            # Use class_label to make concepts class-dependent for testing
            concept_vector = torch.ones(10) * idx

            # Get the path to save the concept label
            concept_label_path = self.get_concept_label_path(sample_path)
            
            # Save the concept label
            torch.save(concept_vector, concept_label_path)


class TestAbstractConceptDataset(unittest.TestCase):
    """
    Test suite for the AbstractConceptDataset class.
    """
    
    def create_toy_folder(      
        self,
        root_path: str,
        class_names: Optional[List[str]] = None,
        samples_per_class: int = 5,
        image_size: Tuple[int, int] = (24, 24),
        seed: int = 42
    ) -> Tuple[str, Set[str]]:
        """
        Create a toy classification dataset folder structure with random images.
        
        Args:
            root_path: Path where to create the dataset
            class_names: List of class names (folders). If None, defaults to ['class1', 'class2', 'class3']
            samples_per_class: Number of samples to create per class
            image_size: Size of the images to create (height, width)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (root_path, set of all filenames)
        """
        np.random.seed(seed)
        
        # Default class names if not provided
        if class_names is None:
            class_names = ['class1', 'class2', 'class3']
        
        # Create root directory if it doesn't exist
        os.makedirs(root_path, exist_ok=True)
        
        all_filenames = set()
        
        # Create class directories and fill with random images
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(root_path, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for sample_idx in range(samples_per_class):
                # Generate random image data
                img_data = np.random.randint(0, 256, (*image_size, 3), dtype=np.uint8)
                img = Image.fromarray(img_data)
                
                # Create filename
                filename = f"{class_name}_sample_{sample_idx}.jpg"
                filepath = os.path.join(class_dir, filename)
                
                # Save image
                img.save(filepath)
                all_filenames.add(filename)
        
        return root_path, all_filenames

    def setUp(self):
        """Set up a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.images_dir = dirf.process_path(os.path.join(self.temp_dir, "images"), dir_ok=True, file_ok=False)
        self.labels_dir =  dirf.process_path(os.path.join(self.temp_dir, "labels"), dir_ok=True, file_ok=False)
        
        # Create the toy dataset
        _, self.all_filenames = self.create_toy_folder(self.images_dir)
        
    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.temp_dir)

    def test_dataset_initialization(self):
        """
        Test that the dataset initializes correctly with the internal SelectiveImageFolderDS.
        """
        # Create the dataset
        transforms = [tr.ToTensor()]

        dataset = ToyConceptsDS(
            root=self.images_dir,
            filenames=self.all_filenames,
            transforms=transforms,
            label_dir=self.labels_dir
        )
        
        # Check that the dataset has the correct structure
        self.assertIsInstance(dataset._ds, SelectiveImageFolderDS)
        self.assertEqual(dataset.label_dir, self.labels_dir)
        self.assertEqual(dataset.label_suffix, 'concept_label')
        
        # Check that the label directory structure was created
        for class_name in dataset.classes:
            class_label_dir = os.path.join(self.labels_dir, class_name)
            self.assertTrue(os.path.exists(class_label_dir), f"Label directory for {class_name} not created")


    def test_concept_label_path(self):
        """
        Test that get_concept_label_path returns the correct path.
        """
        # Create the dataset
        transforms = [tr.ToTensor()]
        dataset = ToyConceptsDS(
            root=self.images_dir,
            filenames=self.all_filenames,
            transforms=transforms,
            label_dir=self.labels_dir
        )
        
        # Get a sample path
        sample_path = dataset._ds.idx2path[0]
        
        # Get the expected concept label path
        class_name = os.path.basename(os.path.dirname(sample_path))
        sample_filename = os.path.splitext(os.path.basename(sample_path))[0]
        expected_path = os.path.join(self.labels_dir, class_name, f"{sample_filename}_concept_label.pt")
        
        # Call the method and compare
        actual_path = dataset.get_concept_label_path(sample_path)
        self.assertEqual(actual_path, expected_path)
        
        # Test with non-absolute path
        with self.assertRaises(ValueError):
            dataset.get_concept_label_path("relative/path/image.jpg")


    def test_prepare_labels_and_getitem(self):
        """
        Test that _prepare_labels creates concept labels and __getitem__ loads them correctly.
        """
        for _ in range(20):
            dataset_filenames = set(random.sample(list(self.all_filenames), random.randint(1, len(self.all_filenames))))
            
            dataset = ToyConceptsDS(
                root=self.images_dir,
                filenames=dataset_filenames,
                transforms=[tr.ToTensor()],
                label_dir=self.labels_dir
            )
            
            # Prepare the labels
            dataset._prepare_labels()
            
            # Check that concept label files were created for all samples
            for idx in range(len(dataset)):
                sample_path = dataset._ds.idx2path[idx]
                concept_label_path = dataset.get_concept_label_path(sample_path)
                self.assertTrue(os.path.exists(concept_label_path), 
                            f"Concept label file not created: {concept_label_path}")
            
            # Test __getitem__
            for idx in range(len(dataset)):
                sample, concept_label, class_label = dataset[idx]
                
                # Check that sample is a tensor
                self.assertIsInstance(sample, torch.Tensor)
                
                # make sure each sample is associated with the correct concept label
                self.assertIsInstance(concept_label, torch.Tensor)
                self.assertEqual(concept_label.shape, (10,))
                self.assertTrue(torch.allclose(concept_label, torch.ones(10) * idx))
                
                # Check that class_label is an integer
                self.assertIsInstance(class_label, int)
                
                # Check that the class label matches the one from the inner dataset
                self.assertEqual(class_label, dataset._ds.idx2class[idx])


    def test_missing_concept_label(self):
        """
        Test that __getitem__ raises FileNotFoundError when a concept label is missing.
        """
        # Create the dataset
        transforms = [tr.ToTensor()]
        dataset = ToyConceptsDS(
            root=self.images_dir,
            filenames=self.all_filenames,
            transforms=transforms,
            label_dir=self.labels_dir
        )
        
        # Don't prepare labels, so they should be missing
        
        # Test __getitem__ raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            dataset[0]

    def test_dataset_properties(self):
        """
        Test that the dataset properties return the correct values.
        """
        # Create the dataset
        transforms = [tr.ToTensor()]
        dataset = ToyConceptsDS(
            root=self.images_dir,
            filenames=self.all_filenames,
            transforms=transforms,
            label_dir=self.labels_dir
        )
        
        # Test the classes property
        self.assertEqual(dataset.classes, dataset._ds.classes)
        
        # Test the class_to_idx property
        self.assertEqual(dataset.class_to_idx, dataset._ds.class_to_idx)
        
        # Test the len method
        self.assertEqual(len(dataset), len(dataset._ds))


if __name__ == '__main__':
    unittest.main() 