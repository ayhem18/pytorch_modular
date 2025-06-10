"""
Tests for the various implementations of AbstractConceptDataset.
"""
import os
import torch
import shutil
import unittest
import tempfile
import numpy as np
import torchvision.transforms as tr

from PIL import Image
from typing import Set, List, Tuple, Optional

from mypt.data.datasets.concepts.basic_concept_dataset import BasicConceptDataset
from mypt.data.datasets.concepts.binary_concept_dataset import BinaryConceptDataset
from mypt.data.datasets.concepts.uniform_concept_dataset import UniformConceptDataset



class TestConceptDatasets(unittest.TestCase):
    """
    Test suite for the different implementations of AbstractConceptDataset.
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
        self.images_dir = os.path.join(self.temp_dir, "images")
        self.labels_dir = os.path.join(self.temp_dir, "labels")
        
        # Define class names for our toy dataset
        self.class_names = ['class1', 'class2', 'class3']
        
        # Create the toy dataset with 5 samples per class
        self.root_path, self.all_filenames = self.create_toy_folder(
            self.images_dir,
            class_names=self.class_names,
            samples_per_class=250
        )
        
        # Create concepts for testing
        self.concepts_list = ["color", "shape", "texture", "size", "material"]
        self.concepts_dict = {
            "class1": ["color", "shape", "size"],
            "class2": ["texture", "material", "weight"],
            "class3": ["density", "hardness", "flexibility"]
        }
        self.uniform_concepts_dict = {
            "class1": ["color", "shape", "size"],
            "class2": ["texture", "material", "weight"],
            "class3": ["density", "hardness", "flexibility"]
        }
        
        # Define transforms
        self.transforms = [tr.ToTensor()]
        
    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.temp_dir)

    def test_basic_concept_dataset(self):
        """
        Test the BasicConceptDataset implementation.
        """
        # Initialize the dataset
        dataset = BasicConceptDataset(
            root=self.images_dir,
            concepts=self.concepts_list,
            filenames=self.all_filenames,
            label_dir=self.labels_dir,
            transforms=self.transforms,
            label_generation_batch_size=32,  # Small batch size for testing
            remove_existing=True
        )
        
        # Verify the dataset was initialized correctly
        self.assertEqual(len(dataset), len(self.all_filenames))
        self.assertEqual(set(dataset.concepts), set(self.concepts_list))
        
        # Check that concept label files were created
        for idx in range(len(dataset._ds)):
            sample_path = dataset._ds.idx2path[idx]
            concept_label_path = dataset.get_concept_label_path(sample_path)
            
            # Verify file exists
            self.assertTrue(os.path.exists(concept_label_path), 
                           f"Concept label file not created: {concept_label_path}")
            
            # Verify content is a tensor with the right shape
            concept_label = torch.load(concept_label_path)
            self.assertIsInstance(concept_label, torch.Tensor)
            self.assertEqual(concept_label.shape, (len(self.concepts_list),))

    def test_binary_concept_dataset(self):
        """
        Test the BinaryConceptDataset implementation.
        """
            
        # Initialize the dataset
        dataset = BinaryConceptDataset(
            root=self.images_dir,
            concepts=self.concepts_list,
            filenames=self.all_filenames,
            label_dir=self.labels_dir,
            similarity="cosine",
            top_k=1,
            label_generation_batch_size=32,  # Small batch size for testing
            transforms=self.transforms,
            remove_existing=True
        )
        
        # Verify the dataset was initialized correctly
        self.assertEqual(len(dataset), len(self.all_filenames))
        self.assertEqual(set(dataset.concepts), set(self.concepts_list))
        
        # Check that concept label files were created
        for idx in range(len(dataset._ds)):
            sample_path = dataset._ds.idx2path[idx]
            concept_label_path = dataset.get_concept_label_path(sample_path)
            
            # Verify file exists
            self.assertTrue(os.path.exists(concept_label_path), 
                            f"Concept label file not created: {concept_label_path}")
            
            # Verify content is a binary tensor with the right shape
            concept_label = torch.load(concept_label_path)
            self.assertIsInstance(concept_label, torch.Tensor)
            self.assertEqual(concept_label.shape, (len(self.concepts_list),))
            
            # Check that values are binary (0 or 1)
            unique_values = torch.unique(concept_label)
            for val in unique_values:
                self.assertTrue(val.item() in [0.0, 1.0], 
                                f"Non-binary value found in concept label: {val.item()}")

    def test_uniform_concept_dataset(self):
        """
        Test the UniformConceptDataset implementation.
        """
        # Initialize the dataset
        dataset = UniformConceptDataset(
            root=self.images_dir,
            concepts=self.uniform_concepts_dict,
            filenames=self.all_filenames,
            label_dir=self.labels_dir,
            label_generation_batch_size=32,  # Small batch size for testing
            transforms=self.transforms,
            remove_existing=True
        )
        
        # Verify the dataset was initialized correctly
        self.assertEqual(len(dataset), len(self.all_filenames))
        self.assertEqual(set(dataset.concepts), set(list(self.uniform_concepts_dict.keys())))    
        
        # Check that concept label files were created
        for idx in range(len(dataset._ds)):
            sample_path = dataset._ds.idx2path[idx]
            concept_label_path = dataset.get_concept_label_path(sample_path)
            
            # Get the class for this sample
            class_name = os.path.basename(os.path.dirname(sample_path))
            
            # Verify file exists
            self.assertTrue(os.path.exists(concept_label_path), 
                           f"Concept label file not created: {concept_label_path}")
            
            # Verify content is a tensor with the right shape (class-specific concepts)
            concept_label = torch.load(concept_label_path)
            self.assertIsInstance(concept_label, torch.Tensor)
            self.assertEqual(concept_label.shape, (len(self.uniform_concepts_dict[class_name]),))


if __name__ == '__main__':
    unittest.main() 