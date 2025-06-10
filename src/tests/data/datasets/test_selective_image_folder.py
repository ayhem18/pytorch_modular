"""
Tests for the SelectiveImageFolderDS class using unittest.
"""
import os
import torch
import shutil
import random
import tempfile
import unittest

import numpy as np

from PIL import Image
import torchvision.transforms as tr
from typing import Set, List, Optional, Tuple, Dict

from mypt.shortcuts import P
from mypt.code_utils import directories_and_files as dirf
from mypt.data.datasets.selective_image_folder import SelectiveImageFolderDS

class TestSelectiveImageFolderDS(unittest.TestCase):
    """
    Test suite for the SelectiveImageFolderDS class.
    """
    
    def create_toy_folder(
        self,
        root_path: str,
        class_names: Optional[List[str]] = None,
        samples_per_class: int = 10,
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
        

        root_path = dirf.process_path(root_path, 
                                      dir_ok=True, 
                                      file_ok=False, 
                                      must_exist=False)
        
        
        all_filenames = set()
        
        # Create class directories and fill with random images
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(root_path, class_name)
            class_dir = dirf.process_path(class_dir, file_ok=False, dir_ok=True, must_exist=False)
            
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
    
    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.temp_dir)

    def test_class_samples_grouped(self):
        """
        Test that samples of the same class are grouped together in the dataset.
        This is important for efficient batch processing and concept datasets.
        """
        # Create toy dataset with multiple classes
        class_names = ['class1', 'class2', 'class3', 'class4', 'class5']
        samples_per_class = 15
        root_path, all_filenames = self.create_toy_folder(
            self.temp_dir, 
            class_names=class_names,
            samples_per_class=samples_per_class
        )
        

        for _ in range(20):
            dataset_filenames = set(random.sample(list(all_filenames), random.randint(1, len(all_filenames))))
            # Initialize the dataset with all filenames
            dataset = SelectiveImageFolderDS(
                root=root_path,
                filenames=dataset_filenames,
                transforms=[tr.ToTensor()]
            )
            
            # Iterate through the dataset and collect class labels
            class_labels = []
            for idx in range(len(dataset)):
                _, class_label = dataset[idx]
                class_labels.append(class_label)
                
                # Verify the class label is correct by checking the file path
                filepath = dataset.idx2path[idx]
                class_dir = os.path.basename(os.path.dirname(filepath))
                expected_class_idx = dataset.class_to_idx[class_dir]
                self.assertEqual(class_label, expected_class_idx, 
                                f"Class label mismatch for idx {idx}: got {class_label}, expected {expected_class_idx}")
            
            # Map each class label to a list of positions where it appears
            class_positions: Dict[int, List[int]] = {}
            for idx, label in enumerate(class_labels):
                if label not in class_positions:
                    class_positions[label] = []
                class_positions[label].append(idx)
            
            # Check that each class's positions form a consecutive sequence
            for label, positions in class_positions.items():
                
                is_consecutive = all(
                    positions[i+1] - positions[i] == 1 
                    for i in range(len(positions)-1)
                )
                
                self.assertTrue(is_consecutive, 
                                f"Positions for class {label} are not consecutive: {positions}")
                
        

    def test_selective_loading(self):
        """
        Test 2: Create a toy folder, create a random set of file names, load each of these file names
        separately. Then create the dataset, iterate through the dataset and save all the elements.
        Make sure that both sets of tensors are the same.
        """
        # Create toy dataset
        root_path, all_filenames = self.create_toy_folder(self.temp_dir)
        
        # Run multiple tests with different random subsets
        for _ in range(20):
            # Select random subset of filenames (at least 30% of files)
            n_files = len(all_filenames)
            subset_size = max(1, int(n_files * (0.3 + 0.5 * np.random.random())))
            subset_filenames = set(np.random.choice(list(all_filenames), subset_size, replace=False))
            
            # Define transforms
            transforms = [tr.ToTensor()]
            
            # Individually load each image
            reference_tensors = {}
            reference_labels = {}
            
            for filename in subset_filenames:
                # Find the file path
                class_name = filename.split('_')[0]  # Extract class from filename
                filepath = os.path.join(root_path, class_name, filename)
                
                # Load the image
                with open(filepath, 'rb') as f:
                    img = Image.open(f).convert('RGB')
                    tensor = tr.ToTensor()(img)
                    reference_tensors[filename] = tensor
                    
                    # Store expected class label based on directory
                    if class_name == 'class1':
                        reference_labels[filename] = 0
                    elif class_name == 'class2':
                        reference_labels[filename] = 1
                    else:  # class3
                        reference_labels[filename] = 2
            
            # Create the dataset
            dataset = SelectiveImageFolderDS(
                root=root_path,
                filenames=subset_filenames,
                transforms=transforms
            )
            
            # Verify dataset size
            self.assertEqual(len(dataset), len(subset_filenames), 
                             f"Dataset size {len(dataset)} doesn't match filenames count {len(subset_filenames)}")
            
            # Iterate through dataset and compare with reference tensors
            dataset_tensors = {}
            dataset_labels = {}
            
            for idx in range(len(dataset)):
                img_tensor, class_label = dataset[idx]
                filename = os.path.basename(dataset.idx2path[idx])
                dataset_tensors[filename] = img_tensor
                dataset_labels[filename] = class_label
            
            # Verify all filenames were loaded
            self.assertEqual(set(dataset_tensors.keys()), subset_filenames, 
                             "Dataset didn't load all expected files")
            
            # Compare tensors and labels
            for filename in subset_filenames:
                ref_tensor = reference_tensors[filename]
                ds_tensor = dataset_tensors[filename]
                
                # Tensors should be equal
                self.assertTrue(torch.allclose(ref_tensor, ds_tensor), f"Tensor mismatch for {filename}")
                
                # Labels should match
                self.assertEqual(dataset_labels[filename], reference_labels[filename], 
                                f"Label mismatch for {filename}")

    def test_transforms_consistency(self):
        """
        Test that images loaded with deterministic transforms match those loaded separately.
        This test verifies that the transformation pipeline works the same way for individual
        loading versus SelectiveImageFolderDS loading.
        """
        # Create toy dataset with specific image size for reproducible transforms
        image_size = (64, 64)
        root_path, all_filenames = self.create_toy_folder(self.temp_dir, image_size=image_size)

        # Define a set of deterministic transforms
        deterministic_transforms = [
            # Resize to smaller dimensions
            tr.Resize((32, 32)),
            # Center crop
            tr.CenterCrop(24),
            # Convert to tensor
            tr.ToTensor(),
            # Fixed normalization values
            tr.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
        ]

        for _ in range(20):        
            # Select a subset of files to test
            subset_size = random.randint(3, len(all_filenames))

            subset_filenames = set(random.sample(list(all_filenames), subset_size))
            
            
            # Individually load and transform each image
            reference_tensors = {}
            reference_labels = {}
            
            for filename in subset_filenames:
                # Find the file path
                class_name = filename.split('_')[0]  # Extract class from filename
                filepath = os.path.join(root_path, class_name, filename)
                
                # Load the image
                with open(filepath, 'rb') as f:
                    img = Image.open(f).convert('RGB')
                    
                    # Apply each transform manually
                    transformed_img = img
                    for transform in deterministic_transforms:
                        transformed_img = transform(transformed_img)
                    
                    reference_tensors[filename] = transformed_img
                    
                    # Store expected class label based on directory
                    if class_name == 'class1':
                        reference_labels[filename] = 0
                    elif class_name == 'class2':
                        reference_labels[filename] = 1
                    else:  # class3
                        reference_labels[filename] = 2
            
            # Create the dataset with the same transforms
            dataset = SelectiveImageFolderDS(
                root=root_path,
                filenames=subset_filenames,
                transforms=deterministic_transforms
            )
            
            # Verify dataset size
            self.assertEqual(len(dataset), len(subset_filenames), 
                            f"Dataset size {len(dataset)} doesn't match filenames count {len(subset_filenames)}")
            
            # Load all images from the dataset
            dataset_tensors = {}
            dataset_labels = {}
            
            for idx in range(len(dataset)):
                img_tensor, class_label = dataset[idx]
                filename = os.path.basename(dataset.idx2path[idx])
                dataset_tensors[filename] = img_tensor
                dataset_labels[filename] = class_label
            
            # Verify all filenames were loaded
            self.assertEqual(set(dataset_tensors.keys()), subset_filenames, 
                            "Dataset didn't load all expected files")
            
            # Compare tensors and labels
            for filename in subset_filenames:
                ref_tensor = reference_tensors[filename]
                ds_tensor = dataset_tensors[filename]
                
                # Shapes should match
                self.assertEqual(ref_tensor.shape, ds_tensor.shape, 
                                f"Shape mismatch for {filename}: {ref_tensor.shape} vs {ds_tensor.shape}")
                
                # Values should be very close or identical
                self.assertTrue(torch.allclose(ref_tensor, ds_tensor), 
                                f"Tensor mismatch for {filename}")
                
                # Labels should match
                self.assertEqual(dataset_labels[filename], reference_labels[filename], 
                                f"Label mismatch for {filename}")

    def test_missing_files(self):
        """
        Test 3: Create a toy folder, pass a set with some filenames that do not exist in the folder.
        Test whether the dataset raises an error.
        """
        # Create toy dataset
        root_path, all_filenames = self.create_toy_folder(self.temp_dir)
        
        # Add non-existent filenames to the set
        filenames_with_missing = all_filenames.copy()
        filenames_with_missing.add("non_existent_file_1.jpg")
        filenames_with_missing.add("non_existent_file_2.jpg")
        
        # Define transforms
        transforms = [tr.ToTensor()]
        
        # Create the dataset - should raise an error
        with self.assertRaises(ValueError) as context:
            SelectiveImageFolderDS(
                root=root_path,
                filenames=filenames_with_missing,
                transforms=transforms
            )
        
        # Verify error message contains the missing filenames
        error_msg = str(context.exception)
        self.assertIn("non_existent_file_1.jpg", error_msg, "Error message should mention missing file")
        self.assertIn("non_existent_file_2.jpg", error_msg, "Error message should mention missing file")

    def test_class_filter(self):
        """
        Test 4: Create a toy folder, and pass a class_condition that always returns false
        (so that no subfolder will be considered a class directory). Make sure the dataset raises an error.
        """
        # Create toy dataset
        root_path, all_filenames = self.create_toy_folder(self.temp_dir)
        
        # Define transforms
        transforms = [tr.ToTensor()]
        

        # Create the dataset - should raise an error due to no valid classes
        with self.assertRaises(ValueError) as context:
            SelectiveImageFolderDS(
                root=root_path,
                filenames=all_filenames,
                transforms=transforms,
                is_class_dir=lambda _: False
            )
        
        # Verify error message is about no valid files
        error_msg = str(context.exception)
        self.assertIn("No valid class directories found in the dataset", error_msg, "Error should be about missing class directories")

    def test_filename_uniqueness(self):
        """
        Test 5: Create a toy folder with duplicate filenames across different classes.
        Verify the dataset raises a duplicate filename error.
        """
        # Create toy dataset with default class names
        root_path, all_filenames = self.create_toy_folder(self.temp_dir)
        
        # Create a duplicate file in a different class
        duplicate_filename = list(all_filenames)[0]  # Get first filename
        src_class = duplicate_filename.split('_')[0]  # Extract source class
        
        # Find target class (different from source)
        target_class = "class2" if src_class != "class2" else "class1"
        
        # Create duplicate
        src_path = os.path.join(root_path, src_class, duplicate_filename)
        dst_path = os.path.join(root_path, target_class, duplicate_filename)
        
        # Copy the file to create a duplicate
        shutil.copy(src_path, dst_path)
        
        # Define transforms
        transforms = [tr.ToTensor()]
        
        # Create the dataset - should raise an error
        with self.assertRaises(ValueError) as context:
            SelectiveImageFolderDS(
                root=root_path,
                filenames=all_filenames,
                transforms=transforms
            )
        
        # Verify error message mentions duplicate filename
        error_msg = str(context.exception)
        self.assertIn("Duplicate filename detected", error_msg, "Error should mention duplicate filename")
        self.assertIn(duplicate_filename, error_msg, "Error should include the duplicate filename")

    def test_empty_set(self):
        """
        Test 6: Create a toy folder, pass an empty set of filenames.
        Verify the dataset raises "No valid files found" error.
        """
        # Create toy dataset
        root_path, all_filenames = self.create_toy_folder(self.temp_dir)
        
        # Define transforms
        transforms = [tr.ToTensor()]
        
        # Create the dataset with empty set - should raise an error
        with self.assertRaises(ValueError) as context:
            SelectiveImageFolderDS(
                root=root_path,
                filenames=set(),  # Empty set
                transforms=transforms
            )
        
        # Verify error message
        error_msg = str(context.exception)
        self.assertIn("The provided set of filenames is empty", error_msg, "Error should be about no valid files")

    def test_transforms(self):
        """
        Test 8: Create a toy folder, pass different transforms.
        Verify transforms are correctly applied to the images.
        """
        # Create toy dataset with specific image size for predictable resizing
        image_size = (100, 100)
        root_path, all_filenames = self.create_toy_folder(self.temp_dir, image_size=image_size)
        
        # Select a subset of files to test
        subset_size = min(5, len(all_filenames))
        subset_filenames = set(list(all_filenames)[:subset_size])
        
        # Test different transforms
        transform_tests = [
            # Resize transform
            {
                "transforms": [tr.Resize((50, 50)), tr.ToTensor()],
                "verification": lambda tensor: tensor.shape == (3, 50, 50)
            },
            # Grayscale transform
            {
                "transforms": [tr.Grayscale(), tr.ToTensor()],
                "verification": lambda tensor: tensor.shape[0] == 1  # Single channel
            },
            # Normalization transform
            {
                "transforms": [tr.ToTensor(), tr.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])],
                "verification": lambda tensor: tensor.min() < 0 and tensor.max() > 0  # Check normalization effect
            },
            # Color jitter transform (tougher to verify, just check it doesn't crash)
            {
                "transforms": [tr.ColorJitter(brightness=0.5), tr.ToTensor()],
                "verification": lambda tensor: tensor.shape == (3, 100, 100)
            },
            # Compose multiple transforms
            {
                "transforms": [tr.Resize((64, 64)), tr.RandomCrop(32), tr.ToTensor()],
                "verification": lambda tensor: tensor.shape == (3, 32, 32)
            }
        ]
        
        for i, test_case in enumerate(transform_tests):
            # Create dataset with the transforms
            dataset = SelectiveImageFolderDS(
                root=root_path,
                filenames=subset_filenames,
                transforms=test_case["transforms"]
            )
            
            # Check all samples
            for idx in range(len(dataset)):
                img_tensor, _ = dataset[idx]
                
                # Verify transform was applied correctly
                self.assertTrue(test_case["verification"](img_tensor), 
                                f"Transform test {i+1} failed for sample {idx}")


if __name__ == '__main__':
    unittest.main() 