import numpy as np
import torch
import unittest
from tqdm import tqdm
from typing import List, Tuple

from tests.custom_base_test import CustomModuleBaseTest
from mypt.data.datasets.synthetic.sequence.sequence_cls import SyntheticSequenceClsDataset


class TestSyntheticSequenceClsDataset(CustomModuleBaseTest):
    """
    This test class is used to verify certain properties of the SyntheticSequenceClsDataset:

    1. an item of class label 1 has mostly increasing means
    2. an item of class label 0 has mostly decreasing means
    3. the dataset is reproducible when the seed is set
    

    4. beyond descending / ascending order of means, overall order of values: 
    5. basic properties of the dataset: length, dimension, number of samples, etc.
    6. balance of classes (close to 50/50)
    """
    
    """Test class for SyntheticSequenceClsDataset"""

    def setUp(self) -> None:
        """Set up test parameters"""
        self.seed = 42
        self.max_len = 32
        self.num_samples = 1000
        self.dim = 16
        self.max_mean = 200
        
        # Create a dataset with fixed parameters for consistent testing
        self.dataset = SyntheticSequenceClsDataset(
            max_len=self.max_len,
            num_samples=self.num_samples,
            dim=self.dim,
            seed=self.seed,
            max_mean=self.max_mean,
            all_same_length=True
        )
        
    #########################################################
    # Helper functions
    #########################################################

    def _extract_sequence_means(self, sequence: torch.Tensor) -> List[float]:
        """Extract the mean of each position in a sequence"""
        # sequence shape: [seq_len, dim]
        return torch.mean(sequence, dim=1).tolist()
    
    def _count_increasing_pairs(self, values: List[float]) -> int:
        """Count the number of increasing adjacent pairs in a list"""
        return sum(1 for i in range(len(values) - 1) if values[i] < values[i + 1])
    
    def _count_decreasing_pairs(self, values: List[float]) -> int:
        """Count the number of decreasing adjacent pairs in a list"""
        return sum(1 for i in range(len(values) - 1) if values[i] > values[i + 1])
    
    def _check_cross_sequence_comparison(self, sequence: torch.Tensor) -> Tuple[int, int]:
        """
        Check if at least half of values in sequence i are larger/smaller than 
        half of values in sequence i+1
        
        Returns:
            Tuple[int, int]: (count_larger_pairs, count_smaller_pairs)
        """
        seq_len, _ = sequence.shape
        larger_count = 0
        smaller_count = 0
        
        for i in range(seq_len - 1):
            current_seq = sequence[i]
            next_seq = sequence[i + 1]
            
            # Sort the values
            current_sorted = torch.sort(current_seq).values
            next_sorted = torch.sort(next_seq).values
            
            mid_point = len(current_sorted) // 2
            
            # Check if top half of current_seq values are larger than bottom half of next_seq values
            if torch.all(current_sorted[mid_point:] > next_sorted[:mid_point]):
                larger_count += 1
                
            # Check if bottom half of current_seq values are smaller than top half of next_seq values
            if torch.all(current_sorted[:mid_point] < next_sorted[mid_point:]):
                smaller_count += 1
                
        return larger_count, smaller_count

    def _test_increasing_means_for_class_1_one_ds(self, seed: int):
        """Test that class 1 items have mostly increasing means"""
        
        dataset = SyntheticSequenceClsDataset(
            max_len=self.max_len,
            num_samples=self.num_samples,
            dim=self.dim,
            seed=seed,
            max_mean=self.max_mean,
            all_same_length=True
        )

        class_1_indices = [i for i in range(self.num_samples) if dataset[i][1] == 1]
        
        increasing_ratios = [None for _ in class_1_indices]

        for i, idx in enumerate(class_1_indices):
            sequence, _ = dataset[idx]
            means = self._extract_sequence_means(sequence)
            
            if len(means) > 1:
                increasing_pairs = self._count_increasing_pairs(means)
                total_pairs = len(means) - 1
                increasing_ratio = increasing_pairs / total_pairs
                increasing_ratios[i] = increasing_ratio
            else:
                increasing_ratios[i] = 0

        avg_increasing_ratio = sum(increasing_ratios) / len(increasing_ratios) if increasing_ratios else 0
        self.assertGreater(avg_increasing_ratio, 0.98, 
                          f"Class 1 samples should have mostly increasing means, but avg ratio is {avg_increasing_ratio:.2f}")

    def _test_decreasing_means_for_class_0_one_ds(self, seed: int):
        """Test that class 0 items have mostly decreasing means"""
        dataset = SyntheticSequenceClsDataset(
            max_len=self.max_len,
            num_samples=self.num_samples,
            dim=self.dim,
            seed=seed,
            max_mean=self.max_mean,
            all_same_length=True
        )

        class_0_indices = [i for i in range(self.num_samples) if dataset[i][1] == 0]

        decreasing_ratios = [None for _ in class_0_indices]
        
        for i, idx in enumerate(class_0_indices):
            sequence, _ = dataset[idx]
            means = self._extract_sequence_means(sequence)
            
            if len(means) > 1:
                decreasing_pairs = self._count_decreasing_pairs(means)
                total_pairs = len(means) - 1
                decreasing_ratio = decreasing_pairs / total_pairs
                decreasing_ratios[i] = decreasing_ratio
            else:
                decreasing_ratios[i] = 0
            
        avg_decreasing_ratio = sum(decreasing_ratios) / len(decreasing_ratios) if decreasing_ratios else 0
        self.assertGreater(avg_decreasing_ratio, 0.98, 
                          f"Class 0 samples should have mostly decreasing means, but avg ratio is {avg_decreasing_ratio:.2f}")


    def _test_percentile_property_for_seed(self, seed: int):
        """Helper to test the percentile property for a dataset created with a given seed."""
        
        max_len = np.random.randint(10, 30)

        max_mean = np.random.randint(max(max_len * 2 + 10, 50), max(max_len * 2 + 15, 51))

        dataset = SyntheticSequenceClsDataset(
            max_len=max_len,
            num_samples=100,  # Small dataset
            dim=self.dim,
            seed=seed,
            max_mean=max_mean,
            all_same_length=False # Use variable length for a more robust test
        )

        class_1_ratios = []
        class_0_ratios = []

        for sequence, label in dataset:
            seq_len = sequence.size(0)

            if seq_len < 2:
                continue

            num_pairs = seq_len - 1
            correct_pairs = 0

            for i in range(num_pairs):
                if label == 1:
                    # For increasing sequences, 20th percentile of next should be > 80th of current
                    percentile_current = torch.quantile(sequence[i], 0.8)
                    percentile_next = torch.quantile(sequence[i+1], 0.2)
                    # self.assertGreaterEqual(percentile_next, percentile_current)

                    if percentile_next > percentile_current:
                        correct_pairs += 1

                else: # label == 0
                    # For decreasing sequences, 20th percentile of current should be > 80th of next
                    percentile_current = torch.quantile(sequence[i], 0.2)
                    percentile_next = torch.quantile(sequence[i+1], 0.8)
                    # self.assertGreaterEqual(percentile_current, percentile_next)
                    if percentile_current > percentile_next:
                        correct_pairs += 1

            ratio = correct_pairs / num_pairs
            if label == 1:
                class_1_ratios.append(ratio)
            else:
                class_0_ratios.append(ratio)
        
        # Calculate average success ratio for each class
        avg_ratio_1 = sum(class_1_ratios) / len(class_1_ratios) if class_1_ratios else 1.0
        avg_ratio_0 = sum(class_0_ratios) / len(class_0_ratios) if class_0_ratios else 1.0

        # Assert that the property holds for a high percentage of pairs on average
        self.assertGreater(avg_ratio_1, 0.95, f"For seed {seed}, class 1 failed percentile test with avg ratio {avg_ratio_1:.2f}")
        self.assertGreater(avg_ratio_0, 0.95, f"For seed {seed}, class 0 failed percentile test with avg ratio {avg_ratio_0:.2f}")


    #########################################################
    # Tests
    #########################################################

    def test_dataset_basic_properties(self):
        """Test basic properties of the dataset"""
        self.assertEqual(len(self.dataset), self.num_samples)
        
        sequence, label = self.dataset[0]
        self.assertIsInstance(sequence, torch.Tensor)
        self.assertIsInstance(label, int)
        self.assertEqual(sequence.dim(), 2)
        self.assertEqual(sequence.size(1), self.dim)
        
        labels = [self.dataset[i][1] for i in range(100)]
        self.assertTrue(all(label in [0, 1] for label in labels))
        
        class_counts = {0: 0, 1: 0}
        for i in range(self.num_samples):
            _, label = self.dataset[i]
            class_counts[label] += 1
            
        self.assertGreater(class_counts[0] / self.num_samples, 0.45)
        self.assertLess(class_counts[0] / self.num_samples, 0.55)
        self.assertGreater(class_counts[1] / self.num_samples, 0.45)
        self.assertLess(class_counts[1] / self.num_samples, 0.55)


    def test_increasing_means_for_class_1(self):
        """Test that class 1 items have mostly increasing means"""
        for i in tqdm(range(100), desc="Testing increasing means for class 1 with different seeds"):
            self._test_increasing_means_for_class_1_one_ds(seed=i)

    def test_decreasing_means_for_class_0(self):
        """Test that class 0 items have mostly decreasing means"""
        for i in tqdm(range(100), desc="Testing decreasing means for class 0 with different seeds"):
            self._test_decreasing_means_for_class_0_one_ds(seed=i)


    def test_max_len_and_all_same_length_parameters(self):
        """Test `max_len` and `all_same_length` parameters."""
        # Test all_same_length=True
        dataset_same_len = SyntheticSequenceClsDataset(max_len=10, num_samples=50, all_same_length=True)
        lengths = {s.size(0) for s, l in dataset_same_len}
        self.assertEqual(lengths, {10})
        
        # Test all_same_length=False
        dataset_var_len = SyntheticSequenceClsDataset(max_len=20, num_samples=50, all_same_length=False)
        lengths = {s.size(0) for s, l in dataset_var_len}
        self.assertLessEqual(max(lengths), 20)
        self.assertGreater(len(lengths), 1, "Expected multiple sequence lengths")

    def test_seed_reproducibility(self):
        """Test that using the same seed produces the same dataset"""
        dataset1 = SyntheticSequenceClsDataset(seed=123, num_samples=50)
        dataset2 = SyntheticSequenceClsDataset(seed=123, num_samples=50)
        dataset3 = SyntheticSequenceClsDataset(seed=123, num_samples=50)
        

        for i in range(50):
            seq1, label1 = dataset1[i]
            seq2, label2 = dataset2[i]
            seq3, label3 = dataset3[i]
            self.assertEqual(label1, label2)
            self.assertEqual(label1, label3)
            self.assertTrue(torch.allclose(seq1, seq2))
            self.assertTrue(torch.allclose(seq1, seq3))


    def test_cross_sequence_percentile_property(self):
        """
        Tests that for class 1, the 20th percentile of seq[i+1] is > 80th of seq[i],
        and the opposite for class 0, across many random seeds.
        """
        for i in tqdm(range(100), desc="Testing cross-sequence percentile property"):
            self._test_percentile_property_for_seed(seed=i)



if __name__ == '__main__':
    unittest.main()