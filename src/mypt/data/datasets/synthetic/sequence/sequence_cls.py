import torch
import random
import numpy as np

from typing import Tuple, List
from torch.utils.data import Dataset

__all__ = ["SyntheticSequenceClsDataset"]


class SyntheticSequenceClsDataset(Dataset):
    """Synthetic dataset for binary sequence order classification.

    Each sample is a sequence of vectors drawn from Gaussians whose means
    are strictly *ascending* or *descending* along the sequence positions.

    The task: predict +1 if the underlying means are non-decreasing, 0 if
    non-increasing.
    """

    def __init__(
        self,
        max_len: int = 32,
        num_samples: int = 10_000,
        dim: int = 16,
        seed: int = 0,
        max_mean: float = 100,
        all_same_length: bool = True,
    ) -> None:
        """Parameters
        ----------
        max_len : int
            Maximum sequence length *S*.
        num_samples : int
            Number of samples in the dataset.
        dim : int
            Feature dimension *D* of each token.
        seed : int | None
            Global seed for full reproducibility.  Will seed *python.random*,
            *numpy* and *torch* RNGs locally.
        all_same_length : bool
            If ``True`` every sequence has length *max_len*.
            Otherwise a length *n \in [1, max_len]* is drawn uniformly for
            each sample.
        """
        super().__init__()

        if max_len < 1:
            raise ValueError("max_len must be positive")
        if dim < 1:
            raise ValueError("dim must be positive")
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if max_mean < 2 * max_len:
            raise ValueError("max_mean must be at least twice max_len for clear separation.")

        self.max_len = max_len
        self.num_samples = num_samples
        self.dim = dim
        self.all_same_length = all_same_length
        self.max_mean = max_mean

        # Create independent RNGs so external seeding does not interfere.
        self._py_rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._torch_gen = torch.Generator()

        self._torch_gen.manual_seed(seed)

        # Pre-generate data for full determinism and speed in __getitem__
        self._sequences: List[torch.Tensor] = [None] * self.num_samples
        self._labels: List[int] = [None] * self.num_samples

        for i in range(self.num_samples):
            seq, label = self._generate_sequence()
            self._sequences[i] = seq
            self._labels[i] = label

    # ---------------------------------------------------------------------
    # PyTorch Dataset interface
    # ---------------------------------------------------------------------
    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self._sequences[idx], self._labels[idx]

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _generate_sequence(self) -> Tuple[torch.Tensor, int]:
        """Create one synthetic sample (sequence, label)."""
        # Decide length
        if self.all_same_length:
            length = self.max_len
        else:
            # inclusive range [1, max_len]
            length = self._py_rng.randint(1, self.max_len)

        # Draw *length* distinct integer means from [-max_mean, max_mean]
        mean_population = range(-int(self.max_mean), int(self.max_mean) + 1)
        base_means = self._py_rng.sample(mean_population, k=length)
        base_means = np.array(base_means)

        # Decide order: ascending (+1) or descending (0)
        if self._py_rng.random() < 0.5:
            # non-decreasing → label +1
            means_sorted = np.sort(base_means)
            label = 1
        else:
            # non-increasing → label 0
            means_sorted = np.sort(base_means)[::-1]
            label = 0

        # Draw token vectors: N(mean, 0.5^2 I)
        noise = self._np_rng.normal(loc=0.0, scale=0.5, size=(length, self.dim))
        seq = means_sorted[:, None] + noise  # broadcast means over dim
        seq_tensor = torch.tensor(seq, dtype=torch.float32)
        return seq_tensor, label