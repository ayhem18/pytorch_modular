"""
This script contains the mapping between string shortcuts and different objects that can be used across diverse scenarios
"""

from typing import Union
from pathlib import Path

from .similarities.cosineSim import CosineSim
from .distances.euclidean import inter_euclidean_distances
from .distances.MMD import GaussianMMD


str2distance = {
    "cosine_sim": CosineSim,
    "euclidean": inter_euclidean_distances,
    "mmd": GaussianMMD
}


P = Union[Path, str]

