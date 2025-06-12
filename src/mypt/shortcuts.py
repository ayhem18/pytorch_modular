"""
This script contains the mapping between string shortcuts and different objects that can be used across diverse scenarios
"""

from pathlib import Path
from typing import Dict, List, Union

from .distances.MMD import GaussianMMD
from .similarities.cosineSim import CosineSim
from .distances.euclidean import inter_euclidean_distances


str2distance = {
    "cosine_sim": CosineSim,
    "euclidean": inter_euclidean_distances,
    "mmd": GaussianMMD
}


P = Union[Path, str]

CONCEPTS_TYPE = Union[P, Dict[str, List[str]]]
