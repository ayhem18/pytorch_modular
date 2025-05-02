"""
Utility functions for combinatorics and caching for the convolutional block design.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import mypt.code_utils.directories_and_files as dirf

# Define the data folder path
_script_dir = os.path.dirname(os.path.abspath(__file__)) 
_current_dir = _script_dir

while 'package_data' not in os.listdir(_current_dir):
    _current_dir = os.path.dirname(_current_dir)

DATA_FOLDER = dirf.process_path(os.path.join(_current_dir, 'package_data', 'conv_block_design'), 
                                dir_ok=True, file_ok=False, must_exist=False)


def get_combinations_with_replacement(elements: List[int], n: int, memo: Optional[Dict] = None) -> List[Tuple[int, ...]]:
    """
    Get all combinations with replacement of the elements in the list.
    
    Args:
        elements: List of elements to create combinations from
        n: Number of elements to include in each combination
        memo: Memoization dictionary to avoid recalculating combinations
        
    Returns:
        List of tuples, each representing a combination
    """
    if memo is None:
        memo = {}

    if n == 0:
        return [()]

    if n == 1:
        res = [(e,) for e in elements]
        memo[1] = res
        return res

    if n in memo:
        return memo[n]

    res_n_minus_1 = get_combinations_with_replacement(elements, n - 1, memo) 

    res_n = []

    for e in elements: 
        add_e = [r + (e,) for r in res_n_minus_1]
        res_n.extend(add_e)

    res_n = list(set(tuple(sorted(l, reverse=True)) for l in res_n))
    memo[n] = res_n
    return res_n


def get_combs_with_rep_range(min_n: int, max_n: int, elements: List[int]) -> Dict[int, List[Tuple[int, ...]]]:
    """
    Get all possible combinations with replacement for a range of sizes.
    
    Args:
        min_n: Minimum number of elements in each combination
        max_n: Maximum number of elements in each combination
        elements: List of elements to create combinations from
        
    Returns:
        Dictionary mapping size to list of combinations of that size
    """
    min_n, max_n = sorted([min_n, max_n])

    memo = {}

    for i in range(min_n, max_n + 1):
        get_combinations_with_replacement(elements, i, memo)

    items = list(memo.items())
    for k, v in items:
        if k < min_n or k > max_n:
            del memo[k] 

    return memo


def get_possible_kernel_combs(min_n: int, max_n: int, max_kernel_size: int, min_kernel_size: int) -> List[List[int]]:
    """
    Get all possible kernel combinations for the convolutional block design.
    Results are cached in a file to avoid recomputation for the same parameters.
    
    Args:
        min_n: Minimum number of conv layers
        max_n: Maximum number of conv layers
        max_kernel_size: Maximum kernel size to consider
        min_kernel_size: Minimum kernel size to consider
        
    Returns:
        List of possible kernel size combinations
    """
    min_n, max_n = sorted([min_n, max_n])
    
    # Create a filename for caching
    cache_filename = f"{min_n}_{max_n}_{min_kernel_size}_{max_kernel_size}.pkl"
    cache_path = os.path.join(DATA_FOLDER, cache_filename)
    
    # Check if the cache file exists
    if os.path.exists(cache_path):
        try:
            # Load the cached results using pickle
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load cached kernel combinations: {e}")
            # If loading fails, continue with computation
    
    # Compute the kernel combinations
    ks = [k for k in [3, 5, 7] if min_kernel_size <= k <= max_kernel_size]

    if len(ks) == 0:
        raise ValueError(f"No kernel sizes in the range {min_kernel_size} to {max_kernel_size}") 
    
    res_dict = get_combs_with_rep_range(min_n, max_n, ks)

    res = []
    for _, combs in res_dict.items():
        res.extend(combs)
        
    # Cache the results using pickle
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(res, f)
    except Exception as e:
        print(f"Warning: Could not save kernel combinations to cache: {e}")
    
    return res 