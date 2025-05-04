"""
Utility functions for combinatorics and caching for the convolutional block design.
"""

import os
import math
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple

import mypt.code_utils.directories_and_files as dirf

# Define the data folder path
_script_dir = os.path.dirname(os.path.abspath(__file__)) 
_current_dir = _script_dir

while 'package_data' not in os.listdir(_current_dir):
    _current_dir = os.path.dirname(_current_dir)

DATA_FOLDER = dirf.process_path(os.path.join(_current_dir, 'package_data', 'conv_block_design'), 
                                dir_ok=True, file_ok=False, must_exist=False)


def compute_log_linear_sequence(input_dim: int, output_dim: int, n: int) -> List[int]:
    """
    Compute a sequence of n integers that progress linearly on a log2 scale
    between input_dim and output_dim.
    
    If log2(max_dim) - log2(min_dim) < 1, use a simple linear progression instead.
    
    Args:
        input_dim: Starting dimension
        output_dim: Ending dimension
        n: Number of elements in the sequence (including input_dim and output_dim)
        
    Returns:
        List of n integers forming a progression from input_dim to output_dim
    """
    if n < 2:
        raise ValueError(f"Number of elements must be at least 2, got {n}")
    
    if n == 2:
        return [input_dim, output_dim]
    
    min_dim, max_dim = min(input_dim, output_dim), max(input_dim, output_dim)
    
    if min_dim <= 0:
        raise ValueError(f"Input dimension must be positive, got {min_dim}")
    
    log_min = math.log2(min_dim)
    log_max = math.log2(max_dim)
    
    # Determine if we should use log or linear progression
    if log_max - log_min >= 1:
        # Use logarithmic progression
        if input_dim < output_dim:
            # Increasing sequence
            log_sequence = np.linspace(math.log2(input_dim), math.log2(output_dim), n)
            sequence = [int(round(2**x)) for x in log_sequence]
        else:
            # Decreasing sequence
            log_sequence = np.linspace(math.log2(input_dim), math.log2(output_dim), n)
            sequence = [int(round(2**x)) for x in log_sequence]
    else:
        # Use linear progression
        sequence = [int(round(x)) for x in np.linspace(input_dim, output_dim, n)]
    
    # Ensure boundary values are exact
    sequence[0] = input_dim
    sequence[-1] = output_dim
    
    # Handle potential duplicates by slightly adjusting values
    result = [sequence[0]]
    for i in range(1, len(sequence)-1):
        if sequence[i] == result[-1]:
            # Adjust duplicate value
            if input_dim < output_dim:
                sequence[i] += 1
            else:
                sequence[i] -= 1
        result.append(sequence[i])
    result.append(sequence[-1])
    
    return result


def get_combinations_with_replacement(elements: List[int], n: int, reverse:bool, memo: Optional[Dict] = None) -> List[Tuple[int, ...]]:
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

    res_n_minus_1 = get_combinations_with_replacement(elements, n - 1, reverse, memo) 

    res_n = []

    for e in elements: 
        add_e = [r + (e,) for r in res_n_minus_1]
        res_n.extend(add_e)

    res_n = list(set(tuple(sorted(l, reverse=reverse)) for l in res_n))
    memo[n] = res_n
    return res_n


def get_combs_with_rep_range(min_n: int, max_n: int, elements: List[int], reverse: bool) -> Dict[int, List[Tuple[int, ...]]]:
    """
    Get all possible combinations with replacement for a range of sizes.
    
    Args:
        min_n: Minimum number of elements in each combination
        max_n: Maximum number of elements in each combination
        elements: List of elements to create combinations from
        reverse: Whether to reverse the order of the elements
    Returns:
        Dictionary mapping size to list of combinations of that size
    """
    min_n, max_n = sorted([min_n, max_n])

    memo = {}

    for i in range(min_n, max_n + 1):
        get_combinations_with_replacement(elements, i, reverse, memo)

    items = list(memo.items())
    for k, v in items:
        if k < min_n or k > max_n:
            del memo[k] 

    return memo


def get_possible_kernel_combs(min_n: int, max_n: int, max_kernel_size: int, min_kernel_size: int, reverse: bool) -> List[List[int]]:
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
    if reverse:
        cache_filename = f"{min_n}_{max_n}_{min_kernel_size}_{max_kernel_size}_reverse.pkl"
    else:
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
    
    res_dict = get_combs_with_rep_range(min_n, max_n, ks, reverse)

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