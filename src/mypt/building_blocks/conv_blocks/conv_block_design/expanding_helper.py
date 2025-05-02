"""
Helper functions for the transpose convolutional block design.
This module finds transpose convolution blocks that expand dimensions from input_dim to output_dim,
where input_dim is smaller than output_dim.
"""

import os
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

from mypt.building_blocks.conv_blocks.conv_block_design.comb_utils import (
    get_possible_kernel_combs,
    DATA_FOLDER
)

# Mapping of kernel sizes to indices and costs
_kernel_size_to_index = {3: 0, 5: 1, 7: 2}
kernel_size_to_cost = {3: 0.5, 5: 2, 7: 6}


def _get_transpose_conv_representation(kernel_comb: List[int]) -> List[Dict]:
    """
    Get the representation of the kernel combination for transpose convolutions.
    
    Args:
        kernel_comb: List of kernel sizes
        
    Returns:
        List of dicts representing transpose convolution layers
    """
    def _get_tconv_rep(ks: int) -> Dict:
        return {"type": "tconv", "kernel_size": ks, "stride": 2, "padding": (ks - 1) // 2}

    return [_get_tconv_rep(ks) for ks in kernel_comb]


def _get_output_dim(input_dim: int, tconv_rep: Dict) -> int:
    """
    Get the output dimension after applying a transpose convolution layer.
    
    Args:
        input_dim: Input dimension
        tconv_rep: Dict representing a transpose convolution layer
        
    Returns:
        Output dimension
    """
    if tconv_rep["type"] == "tconv":
        # For transpose convolution: output_dim = (input_dim - 1) * stride + kernel_size - 2*padding
        ks = tconv_rep["kernel_size"]
        stride = tconv_rep["stride"]
        padding = tconv_rep["padding"]
        return (input_dim - 1) * stride + ks - 2 * padding

    else:
        raise ValueError(f"Invalid transpose convolution representation: {tconv_rep}")


def _get_input_dim(output_dim: int, tconv_rep: Dict) -> Tuple[int, int]:
    """
    Get the possible input dimensions that would result in the given output dimension
    after applying a transpose convolution layer.
    
    Args:
        output_dim: Target output dimension
        tconv_rep: Dict representing a transpose convolution layer
        
    Returns:
        A tuple of (min_input_dim, max_input_dim+1) representing the range of possible input dimensions
    """
    if tconv_rep["type"] == "tconv":
        ks = tconv_rep["kernel_size"]
        stride = tconv_rep["stride"]
        padding = tconv_rep["padding"]
        
        # For transpose convolution, solving for input_dim:
        # output_dim = (input_dim - 1) * stride + ks - 2*padding
        # input_dim = (output_dim - ks + 2*padding) / stride + 1
        
        input_dim = (output_dim - ks + 2*padding) / stride + 1
        
        # Return only valid integer solutions
        if input_dim.is_integer():
            input_dim = int(input_dim)
            return (input_dim, input_dim + 1)
        else:
            # No valid integer input dimension
            return (0, 0)
    else:
        raise ValueError(f"Invalid transpose convolution representation: {tconv_rep}")


def get_output_dim(input_dim: int, tconv_reps: List[Dict]) -> int:
    """
    Get the output dimension after applying a sequence of transpose convolution layers.
    
    Args:
        input_dim: Input dimension
        tconv_reps: List of dicts representing transpose convolution layers
        
    Returns:
        Output dimension
    """
    res = input_dim
    for tconv_rep in tconv_reps:
        res = _get_output_dim(res, tconv_rep)
    return res


def get_input_dim(output_dim: int, tconv_reps: List[Dict]) -> List[int]:
    """
    Get the possible input dimensions that would result in the given output dimension
    after applying a sequence of transpose convolution layers.
    
    Args:
        output_dim: Target output dimension
        tconv_reps: List of dicts representing transpose convolution layers
        
    Returns:
        List of possible input dimensions
    """
    if len(tconv_reps) == 0:
        return [output_dim]

    possible_values = []

    # First get the possible input dimensions of the sublist
    sub_reps = tconv_reps[1:]
    possible_values_sub = get_input_dim(output_dim, sub_reps)

    for value in possible_values_sub:
        # Apply the first tconv_rep to get the final result
        low_high = _get_input_dim(value, tconv_reps[0])
        if low_high[0] < low_high[1]:  # If there are valid solutions
            possible_values.extend(list(range(low_high[0], low_high[1])))
    
    return possible_values


def _cost_function(block: List[Dict]) -> float:
    """
    Calculate the cost of a block based on kernel sizes.
    
    Args:
        block: List of dicts representing transpose convolution layers
        
    Returns:
        Cost value
    """
    # Sum of kernel size costs
    return sum(kernel_size_to_cost.get(layer["kernel_size"], 0) for layer in block)


def dp_function(input_dim: int, output_dim: int, current_max_ks: int, 
                min_n: int, max_n: int, memo_cost: List[List], memo_block: Dict):
    """
    Dynamic programming function to find the optimal transpose convolution block.
    
    Args:
        input_dim: Input dimension
        output_dim: Target output dimension
        current_max_ks: Maximum kernel size for the current block
        min_n: Minimum number of layers in a block
        max_n: Maximum number of layers in a block
        memo_cost: Memoization table for costs
        memo_block: Memoization table for blocks
    """
    # Check if the solution is already memoized
    if memo_cost[output_dim - input_dim][_kernel_size_to_index[current_max_ks]] != -1:
        return memo_cost[output_dim - input_dim][_kernel_size_to_index[current_max_ks]]

    # Base case: if input_dim equals output_dim, no need for any block
    if input_dim == output_dim:
        memo_cost[0][_kernel_size_to_index[current_max_ks]] = 0
        memo_block[(input_dim, _kernel_size_to_index[current_max_ks])] = []
        return 0

    # If output_dim is less than input_dim, this isn't the expanding case we want
    if output_dim < input_dim:
        memo_cost[output_dim - input_dim][_kernel_size_to_index[current_max_ks]] = None
        memo_block[(input_dim, _kernel_size_to_index[current_max_ks])] = None
        return None

    # Generate kernel combinations with kernel sizes <= current_max_ks
    kernel_combs = get_possible_kernel_combs(min_n, max_n, current_max_ks, 3)
    
    # Filter combinations with decreasing kernel sizes
    valid_kernel_combs = []
    for comb in kernel_combs:
        is_decreasing = True
        for i in range(len(comb) - 1):
            if comb[i] < comb[i + 1]:
                is_decreasing = False
                break
        if is_decreasing:
            valid_kernel_combs.append(comb)
    
    # Build transpose convolution blocks
    tconv_blocks = [_get_transpose_conv_representation(kc) for kc in valid_kernel_combs]
    
    best_cost = float('inf')
    best_block = None

    # Try each block and find the best one
    for block, ks in zip(tconv_blocks, valid_kernel_combs):
        expanded_dim = get_output_dim(input_dim, block)
        
        # If the expanded dimension matches the target, this is a solution
        if expanded_dim == output_dim:
            cost = _cost_function(block)
            if cost < best_cost:
                best_cost = cost
                best_block = block
        # If we've expanded beyond the target, not a valid solution
        elif expanded_dim > output_dim:
            continue
        # If we haven't reached the target, try to expand further with recursive blocks
        else:
            # We'll only use kernel sizes <= the smallest kernel in the current block
            min_ks = min(ks)
            possible_next_max_ks = [k for k in _kernel_size_to_index.keys() if k <= min_ks]
            
            for nks in possible_next_max_ks:
                # Recursively try to expand from the current expanded dimension to the target
                dp_function(expanded_dim, output_dim, nks, min_n, max_n, memo_cost, memo_block)
                rec_cost_idx = output_dim - expanded_dim
                if rec_cost_idx >= len(memo_cost) or rec_cost_idx < 0:
                    continue
                    
                rec_cost = memo_cost[rec_cost_idx][_kernel_size_to_index[nks]]
                if rec_cost is None:
                    continue
                
                rec_block = memo_block.get((expanded_dim, _kernel_size_to_index[nks]))
                if rec_block is None:
                    continue
                
                # Compute the cost of the combined block
                total_cost = _cost_function(block) + rec_cost
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_block = block + rec_block
    
    # Memoize the results
    memo_cost_idx = output_dim - input_dim
    if memo_cost_idx < len(memo_cost):
        if best_block is not None:
            memo_cost[memo_cost_idx][_kernel_size_to_index[current_max_ks]] = best_cost
            memo_block[(input_dim, _kernel_size_to_index[current_max_ks])] = best_block
        else:
            memo_cost[memo_cost_idx][_kernel_size_to_index[current_max_ks]] = None
            memo_block[(input_dim, _kernel_size_to_index[current_max_ks])] = None
    
    return best_cost


def main_function(input_dim: int, output_dim: int, min_n: int, max_n: int,
                  memo_cost: Optional[List[List[int]]] = None,
                  memo_block: Optional[Dict] = None):
    """
    Main function to find the optimal transpose convolution block that expands from input_dim to output_dim.
    
    Args:
        input_dim: Input dimension
        output_dim: Target output dimension
        min_n: Minimum number of layers in a block
        max_n: Maximum number of layers in a block
        memo_cost: Memoization table for costs
        memo_block: Memoization table for blocks
        
    Returns:
        Tuple of (best_block, best_cost)
    """
    if input_dim >= output_dim:
        raise ValueError("For expanding_helper, input_dim must be smaller than output_dim")
    
    # Initialize memoization tables
    if memo_cost is None:
        memo_cost = [[-1, -1, -1] for _ in range(output_dim - input_dim + 1)]
    
    if memo_block is None:
        memo_block = {}
    
    # Try all starting kernel sizes
    for ks in [7, 5, 3]:
        dp_function(input_dim, output_dim, ks, min_n, max_n, memo_cost, memo_block)
    
    # Find the best block across all kernel sizes
    best_block = None
    best_cost = float('inf')
    
    for ks_idx in range(3):
        cost = memo_cost[output_dim - input_dim][ks_idx]
        if cost is not None and cost < best_cost:
            best_cost = cost
            ks = list(_kernel_size_to_index.keys())[ks_idx]
            best_block = memo_block.get((input_dim, ks_idx))
    
    if best_block is None:
        return None, None
    
    return best_block, best_cost


def compute_all_possible_outputs(min_n: int, max_n: int, input_dim: int, res: Set[int],
                                memo: Set[int], num_blocks: int):
    """
    Recursively compute all possible output dimensions that could result from the given input dimension
    when applying a series of transpose convolutional blocks.
    
    Args:
        min_n: Minimum number of transpose conv layers in a block
        max_n: Maximum number of transpose conv layers in a block
        input_dim: The starting input dimension
        res: Set to collect all possible output dimensions
        memo: Set to track dimensions we've already processed
        num_blocks: Number of blocks to consider applying
    """
    # If we've already processed this input_dim, don't repeat the work
    if input_dim in memo:
        return
    
    # Mark this input_dim as processed and add it to the results
    memo.add(input_dim)
    res.add(input_dim)
    
    # Nothing to be done with 0 blocks
    if num_blocks == 0:
        return
    
    # Get all possible kernel combinations for the given constraints
    kernel_combs = get_possible_kernel_combs(min_n, max_n, max_kernel_size=7, min_kernel_size=3)
    
    # For each kernel combination, create a block representation
    tconv_blocks = [_get_transpose_conv_representation(kc) for kc in kernel_combs]
    
    # For each possible block, compute the output dimension
    for block in tconv_blocks:
        output_dim = get_output_dim(input_dim, block)
        
        # Continue the recursion with this output as the new input
        compute_all_possible_outputs(min_n, max_n, output_dim, res, memo, num_blocks - 1) 