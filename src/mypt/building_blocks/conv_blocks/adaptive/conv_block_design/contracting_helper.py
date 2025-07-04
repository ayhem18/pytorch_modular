"""
Helper functions for the convolutional block design.
"""

import numpy as np

from typing import Dict, List, Optional, Tuple, Set

from mypt.building_blocks.conv_blocks.adaptive.conv_block_design.conv_design_utils import (
    get_possible_kernel_combs,
)


_kernel_size_to_index = {3: 0, 5: 1, 7: 2}
kernel_size_to_cost = {3: 0.5, 5: 2, 7: 6}


def _get_conv_representation(kernel_comb: List[int]) -> List[Dict]:
    """
    Get the representation of the kernel combination.
    """
    def _get_conv_rep(ks: int) -> Dict:
        return {"type": "conv", "kernel_size": ks, "stride": 1}

    return [_get_conv_rep(ks) for ks in kernel_comb]


def _get_pool_representation(pool_comb: List[Tuple[int, int]]) -> List[Dict]:
    """
    Get the representation of the pool combination.
    """
    def _get_pool_rep(ks: int, s: int) -> Dict:
        return {"type": "pool", "kernel_size": ks, "stride": s}

    return [_get_pool_rep(ks, s) for ks, s in pool_comb]


def _get_output_dim(input_dim: int, conv_rep: Dict) -> int:
    """
    Get the output dimension of the convolutional block.
    """
    if conv_rep["type"] == "conv":
        ks = conv_rep["kernel_size"]
        return input_dim - ks + 1

    elif conv_rep["type"] == "pool":
        ks, s = conv_rep["kernel_size"], conv_rep["stride"]
        return (input_dim - ks) // s + 1

    else:
        raise ValueError(f"Invalid convolution representation: {conv_rep}")


def _get_input_dim(output_dim: int, conv_rep: Dict) -> Tuple[int, int]:
    """
    Get the input dimension of the convolutional block.
    """
    if conv_rep["type"] == "conv":
        ks = conv_rep["kernel_size"]
        res = output_dim + ks - 1 
        return (res, res)
    
    elif conv_rep["type"] == "pool":
        ks, s = conv_rep["kernel_size"], conv_rep["stride"]
        temp_res = (output_dim - 1) * s + ks 
        return (temp_res, temp_res + 1) 
    else:
        raise ValueError(f"Invalid convolution representation: {conv_rep}")


def get_output_dim(input_dim: int, conv_reps: List[Dict]) -> int:
    """
    Get the output dimension of the convolutional block.
    """
    res = input_dim
    for conv_rep in conv_reps:
        res = _get_output_dim(res, conv_rep)
    return res 


def get_input_dim(output_dim: int, conv_reps: List[Dict]) -> List[int]:
    """
    Get the input dimension of the convolutional block.
    """
    # there should be at most one pool layer and it must be the last one
    pool_count = sum([1 for conv_rep in conv_reps if conv_rep["type"] == "pool"])
    
    if pool_count > 1:
        raise ValueError("There should be at most one pool layer")

    if pool_count == 1:
        if conv_reps[-1]['type'] != "pool":
            raise ValueError("The last layer should be a pool layer")

        # in this case we have 2 possible outputs 
        r1, r2 = _get_input_dim(output_dim, conv_reps[-1])
        odim = [r1, r2]
        conv_reps = conv_reps[:-1]
    else:
        odim = [output_dim]

    res_list = []

    for start_val in odim:
        res = start_val
        # make sure to run in reverse order
        for conv_rep in conv_reps[::-1]:
            res = _get_input_dim(res, conv_rep)[0]
        res_list.append(res)

    return res_list

def _cost_function(block: List[Dict]) -> float:
    """
    The cost of a block would be the number of pool layers...
    """
    return len([conv_rep for conv_rep in block if conv_rep["type"] == "pool"])


def _build_base_cases(output_dim: int, 
                      min_n: int, 
                      max_n: int, 
                      memo_cost: List[List[int]], 
                      memo_block: List[List[List[int]]],
                      pool_layer_params: Tuple[int, int]):

    # the first step is to generate all possible kernel combinations
    kernel_combs = get_possible_kernel_combs(min_n, max_n, max_kernel_size=7, min_kernel_size=3, reverse=True) 

    # build convolutional blocks
    conv_blocks = [_get_conv_representation(kc) for kc in kernel_combs]
    
    # limit the max pooling kernel size to 2
    conv_pool_blocks = [cb + _get_pool_representation([pool_layer_params]) for cb in conv_blocks]

    # at this point we have the possible counts for the no pool case
    # we need to do the same for the pool case
    possible_counts_pool = [(get_input_dim(output_dim, cb), cb, max(ks)) for cb, ks in zip(conv_pool_blocks, kernel_combs)]

    
    # time to populate the memo
    for (counts, cb, max_ks) in possible_counts_pool:
        for c in counts:
            if c - output_dim >= len(memo_cost):
                continue

            cost = _cost_function(cb)
            
            # the block `cb` produces an output of dimension `output_dim` if the input is of dimension `c`
            # the largest kernel size of `cb` is `max_ks`. In other words, it is 

            for ks in _kernel_size_to_index.keys():
                if ks < max_ks:
                    continue

                # if the cost is already computed, update it if the new cost is lower
                if memo_cost[c - output_dim][_kernel_size_to_index[ks]] != -1: 
                    # check if the cost needs to be updated
                    if cost < memo_cost[c - output_dim][_kernel_size_to_index[ks]]:
                        memo_cost[c - output_dim][_kernel_size_to_index[ks]] = cost
                        memo_block[(c, _kernel_size_to_index[ks])] = cb
                else:
                    memo_cost[c - output_dim][_kernel_size_to_index[ks]] = cost
                    memo_block[(c, _kernel_size_to_index[ks])] = cb

    for i in range(3):
        memo_cost[0][i] = 0
        memo_block[(output_dim, i)] = []


def best_conv_block_dp(input_dim: int, 
                output_dim: int, 
                current_max_ks:int,
                min_n: int, max_n: int, 
                memo_cost: List[List[int]], 
                memo_block: Dict,
                pool_layer_params: Tuple[int, int]):
    
    if memo_cost[input_dim - output_dim][_kernel_size_to_index[current_max_ks]] != -1:
        return memo_cost[input_dim - output_dim][_kernel_size_to_index[current_max_ks]], memo_block[(input_dim, _kernel_size_to_index[current_max_ks])]

    # now we actually need to compute stuff
    kernel_combs = get_possible_kernel_combs(min_n=min_n, max_n=max_n, max_kernel_size=current_max_ks, min_kernel_size=3, reverse=True)   

    # build convolutional blocks
    conv_blocks = [_get_conv_representation(kc) for kc in kernel_combs]
    
    # limit the max pooling kernel size to 2
    conv_pool_blocks = [cb + _get_pool_representation([pool_layer_params]) for cb in conv_blocks]

    possible_counts_pool = [(get_output_dim(input_dim, cb), cb, min(ks)) for cb, ks in zip(conv_pool_blocks, kernel_combs)]


    best_cost = float('inf') 
    best_block = None

    for (count, cb, min_ks) in possible_counts_pool:
        if count - output_dim < 0 or count - output_dim > len(memo_cost):
            # this means that applying the block leads to a negative dimension
            continue 


        possible_next_max_ks = [ks for ks in _kernel_size_to_index.keys() if ks <= min_ks]

        for nks in possible_next_max_ks:
            best_conv_block_dp(count, output_dim, nks, min_n, max_n, memo_cost, memo_block, pool_layer_params) 
            # get the cost of the count and its best block
            rec_block = memo_block[(count, _kernel_size_to_index[nks])]
            if rec_block is None:
                continue

            # compute the cost by adding the current block to the recursive block
            new_block = cb + rec_block 
            cost = _cost_function(new_block)    

            if cost < best_cost:
                best_cost = cost
                best_block = new_block


    if best_block is not None:
        # set the best cost and block for the current input_dim - output_dim
        memo_cost[input_dim - output_dim][_kernel_size_to_index[current_max_ks]] = best_cost 
        memo_block[(input_dim, _kernel_size_to_index[current_max_ks])] = best_block 
    
    else:
        memo_cost[input_dim - output_dim][_kernel_size_to_index[current_max_ks]] = None
        memo_block[(input_dim, _kernel_size_to_index[current_max_ks])] = None

    return best_cost, best_block


def _input_validation(input_dim: int, output_dim: int, min_n: int, max_n: int):
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("For expanding_helper, input_dim and output_dim must be positive")

    # the first step is to make sure the output_dim can be reached by applying at least one block
    minimal_block = [{"type": "conv", "kernel_size": 3, "stride": 1}, {"type": "pool", "kernel_size": 2, "stride": 2}]    

    if get_output_dim(input_dim, minimal_block) < output_dim:
        raise ValueError("Applying the minimal block leads to a dimension smaller than the output dimension. The output dimension is too small.")


    if min_n <= 0 or max_n <= 0:
        raise ValueError("For contracting_helper, min_n and max_n must be positive")

    if min_n > max_n:
        raise ValueError("For contracting_helper, min_n must be less than max_n")



def best_conv_block(input_dim: int, 
                    output_dim: int, 
                    min_n: int, 
                    max_n: int, 
                    memo_cost: Optional[List[List[int]]] = None, 
                    memo_block: Optional[Dict] = None,
                    pool_layer_params: Tuple[int, int] = (2, 2)) -> Tuple[List[Dict], int]:
    _input_validation(input_dim, output_dim, min_n, max_n)

    if memo_cost is None:
        memo_cost = [[-1, -1, -1] for _ in range(output_dim, input_dim + 1)]

    if memo_block is None:
        memo_block = {}

    _build_base_cases(output_dim, min_n, max_n, memo_cost, memo_block, pool_layer_params)

    best_conv_block_dp(input_dim, output_dim, 7, min_n, max_n, memo_cost, memo_block, pool_layer_params)

    # find the best block
    best_block = memo_block[(input_dim, 2)] 

    if best_block is None:
        return None, None

    best_cost_index = np.argmin(memo_cost[input_dim - output_dim])

    best_cost = memo_cost[input_dim - output_dim][best_cost_index]
    return best_block, best_cost


def _compute_all_possible_inputs_dp(min_n: int, max_n: int, output_dim: int, res: Set[int], 
                               memo: Set[int], num_blocks: int):
    """
    Recursively compute all possible input dimensions that could result in the given output dimension
    when applying a series of convolutional blocks.
    
    Args:
        min_n: Minimum number of conv layers in a block
        max_n: Maximum number of conv layers in a block
        output_dim: The target output dimension
        res: Set to collect all possible input dimensions
        memo: Set to track dimensions we've already processed
        num_blocks: Number of blocks to consider applying
    """
    # If we've already processed this output_dim, don't repeat the work
    if output_dim in memo:
        return

    # Mark this output_dim as processed and add it to the results
    memo.add(output_dim)
    res.add(output_dim)

    # nothing to be done with 0 blocks
    if num_blocks == 0:
        return
    
    # Get all possible kernel combinations for the given constraints
    kernel_combs = get_possible_kernel_combs(min_n, max_n, max_kernel_size=7, min_kernel_size=3, reverse=True)
    
    # For each kernel combination, create a block representation
    conv_blocks = [_get_conv_representation(kc) for kc in kernel_combs]
    
    # Add pooling to each block (with kernel size 2)
    conv_pool_blocks = [cb + _get_pool_representation([(2, 2)]) for cb in conv_blocks]
    
    # For each possible block, compute the input dimension that would result in output_dim
    for block in conv_pool_blocks:
        potential_inputs = get_input_dim(output_dim, block)

        # For each potential input, continue the recursion
        for input_dim in potential_inputs:
            _compute_all_possible_inputs_dp(min_n, max_n, input_dim, res, memo, num_blocks - 1)


def compute_all_possible_inputs(min_n: int, max_n: int, output_dim: int, num_blocks: int) -> Set[int]:
    """
    Recursively compute all possible input dimensions that could result in the given output dimension
    when applying a series of convolutional blocks.
    """
    res = set()
    memo = set()
    _compute_all_possible_inputs_dp(min_n, max_n, output_dim, res, memo, num_blocks)    
    res.remove(output_dim)
    return res
