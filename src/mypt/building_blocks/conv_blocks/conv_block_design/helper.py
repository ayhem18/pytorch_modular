"""
Helper functions for the convolutional block design.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

import mypt.code_utils.directories_and_files as dirf


_script_dir = os.path.dirname(os.path.abspath(__file__)) 
_current_dir = _script_dir

while 'package_data' not in os.listdir(_current_dir):
    _current_dir = os.path.dirname(_current_dir)

_DATA_FOLDER = dirf.process_path(os.path.join(_current_dir, 'conv_block_design'), dir_ok=True, file_ok=False, must_exist=False)



def get_combinations_with_replacement(elements: List[int], n: int, memo: Optional[Dict] = None) -> List[Tuple[int, ...]]:
    """
    Get all combinations with replacement of the elements in the list.
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

    res_n = list(set(tuple(sorted(l)) for l in res_n))
    memo[n] = res_n
    return res_n


def get_combs_with_rep_range(min_n: int, max_n: int, elements: List[int]) -> Dict[int, List[Tuple[int, ...]]]:
    min_n, max_n = sorted([min_n, max_n])

    memo = {}

    for i in range(min_n, max_n + 1):
        get_combinations_with_replacement(elements, i, memo)

    return memo



def get_possible_kernel_combs(min_n: int, max_n: int, max_kernel_size: int, min_kernel_size: int) -> List[List[int]]:
    """
    Get all possible kernel combinations for the convolutional block design.
    """
    min_n, max_n = sorted([min_n, max_n])

    ks = [k for k in [3, 5, 7] if min_kernel_size <= k <= max_kernel_size]
    if len(ks) == 0:
        raise ValueError(f"No kernel sizes in the range {min_kernel_size} to {max_kernel_size}") 
    
    res_dict = get_combs_with_rep_range(min_n, max_n, ks)

    res = []

    for _, combs in res_dict.items():
        res.extend(combs)

    return res


def _get_conv_representation(kernel_comb: List[int]) -> str:
    """
    Get the representation of the kernel combination.
    """
    def _get_conv_rep(ks: int) -> Dict:
        return {"type": "conv", "kernel_size": ks, "stride": 1}

    return [_get_conv_rep(ks) for ks in kernel_comb]

def _get_pool_representation(pool_comb: List[int]) -> str:
    """
    Get the representation of the pool combination.
    """
    def _get_pool_rep(ks: int) -> Dict:
        return {"type": "pool", "kernel_size": ks, "stride": ks}

    return [_get_pool_rep(ks) for ks in pool_comb]


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
    pool_count = sum(1 for conv_rep in conv_reps if conv_rep["type"] == "pool")
    
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
            res = _get_input_dim(res, conv_rep)
        res_list.append(res)

    return res_list


_kernel_size_to_index = {3: 0, 5: 1, 7: 2}


def _get_cost_function(block: List[Dict]) -> int:
    """
    Get the cost of the block.
    """
    n = len(block)
    return np.mean([conv_rep["kernel_size"] for conv_rep in block]) + np.sqrt(n)



def _build_base_cases(output_dim: int, min_n: int, max_n: int, memo_cost: List[List[int]], memo_block: List[List[List[int]]]):
    # the first step is to generate all possible kernel combinations
    kernel_combs = get_possible_kernel_combs(min_n, max_n, max_kernel_size=7, min_kernel_size=3) 

    # build convolutional blocks
    conv_blocks = [_get_conv_representation(kc) for kc in kernel_combs]
    
    # limit the max pooling kernel size to 2
    conv_pool_blocks = [cb + [_get_pool_representation([2])] for cb in conv_blocks]

    possible_counts_no_pool = [(get_input_dim(output_dim, cb), cb, max(ks)) for cb, ks in zip(conv_blocks, kernel_combs)]

    # at this point we have the possible counts for the no pool case
    # we need to do the same for the pool case
    possible_counts_pool = [(get_input_dim(output_dim, cb), cb, max(ks)) for cb, ks in zip(conv_pool_blocks, kernel_combs)]

    
    # time to populate the memo
    # no pool case
    for (counts, cb, max_ks) in possible_counts_no_pool:
        for c in counts:
            if c - output_dim > len(memo_cost):
                continue

            cost = _get_cost_function(cb) 
            # if the cost is already computed, update it if the new cost is lower
            if memo_cost[c - output_dim][_kernel_size_to_index[max_ks]] != -1: 
                # check if the cost needs to be updated
                if cost < memo_cost[c - output_dim][_kernel_size_to_index[max_ks]]:
                    memo_cost[c - output_dim][_kernel_size_to_index[max_ks]] = cost
                    memo_block[(c, _kernel_size_to_index[max_ks])] = cb

            else:
                memo_cost[c - output_dim][_kernel_size_to_index[max_ks]] = cost
                memo_block[(c, _kernel_size_to_index[max_ks])] = cb

    # pool case
    for (counts, cb, max_ks) in possible_counts_pool:
        for c in counts:
            if c - output_dim > len(memo_cost):
                continue

            cost = _get_cost_function(cb)
            # if the cost is already computed, update it if the new cost is lower
            if memo_cost[c - output_dim][_kernel_size_to_index[max_ks]] != -1: 
                # check if the cost needs to be updated
                if cost < memo_cost[c - output_dim][_kernel_size_to_index[max_ks]]:
                    memo_cost[c - output_dim][_kernel_size_to_index[max_ks]] = cost
                    memo_block[(c, _kernel_size_to_index[max_ks])] = cb
            else:
                memo_cost[c - output_dim][_kernel_size_to_index[max_ks]] = cost
                memo_block[(c, _kernel_size_to_index[max_ks])] = cb



def dp_function(input_dim: int, output_dim: int, 
                current_max_ks:int,
                min_n: int, max_n: int, 
                memo_cost: List[List[int]], 
                memo_block: Dict):
    
    if memo_cost[input_dim - output_dim][_kernel_size_to_index[current_max_ks]] != -1:
        return memo_cost[input_dim - output_dim], memo_block[input_dim - output_dim]

    # now we actually need to compute stuff
    kernel_combs = get_possible_kernel_combs(min_n=min_n, max_n=max_n, max_kernel_size=current_max_ks, min_kernel_size=3)   

    # build convolutional blocks
    conv_blocks = [_get_conv_representation(kc) for kc in kernel_combs]
    
    # limit the max pooling kernel size to 2
    conv_pool_blocks = [cb + [_get_pool_representation([2])] for cb in conv_blocks]

    possible_counts_no_pool = [(get_output_dim(input_dim, cb), cb, min(ks)) for cb, ks in zip(conv_blocks, kernel_combs)]

    # we need to do the same for the pool case
    possible_counts_pool = [(get_output_dim(input_dim, cb), cb, min(ks)) for cb, ks in zip(conv_pool_blocks, kernel_combs)]



    best_cost = float('inf') 
    best_block = None

    # pool case
    for (count, cb, min_ks) in possible_counts_pool:
        if count - output_dim < 0 or count - output_dim > len(memo_cost):
            # this means that applying the block leads to a negative dimension
            continue 


        possible_next_max_ks = [ks for ks in _kernel_size_to_index.keys() if ks <= min_ks]

        for nks in possible_next_max_ks:
            dp_function(count, output_dim, nks, min_n, max_n, memo_cost, memo_block) 
            # get the cost of the count and its best block
            rec_block = memo_block[(count, _kernel_size_to_index[nks])]

            # compute the cost by adding the current block to the recursive block
            new_block = cb + rec_block 
            cost = _get_cost_function(new_block)    

            if cost < best_cost:
                best_cost = cost
                best_block = new_block


    for (count, cb, min_ks) in possible_counts_no_pool:
        if count - output_dim < 0 or count - output_dim > len(memo_cost):
            # this means that applying the block leads to a negative dimension
            continue 

        possible_next_max_ks = [ks for ks in _kernel_size_to_index.keys() if ks <= min_ks]

        for nks in possible_next_max_ks:
            dp_function(count, output_dim, nks, min_n, max_n, memo_cost, memo_block) 

            # get the cost of the count and its best block
            rec_block = memo_block[(count, _kernel_size_to_index[min_ks])]

            # compute the cost by adding the current block to the recursive block
            new_block = cb + rec_block 
            cost = _get_cost_function(new_block)    

            if cost < best_cost:
                best_cost = cost
                best_block = new_block


    # set the best cost and block for the current input_dim - output_dim
    memo_cost[input_dim - output_dim][_kernel_size_to_index[current_max_ks]] = best_cost 
    memo_block[(input_dim, _kernel_size_to_index[current_max_ks])] = best_block

    return best_cost, best_block


def main_function(input_dim: int, output_dim: int, min_n: int, max_n: int, 
                  memo_cost: Optional[List[List[int]]] = None, 
                  memo_block: Optional[Dict] = None):
    if memo_cost is None:
        memo_cost = [[-1, -1, -1] for _ in range(output_dim, input_dim + 1)]

    if memo_block is None:
        memo_block = {}

    _build_base_cases(output_dim, min_n, max_n, memo_cost, memo_block)

    for max_ks in _kernel_size_to_index.keys():
        dp_function(input_dim, output_dim, max_ks, min_n, max_n, memo_cost, memo_block)

    # find the best block
    best_cost_index = np.argmin(memo_cost[input_dim - output_dim])
    best_block = memo_block[(input_dim, best_cost_index)]

    return best_block


