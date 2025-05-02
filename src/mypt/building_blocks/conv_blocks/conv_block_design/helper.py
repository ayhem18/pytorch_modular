"""
Helper functions for the convolutional block design.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

import mypt.code_utils.directories_and_files as dirf


_script_dir = os.path.dirname(os.path.abspath(__file__)) 
_current_dir = _script_dir

while 'package_data' not in os.listdir(_current_dir):
    _current_dir = os.path.dirname(_current_dir)

_DATA_FOLDER = dirf.process_path(os.path.join(_current_dir, 'package_data', 'conv_block_design'), dir_ok=True, file_ok=False, must_exist=False)



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

    res_n = list(set(tuple(sorted(l, reverse=True)) for l in res_n))
    memo[n] = res_n
    return res_n


def get_combs_with_rep_range(min_n: int, max_n: int, elements: List[int]) -> Dict[int, List[Tuple[int, ...]]]:
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
    
    # Create a filename for caching - changed to .pkl extension
    cache_filename = f"{min_n}_{max_n}_{min_kernel_size}_{max_kernel_size}.pkl"
    cache_path = os.path.join(_DATA_FOLDER, cache_filename)
    
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
        with open(cache_path, 'wb') as f:
            pickle.dump(res, f)
    except Exception as e:
        print(f"Warning: Could not save kernel combinations to cache: {e}")
    
    return res


def _get_conv_representation(kernel_comb: List[int]) -> str:
    """
    Get the representation of the kernel combination.
    """
    def _get_conv_rep(ks: int) -> Dict:
        return {"type": "conv", "kernel_size": ks, "stride": 1}

    return [_get_conv_rep(ks) for ks in kernel_comb]


def _get_pool_representation(pool_comb: List[Tuple[int, int]]) -> str:
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


_kernel_size_to_index = {3: 0, 5: 1, 7: 2}

kernel_size_to_cost = {3: 0.5, 5: 2, 7: 6}


# def _cost_function(block: List[Dict]) -> float:
#     """
#     Get the cost of the block.
#     """
#     # num_pool = sum([1 for conv_rep in block if conv_rep["type"] == "pool"])
#     # cost_conv = sum([kernel_size_to_cost[conv_rep["kernel_size"]] for conv_rep in block if conv_rep["type"] == "conv"])
#     convs = [conv_rep for conv_rep in block if conv_rep["type"] == "conv"]
    
#     cost_conv = 0

#     for i, conv in enumerate(convs, start=1):
#         cost_conv += conv["kernel_size"] * round(np.sqrt(i).item(), 2)

#     return cost_conv # + cost_pool


def _cost_function(block: List[Dict]) -> float:
    """
    Get the cost of the block.
    """
    # num_pool = sum([1 for conv_rep in block if conv_rep["type"] == "pool"])
    # cost_conv = sum([kernel_size_to_cost[conv_rep["kernel_size"]] for conv_rep in block if conv_rep["type"] == "conv"])
    return len([conv_rep for conv_rep in block if conv_rep["type"] == "pool"])

    #  basically: the best block is the shortest
    # return len(block) 


def _build_base_cases(output_dim: int, min_n: int, max_n: int, memo_cost: List[List[int]], memo_block: List[List[List[int]]]):
    # the first step is to generate all possible kernel combinations
    kernel_combs = get_possible_kernel_combs(min_n, max_n, max_kernel_size=7, min_kernel_size=3) 

    # build convolutional blocks
    conv_blocks = [_get_conv_representation(kc) for kc in kernel_combs]
    
    # limit the max pooling kernel size to 2
    conv_pool_blocks = [cb + _get_pool_representation([(2, 2)]) for cb in conv_blocks]

    # possible_counts_no_pool = [(get_input_dim(output_dim, cb), cb, max(ks)) for cb, ks in zip(conv_blocks, kernel_combs)]

    # at this point we have the possible counts for the no pool case
    # we need to do the same for the pool case
    possible_counts_pool = [(get_input_dim(output_dim, cb), cb, max(ks)) for cb, ks in zip(conv_pool_blocks, kernel_combs)]

    
    # time to populate the memo
    # no pool case
    # for (counts, cb, max_ks) in possible_counts_no_pool:
    #     for c in counts:
    #         if c - output_dim >= len(memo_cost):
    #             continue

    #         cost = _cost_function(cb) 
    #         # if the cost is already computed, update it if the new cost is lower
    #         if memo_cost[c - output_dim][_kernel_size_to_index[max_ks]] != -1: 
    #             # check if the cost needs to be updated
    #             if cost < memo_cost[c - output_dim][_kernel_size_to_index[max_ks]]:
    #                 memo_cost[c - output_dim][_kernel_size_to_index[max_ks]] = cost
    #                 memo_block[(c, _kernel_size_to_index[max_ks])] = cb

    #         else:
    #             memo_cost[c - output_dim][_kernel_size_to_index[max_ks]] = cost
    #             memo_block[(c, _kernel_size_to_index[max_ks])] = cb

    # pool case
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



def dp_function(input_dim: int, 
                output_dim: int, 
                current_max_ks:int,
                min_n: int, max_n: int, 
                memo_cost: List[List[int]], 
                memo_block: Dict):
    
    if memo_cost[input_dim - output_dim][_kernel_size_to_index[current_max_ks]] != -1:
        return memo_cost[input_dim - output_dim][_kernel_size_to_index[current_max_ks]], memo_block[(input_dim, _kernel_size_to_index[current_max_ks])]

    # now we actually need to compute stuff
    kernel_combs = get_possible_kernel_combs(min_n=min_n, max_n=max_n, max_kernel_size=current_max_ks, min_kernel_size=3)   

    # build convolutional blocks
    conv_blocks = [_get_conv_representation(kc) for kc in kernel_combs]
    
    # limit the max pooling kernel size to 2
    conv_pool_blocks = [cb + _get_pool_representation([(2, 2)]) for cb in conv_blocks]

    # possible_counts_no_pool = [(get_output_dim(input_dim, cb), cb, min(ks)) for cb, ks in zip(conv_blocks, kernel_combs)]

    possible_counts_pool = [(get_output_dim(input_dim, cb), cb, min(ks)) for cb, ks in zip(conv_pool_blocks, kernel_combs)]


    best_cost = float('inf') 
    best_block = None

    for (count, cb, min_ks) in possible_counts_pool:
        if count - output_dim < 0 or count - output_dim > len(memo_cost):
            # this means that applying the block leads to a negative dimension
            continue 


        possible_next_max_ks = [ks for ks in _kernel_size_to_index.keys() if ks <= min_ks]

        for nks in possible_next_max_ks:
            dp_function(count, output_dim, nks, min_n, max_n, memo_cost, memo_block) 
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


    # for (count, cb, min_ks) in possible_counts_no_pool:
    #     if count - output_dim < 0 or count - output_dim > len(memo_cost):
    #         # this means that applying the block leads to a negative dimension
    #         continue 

    #     possible_next_max_ks = [ks for ks in _kernel_size_to_index.keys() if ks <= min_ks]

    #     for nks in possible_next_max_ks:
    #         dp_function(count, output_dim, nks, min_n, max_n, memo_cost, memo_block) 

    #         rec_block = memo_block[(count, _kernel_size_to_index[nks])]
    #         if rec_block is None:
    #             continue

    #         # compute the cost by adding the current block to the recursive block
    #         new_block = cb + rec_block 
    #         cost = _cost_function(new_block)    

    #         if cost < best_cost:
    #             best_cost = cost
    #             best_block = new_block

    if best_block is not None:
        # set the best cost and block for the current input_dim - output_dim
        memo_cost[input_dim - output_dim][_kernel_size_to_index[current_max_ks]] = best_cost 
        memo_block[(input_dim, _kernel_size_to_index[current_max_ks])] = best_block 
    
    else:
        memo_cost[input_dim - output_dim][_kernel_size_to_index[current_max_ks]] = None
        memo_block[(input_dim, _kernel_size_to_index[current_max_ks])] = None

    return best_cost, best_block


def main_function(input_dim: int, output_dim: int, min_n: int, max_n: int, 
                  memo_cost: Optional[List[List[int]]] = None, 
                  memo_block: Optional[Dict] = None):
    if memo_cost is None:
        memo_cost = [[-1, -1, -1] for _ in range(output_dim, input_dim + 1)]

    if memo_block is None:
        memo_block = {}

    # the first step is to make sure the output_dim can be reached by applying at least one block

    minimal_block = [{"type": "conv", "kernel_size": 3, "stride": 1}, {"type": "pool", "kernel_size": 2, "stride": 2}]    

    if get_output_dim(input_dim, minimal_block) < output_dim:
        raise ValueError("Applying the minimal block leads to a dimension smaller than the output dimension. The output dimension is too small.")

    _build_base_cases(output_dim, min_n, max_n, memo_cost, memo_block)

    dp_function(input_dim, output_dim, 7, min_n, max_n, memo_cost, memo_block)

    # find the best block
    best_block = memo_block[(input_dim, 2)] 

    if best_block is None:
        return None, None

    best_cost_index = np.argmin(memo_cost[input_dim - output_dim])

    best_cost = memo_cost[input_dim - output_dim][best_cost_index]
    return best_block, best_cost


def main():
    input_dim = 180
    memo_block = {}
    output_dim = 32
    best_block, best_cost = main_function(input_dim=input_dim, output_dim=output_dim, min_n=2, max_n=6, memo_block=memo_block)

    res = input_dim
    for key, v in memo_block.items():
        if input_dim in key:
            res = get_output_dim(input_dim, v)
            print(res)
            print(v)
            # make sure the block is valid: 
            ks = [conv_rep['kernel_size'] for conv_rep in v if conv_rep['type'] == 'conv']
            
            assert res == output_dim, "It seems that one of the blocks is not leading to the correct output dimension"
            assert ks == sorted(ks, reverse=True), "The kernel sizes are not sorted in descending order"


def main2():
    res = get_possible_kernel_combs(min_n=4, max_n=5, max_kernel_size=7, min_kernel_size=7)
    for v in res:
        print(v)


def compute_all_possible_inputs(min_n: int, max_n: int, output_dim: int, res: Set[int], 
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
    kernel_combs = get_possible_kernel_combs(min_n, max_n, max_kernel_size=7, min_kernel_size=3)
    
    # For each kernel combination, create a block representation
    conv_blocks = [_get_conv_representation(kc) for kc in kernel_combs]
    
    # Add pooling to each block (with kernel size 2)
    conv_pool_blocks = [cb + _get_pool_representation([(2, 2)]) for cb in conv_blocks]
    
    # For each possible block, compute the input dimension that would result in output_dim
    for block in conv_pool_blocks:
        potential_inputs = get_input_dim(output_dim, block)

        # For each potential input, continue the recursion
        for input_dim in potential_inputs:
            compute_all_possible_inputs(min_n, max_n, input_dim, res, memo, num_blocks - 1)


if __name__ == "__main__":
    main()

    # for input_dim in range(20, 100):
    #     print("Printing Results for input dimeinsion :", input_dim)

    #     memo_block = {}
    #     output_dim = 1
    #     best_block, best_cost = main_function(input_dim=input_dim, output_dim=output_dim, min_n=1, max_n=6, memo_block=memo_block)

    #     res = input_dim
    #     for key, v in memo_block.items():
    #         if input_dim in key:
    #             res = get_output_dim(input_dim, v)
    #             assert res == output_dim, "It seems that one of the blocks is not leading to the correct output dimension"

    #     print("--------------------------------")

