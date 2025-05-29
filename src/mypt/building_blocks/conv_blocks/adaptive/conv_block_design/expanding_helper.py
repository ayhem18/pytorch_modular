"""
Helper functions for the transpose convolutional block design.
This module finds transpose convolution blocks that expand dimensions from input_dim to output_dim,
where input_dim is smaller than output_dim.
"""

from typing import Dict, List, Optional, Set, Tuple

from mypt.building_blocks.conv_blocks.adaptive.conv_block_design.conv_design_utils import (
    get_possible_kernel_combs,
)

# Mapping of kernel sizes to indices and costs
_kernel_size_to_index = {3: 0, 5: 1, 7: 2}
kernel_size_to_cost = {3: 0.5, 5: 2, 7: 6}



def _get_output_dim(input_dim: int, tconv_rep: Dict) -> int:
    """
    Get the output dimension after applying a transpose convolution layer.
    
    Args:
        input_dim: Input dimension
        tconv_rep: Dict representing a transpose convolution layer
        
    Returns:
        Output dimension
    """
    if tconv_rep["type"] !=  "tconv":
        raise ValueError(f"Invalid transpose convolution representation: {tconv_rep}")

    ks = tconv_rep["kernel_size"]   
    stride = tconv_rep["stride"]
    output_padding = tconv_rep["output_padding"]

    res = (input_dim - 1) * stride + ks + output_padding 
    return res

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


def _get_input_dim(output_dim: int, tconv_rep: Dict) -> Optional[int]:
    """
    Get the possible input dimensions that would result in the given output dimension
    after applying a transpose convolution layer.
    
    Args:
        output_dim: Target output dimension
        tconv_rep: Dict representing a transpose convolution layer
        
    Returns:
        A tuple of (min_input_dim, max_input_dim+1) representing the range of possible input dimensions
    """
    if tconv_rep["type"] != "tconv":
        raise ValueError(f"Invalid transpose convolution representation: {tconv_rep}")

    ks = tconv_rep["kernel_size"]
    stride = tconv_rep["stride"]
    output_padding = tconv_rep["output_padding"]

    input_dim = (output_dim - ks - output_padding) / stride + 1

    if input_dim.is_integer():
        input_dim = int(input_dim)
        return input_dim

    return None


def get_input_dim(output_dim: int, tconv_reps: List[Dict]) -> Optional[int]:
    """
    Get the possible input dimensions that would result in the given output dimension
    after applying a sequence of transpose convolution layers.
    
    Args:
        output_dim: Target output dimension
        tconv_reps: List of dicts representing transpose convolution layers
        
    Returns:
        List of possible input dimensions
    """

    res = output_dim

    for tconv_rep in tconv_reps[::-1]:
        res = _get_input_dim(res, tconv_rep)

        if res is None:
            return None 

    return res


def _get_single_transpose_conv_rep(ks: int, stride: int, output_padding: int) -> Dict:
    """
    Get the representation of a single transpose convolution layer.
    """
    return {"type": "tconv", "kernel_size": ks, "stride": stride, "output_padding": output_padding}


def _cost_function(block: List[Dict]) -> float:
    """
    Calculate the cost of a block based on kernel sizes.
    
    Args:
        block: List of dicts representing transpose convolution layers
        
    Returns:
        Cost value
    """ 
    # the number of blocks with stride > 1
    return len([b for b in block if b["stride"] > 1])


def _build_base_cases(input_dim: int, output_dim: int, min_n: int, max_n: int, memo_cost: List[List], memo_block: Dict):
    """
    Build the base cases for the dynamic programming function.
    """
    # Get all possible kernel combinations for the given constraints
    kernel_combs = get_possible_kernel_combs(min_n, max_n, max_kernel_size=7, min_kernel_size=3, reverse=False)

    # each element in kernel_combs is a list (with number of elements between min_n and max_n)
    # for each elements in these lists, add the stride 1 and output padding 0
    tconvs_no_stride = [[_get_single_transpose_conv_rep(ks, 1, 0) for ks in kc] for kc in kernel_combs]
    
    # add a tranpose convolution with stride 2 and output padding 0
    tconvs_op0 = [b + [{"type": "tconv", "kernel_size": 2, "stride": 2, "output_padding": 0}] for b in tconvs_no_stride]

    # add a tranpose convolution with stride 2 and output padding 1
    tconvs_op1 = [b + [{"type": "tconv", "kernel_size": 2, "stride": 2, "output_padding": 1}] for b in tconvs_no_stride]

    tconvs = tconvs_op0 + tconvs_op1

    possible_counts_unfiltered = [get_input_dim(output_dim, tconv) for tconv in tconvs]

    # filter None values and their corresponding tranpose convolution blocks 
    possible_counts = [i for i in possible_counts_unfiltered if i is not None]
    tconvs = [tconv for tconv, i in zip(tconvs, possible_counts_unfiltered) if i is not None]


    # time to populate the memo    
    for count, tconv in zip(possible_counts, tconvs):
        if count < input_dim:
            continue 

        # find the smallest kernel size in the tconv (with stride 1 !!!!)
        min_ks = min([tconv["kernel_size"] for tconv in tconv if tconv["stride"] == 1])

        cost = _cost_function(tconv)

        for ks in _kernel_size_to_index.keys():
            if ks > min_ks:
                continue

            # if the cost is already computed, update it if the new cost is lower
            if memo_cost[count - input_dim][_kernel_size_to_index[ks]] != -1: 
                # check if the cost needs to be updated
                if cost < memo_cost[count - input_dim][_kernel_size_to_index[ks]]:
                    memo_cost[count - input_dim][_kernel_size_to_index[ks]] = cost
                    memo_block[(count, _kernel_size_to_index[ks])] = tconv
            else:
                memo_cost[count - input_dim][_kernel_size_to_index[ks]] = cost
                memo_block[(count, _kernel_size_to_index[ks])] = tconv

    for i in range(3):
        memo_cost[0][i] = 0
        memo_block[(output_dim, i)] = []




def dp_transpose_conv_block(
                input_dim: int, 
                output_dim: int, 
                current_min_ks: int, 
                min_n: int, 
                max_n: int, 
                memo_cost: List[List], 
                memo_block: Dict):
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
    if memo_cost[output_dim - input_dim][_kernel_size_to_index[current_min_ks]] != -1:
        return memo_cost[output_dim - input_dim][_kernel_size_to_index[current_min_ks]]

    if output_dim < input_dim:
        memo_block[(input_dim, _kernel_size_to_index[current_min_ks])] = None
        return None

    # Generate kernel combinations with kernel sizes <= current_max_ks
    kernel_combs = get_possible_kernel_combs(min_n, max_n, 7, current_min_ks, reverse=False)
    
    # each element in kernel_combs is a list (with number of elements between min_n and max_n)
    # for each elements in these lists, add the stride 1 and output padding 0
    tconvs_no_stride = [[_get_single_transpose_conv_rep(ks, 1, 0) for ks in kc] for kc in kernel_combs]
    
    # add a tranpose convolution with stride 2 and output padding 0
    tconvs_op0 = [b + [{"type": "tconv", "kernel_size": 2, "stride": 2, "output_padding": 0}] for b in tconvs_no_stride]

    # add a tranpose convolution with stride 2 and output padding 1
    tconvs_op1 = [b + [{"type": "tconv", "kernel_size": 2, "stride": 2, "output_padding": 1}] for b in tconvs_no_stride]

    tconvs = tconvs_op0 + tconvs_op1


    best_cost = float('inf')
    best_block = None

    kernel_combs_duplicated = kernel_combs + kernel_combs

    assert len(tconvs) == len(kernel_combs_duplicated), "make sure the number of transpose convolution blocks is the same as the number of kernel combinations"

    # Try each block and find the best one
    for block, block_kernel_sizes in zip(tconvs, kernel_combs_duplicated):
        expanded_dim = get_output_dim(input_dim, block)
        
        if expanded_dim > output_dim:
            continue

        max_ks = max(block_kernel_sizes)
        possible_next_max_ks = [k for k in _kernel_size_to_index.keys() if k >= max_ks]
        
        for nks in possible_next_max_ks:
            # Recursively try to expand from the current expanded dimension to the target
            dp_transpose_conv_block(expanded_dim, output_dim, nks, min_n, max_n, memo_cost, memo_block)
        
            rec_block = memo_block.get((expanded_dim, _kernel_size_to_index[nks]))
            
            if rec_block is None:
                continue
            
            # compute the cost of the generated block
            result_block = block + rec_block

            # Compute the cost of the combined block
            total_cost = _cost_function(result_block)
            if total_cost < best_cost:
                best_cost = total_cost
                best_block = result_block
    
    # Memoize the results
    memo_cost_idx = output_dim - input_dim

    if best_block is not None:
        memo_cost[memo_cost_idx][_kernel_size_to_index[current_min_ks]] = best_cost
        memo_block[(input_dim, _kernel_size_to_index[current_min_ks])] = best_block
    else:
        memo_cost[memo_cost_idx][_kernel_size_to_index[current_min_ks]] = None
        memo_block[(input_dim, _kernel_size_to_index[current_min_ks])] = None
    
    return best_cost


def _input_validation(input_dim: int, output_dim: int, min_n: int, max_n: int):
    minimal_block = [
                    {"type": "tconv", "kernel_size": 3, "stride": 1, "output_padding": 0}, 
                    {"type": "tconv", "kernel_size": 3, "stride": 2, "output_padding": 0}
                    ]    

    if get_output_dim(input_dim, minimal_block) > output_dim:
        raise ValueError("Applying the minimal tranpose convolution block leads to a dimension larger than the output dimension. The input dimension is too large.")
        
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("For expanding_helper, input_dim and output_dim must be positive")

    if min_n <= 0 or max_n <= 0:
        raise ValueError("For expanding_helper, min_n and max_n must be positive")

    if min_n > max_n:
        raise ValueError("For expanding_helper, min_n must be less than max_n")


def best_transpose_conv_block(input_dim: int, output_dim: int, min_n: int, max_n: int,
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
    _input_validation(input_dim, output_dim, min_n, max_n)

    # Initialize memoization tables
    if memo_cost is None:
        memo_cost = [[-1, -1, -1] for _ in range(output_dim - input_dim + 1)]
    
    if memo_block is None:
        memo_block = {}

    _build_base_cases(input_dim, output_dim, min_n, max_n, memo_cost, memo_block)

    dp_transpose_conv_block(input_dim, output_dim, 3, min_n, max_n, memo_cost, memo_block)
    
    # Find the best block across all kernel sizes
    best_block = None
    best_cost = float('inf')

    # extract the best block
    best_block = memo_block.get((input_dim, 0))

    if best_block is None:
        return None, None

    best_cost = memo_cost[output_dim - input_dim][0]
    
    return best_block, best_cost



def dp_compute_all_possible_inputs(min_n: int, 
                                max_n: int, 
                                output_dim: int,
                                min_threshold: int, 
                                res: Set[int],
                                res_ks_size: Set[Tuple[int, int]],
                                memo: Set[int], 
                                num_blocks: int):
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
    # If we've already processed this output_dim, don't repeat the work
    if output_dim in memo or output_dim < min_threshold:
        return
    
    # Mark this output_dim as processed and add it to the results
    memo.add(output_dim)
    
    # Nothing to be done with 0 blocks
    if num_blocks == 0:
        return
    
    # find the largest kernel size associated with the output_dim
    max_size =  max([ks for ks in _kernel_size_to_index.keys() if (output_dim, ks) in res_ks_size])

    # Get all possible kernel combinations for the given constraints
    kernel_combs = get_possible_kernel_combs(min_n, max_n, max_kernel_size=max_size, min_kernel_size=3, reverse=False)
    
    # each element in kernel_combs is a list (with number of elements between min_n and max_n)
    # for each elements in these lists, add the stride 1 and output padding 0
    tconvs_no_stride = [[_get_single_transpose_conv_rep(ks, 1, 0) for ks in kc] for kc in kernel_combs]
    
    # add a tranpose convolution with stride 2 and output padding 0
    tconvs_op0 = [b + [{"type": "tconv", "kernel_size": 2, "stride": 2, "output_padding": 0}] for b in tconvs_no_stride]

    # add a tranpose convolution with stride 2 and output padding 1
    tconvs_op1 = [b + [{"type": "tconv", "kernel_size": 2, "stride": 2, "output_padding": 1}] for b in tconvs_no_stride]

    tconvs = tconvs_op0 + tconvs_op1

 
    # For each possible block, compute the output dimension
    for block in tconvs:
        res_output_dim = get_input_dim(output_dim, block)
        
        if res_output_dim is None or res_output_dim < min_threshold:
            continue

        min_ks = min([tconv["kernel_size"] for tconv in block if tconv["stride"] == 1])

        if res_output_dim in res:
            # check if the min_ks is the same as the previous one
            if (res_output_dim, min_ks) in res_ks_size:
                continue
            else:
                res_ks_size.add((res_output_dim, min_ks))
        else:
            res.add(res_output_dim)
            res_ks_size.add((res_output_dim, min_ks))

        # Continue the recursion with this output as the new input
        dp_compute_all_possible_inputs(min_n, max_n, res_output_dim, min_threshold, res, res_ks_size, memo, num_blocks - 1)  



def compute_all_possible_inputs(min_n: int, 
                                max_n: int, 
                                output_dim: int, 
                                num_blocks: int) -> Set[int]:
    """
    Compute all possible output dimensions that could result from the given input dimension
    when applying a series of transpose convolutional blocks.
    """
    res = set()
    memo = set()
    res_ks_size = set()

    # add the largest kernel size associated with the output_dim
    res_ks_size.add((output_dim, 7))

    dp_compute_all_possible_inputs(min_n, max_n, output_dim, 1, res, res_ks_size, memo, num_blocks) 
    # # remove the output_dim itself from the result
    # res.remove(output_dim) 
    
    return res


def dp_compute_outputs_exhaustive(min_n: int, 
                                  max_n: int, 
                                  output_dim: int, 
                                  num_blocks: int,
                                  min_threshold:int, 
                                  res: Set[int], 
                                  memo: Set[int]):
    # If we've already processed this output_dim, don't repeat the work
    if output_dim in memo or output_dim < min_threshold:
        return
    
    # Mark this output_dim as processed and add it to the results
    memo.add(output_dim)
    res.add(output_dim)

    # Nothing to be done with 0 blocks
    if num_blocks == 0:
        return
    

    # Get all possible kernel combinations for the given constraints
    kernel_combs = get_possible_kernel_combs(min_n, max_n, max_kernel_size=7, min_kernel_size=3, reverse=False)
    
    # each element in kernel_combs is a list (with number of elements between min_n and max_n)
    # for each elements in these lists, add the stride 1 and output padding 0
    tconvs_no_stride = [[_get_single_transpose_conv_rep(ks, 1, 0) for ks in kc] for kc in kernel_combs]
    
    # add a tranpose convolution with stride 2 and output padding 0
    tconvs_op0 = [b + [{"type": "tconv", "kernel_size": 2, "stride": 2, "output_padding": 0}] for b in tconvs_no_stride]

    # add a tranpose convolution with stride 2 and output padding 1
    tconvs_op1 = [b + [{"type": "tconv", "kernel_size": 2, "stride": 2, "output_padding": 1}] for b in tconvs_no_stride]

    tconvs = tconvs_op0 + tconvs_op1

 
    # For each possible block, compute the output dimension
    for block in tconvs:
        res_output_dim = get_input_dim(output_dim, block)
        
        if res_output_dim is None:
            continue

        # Continue the recursion with this output as the new input
        dp_compute_outputs_exhaustive(min_n, max_n, res_output_dim, num_blocks - 1, min_threshold, res, memo)  

def compute_outputs_exhaustive(min_n: int, 
                              max_n: int, 
                              output_dim: int, 
                              num_blocks: int) -> Set[int]:
    """
    Compute all possible output dimensions that could result from the given input dimension
    when applying a series of transpose convolutional blocks.
    """
    res = set()
    memo = set()

    dp_compute_outputs_exhaustive(min_n, max_n, output_dim, num_blocks, 1, res, memo) 
    # remove the output_dim itself from the result
    res.remove(output_dim) 
    
    return res
