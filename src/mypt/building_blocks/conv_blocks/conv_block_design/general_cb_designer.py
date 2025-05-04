"""
This module contains a 
"""

import math
import warnings
import numpy as np

from typing import List, Tuple








# class ContractingConvBlockDesigner:
#     def __init__(self, 
#                  input_shape: Tuple[int, int, int],
#                  output_shape: Tuple[int, int, int],
#                  max_conv_layers_per_block: int = 4,
#                  min_conv_layers_per_block: int = 2):  
#         # make sure that the input shape is larger than the output shape
#         if input_shape[1] <= output_shape[1] or input_shape[2] <= output_shape[2]:
#             raise ValueError("Input shape must be larger than output shape")
        
#         # make sure that number of channels in the output is larger than the input
#         if input_shape[0] >= output_shape[0]:
#             raise ValueError("Number of channels in the output must be larger than the input")
        
#         self.ic = input_shape[0]
#         self.ih = input_shape[1]
#         self.iw = input_shape[2]

#         self.oc = output_shape[0]
#         self.oh = output_shape[1]
#         self.ow = output_shape[2]
        

#         self.max_conv_layers_per_block = max(max_conv_layers_per_block, min_conv_layers_per_block)
#         self.min_conv_layers_per_block = min(max_conv_layers_per_block, min_conv_layers_per_block)

#         if self.min_conv_layers_per_block < 2:
#             raise ValueError("The minimum number of convolution layers per block must be at least 2")

#         if self.max_conv_layers_per_block > 6:
#             warnings.warn("The maximum number of convolution layers per block is larger than 6. Most popular architectures have at most 6 consecutive convolution layers (without pooling or strides)") 

#     def _compute_dimensions_passive(self, input_dim: int, output_dim: int):
#         # this method comes up with the dimensions of the convolution blocks without using pooling layers or strided convolutions 
#         pass


#     def _compute_dimensions(self, input_dim: int, output_dim: int):
        
#         # compute the number of layers in the block
#         ratio = np.log2(input_dim / output_dim)
        
#         if math.ceil(ratio) != math.floor(ratio):
#             # the ratio is not a power of 2
#             # raise ValueError("The ratio of the input and output dimensions must be a power of 2")
#             pass
        
#         num_layers = int(ratio) + 1 

#         if num_layers == 1:

#             return self._compute_dimensions_passive(input_dim, output_dim) 

#         # at this point we need to find the aggreesive part with either pooling or strided convolutions
#         current_dim = output_dim 

#         # each block parameters will contain the following (in a reverse order):
#         # 1. (kernel_size, stride) of the pooling layer (the last layer in the block)
#         # 2. (kernel_size) of the convolution layers ... (a varying number of layers)

#         blocks = []

#         for i in range(num_layers): 
#             pass



# class ExpansiveConvBlockDesigner:
#     def __init__(self, 
#                  input_shape: Tuple[int, int, int]):  
#         pass



