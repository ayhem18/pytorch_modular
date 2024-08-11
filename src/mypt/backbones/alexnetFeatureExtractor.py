"""
This script contains functionalities to build classifiers on top of the pretrained architecture AlexNet provided by Pytorch
"""

import torch

from typing import OrderedDict, Union, Tuple, List, Iterator
from torch import nn
from torchvision.models import alexnet, AlexNet_Weights
from copy import deepcopy


class AlexNetFeatureExtractor(nn.Module):
    """
    Since the architecture is relatively small, we can have more fine-grained control over its different components. The model will be split accordingly: 
    * conv1: [nn.Convolution layer, Relu, MaxPool]
    * conv2: [nn.Convolution layer, Relu, MaxPool]
    * conv3: [nn.Convolution laye>r, Relu]
    * conv4: [nn.Convolution layer, Relu]
    * conv5: [nn.Convolution layer, Relu, MaxPool]
    * adapool: Adaptive Pooling layer
    * fc1: [dropout, linear, relu]
    * fc2: [dropout, linear, relu]
    * fc3: [linear]
    """
    
    __block2index = {"conv1":0 , 
                    "conv2":1 , 
                    "conv3":2 , 
                    "conv4":3 , 
                    "conv5":4 , 
                    "avgpool":5 ,
                    "fc1":6 ,
                    "fc2":7 ,
                    "fc3":8
                    }
    
    __index2block =  {0: "conv1", 
                    1: "conv2", 
                    2: "conv3", 
                    3: "conv4", 
                    4: "conv5", 
                    5: "avgpool",
                    6: "fc1",
                    7: "fc2",
                    8: "fc3",
                    }

    __list_str_arguments = [f'conv{i}' for i in range(1, 6)] + [f'fc{i}' for i in range(1, 4)]
    __str_arguments = ['all', 'conv_block', 'conv_block_adapool'] + __list_str_arguments

    @classmethod
    def __verify_blocks(cls, blocks: Union[str, List[str], int, List[int]]):
        # make sure the argument is one of the expected arguments
        if isinstance(blocks, str):
            # if the argument is a string, then it should be one of the arguments written above
            if blocks not in cls.__str_arguments:
                raise ValueError(f"The initialize received an expected argument: {blocks}. string arguments are expected to be on of {cls.__str_arguments}.")
            return 
        
        if isinstance(blocks, int):
            if not ( 0 <= blocks <= 8):
                raise ValueError(f"The initializer received an expected argument: {blocks}. integer arguments are expected to belong to the interval [0, 8].")
            return 
        
        if isinstance(blocks, List):
            # make sure all the elements are of the same type (either all string or all integers)
            if isinstance(blocks[0], str):
                if not (all([isinstance(b, str) for b in blocks])):
                    raise TypeError(f"All elements in the list must have the same type. Found element of different types")

                # each element should be in the cls.__list_str_arguments
                for b in blocks:
                    if b not in cls.__list_str_arguments:
                        raise ValueError(f"elements of a string list argument are expected to belong to {cls.__list_str_arguments}. Found: {b}")
                
                # make sure the order is corect
                sorted_argument_by_index = sorted(blocks, key=lambda x: cls.__block2index[x], reverse=False)
                if sorted_argument_by_index != blocks:
                    raise ValueError(f"When the blocks are passed as a list of strings. Make sure the order matches the original architecture: Found: {blocks}.\n" 
                                     f"Expected: {sorted_argument_by_index}")
            
        
            if isinstance(blocks[0], int):
                for b in blocks: 
                    if not isinstance(b, int):
                        raise TypeError(f"All elements in the list must have the same type. Found element of different types")
                    # make sure each value belong to the interval
                    cls.__verify_blocks(b)

                # make sure the indices are in ascending order
                sorted_argument_by_index = sorted(blocks)
                if sorted_argument_by_index != blocks:
                    raise ValueError(f"When the blocks are passed as a list of indices. Make sure the indices are in ascending order. Found: {blocks}.\n"
                                     f"Expected: {sorted_argument_by_index}")
                
            return 
        
        # if the 'blocks' argument is none of the above, then raise a Type error
        raise TypeError(f"The initialize received an argument of unexpected type: {type(blocks)}. The argument should be either {int}, {str} or {List}")

    @classmethod
    def __verify_frozen_blocks(cls,
                               model_blocks: Union[str, List[str], int, List[int]], 
                               frozen_blocks: Union[bool, str, List[str], int, List[int]]):
        """This function checks the arguments used to determine which blocks to freeze. They are subject to the following considerations: 
        1. if 'frozen_blocks' is a bool variable, then freeze all blocks or leave them as they are

        2. if 'model_blocks' is a 'str', then it should represent a specific layer, it cannot be one of the 'cls.__str_arguments'
            additionally model_blocks must be of 'str' or [str]
        
        3. if 'model_blocks' is 'int', then it has to a value present in 'model_blocks' (or equal) 

        4. 
        Args:
            model_blocks (Union[str, List[str], int, List[int]]): The blocks extracted from the architecture
            frozen_blocks (Union[bool, str, List[str], int, List[int]]): The blocks to be frozen
        """

        if isinstance(frozen_blocks, bool):
            return        

        if isinstance(frozen_blocks, str):
            # make sure the types match between selected blocks and frozen ones
            if not (isinstance(model_blocks, (str)) or isinstance(model_blocks[0], str)):
                raise ValueError(f"if the model_blocks argument is a string-like, then so does the 'frozen_blocks'")

            # make sure the string is valid: one of the 
            if frozen_blocks not in [cls.__list_str_arguments]:
                raise ValueError(f"the frozen blocks must be one of the following arguments: {cls.__list_str_arguments}")
            
            # make sure not to freeze fully connected layers when we 
            if model_blocks in cls.__str_arguments[1:] and not frozen_blocks.startswith('conv'):
                raise ValueError(f"The model uses only convolutional blocks, but the 'frozen blocks' refers to other components: {frozen_blocks}")
            
            return 

        if isinstance(frozen_blocks, int):
            if not (isinstance(model_blocks, (int)) or isinstance(model_blocks[0], int)):
                raise ValueError(f"if the model_blocks argument is an integer-like argument, then so does the 'frozen_blocks'")
            
            if not (frozen_blocks == model_blocks or frozen_blocks in model_blocks): 
                # the first equality is for the case where 'model_blocks' is an interer, the 2nd of the List case
                raise ValueError(f"the frozen_blocks are expected to be part of the model blocks. Frozen_blocks: {frozen_blocks}, model_blocks: {model_blocks}")
            return 
        
        if isinstance(frozen_blocks, List):            
            if isinstance(model_blocks, List) and not set(frozen_blocks).issubset(model_blocks):
                raise ValueError(f"the frozen_blocks are expected to be part of the model blocks. Frozen_blocks: {frozen_blocks}, model_blocks: {model_blocks}")
        
            if model_blocks in cls.__str_arguments[1:]:
                for b in frozen_blocks:
                    if not b.startswith('conv'):
                        raise ValueError(f"The model uses only convolutional blocks, but the 'frozen blocks' refers to other components: {frozen_blocks}")
            
            if model_blocks == cls.__str_arguments[0]: # if we are using the entire architecture
                cls.__verify_blocks(frozen_blocks)

            return
        
        raise TypeError(f"The initialize received an argument of unexpected type: {type(b)}. The argument should be either {int}, {str} or {List}")

    def __set_convolutional_block(self, convolutional_block: nn.Sequential):
        """
        This function splits the convolutional block introduced by AlexNet into 5 convolutional sub-blocks that can later 
        be addressed and manipulated independently.

        The assumption is that a convolutional block starts with a convolutional layer and ends right before the next convolutional layer 
        (or the end of the convolutional block)
        
        Args:
            convolutional_block (nn.Sequential): the AlexNet convolutional block.
        """
        
        block_index = 0
        current_block = []

        for c in convolutional_block.children():
            # when we find a convolutional layer, 
            if isinstance(c, nn.Conv2d):
                # if the 'current_block' is empty
                if len(current_block) == 0: 
                    current_block.append(deepcopy(c))
                else: 
                    # save the current block and create a new one
                    self.block_indices[block_index] = nn.Sequential(*current_block)
                    current_block = [deepcopy(c)]
                    # increment the block index
                    block_index += 1
            else: 
                current_block.append(deepcopy(c))

        # after iterating through the convolutional block, make sure to add the last blok
        self.block_indices[block_index] = nn.Sequential(*current_block)

        # some test to make sure the distribution is as expected
        assert len(self.block_indices) == 5, f"make sure the total number of block is '5'. Found: {len(self.block_indices)}"
        __lengths = [3, 3, 2, 2, 3]
        for l, (index_block, block) in zip(__lengths, self.block_indices.items()):
            assert len(block) == l, f"Make sure the block with index {index_block}. Found: {len(block)}"

    def __set_fc_block(self, fc_block: nn.Sequential):
        """
        This function splits the fully connected block introduced by AlexNet into 3 fully-connected sub-blocks that can later 
        be addressed and manipulated independently.

        Args:
            fc_block (nn.Sequential): the fully connected block introduced by AlexNet.
        """
        
        block_index = 0
        blocks = [(0, 2), (3, 5), (6,)]
        
        current_block = []
        for layer_index, c in enumerate(fc_block.children()):
            if layer_index > blocks[block_index][-1]:
                self.block_indices[len(self.block_indices)] = nn.Sequential(*current_block)
                current_block = [deepcopy(c)]
                block_index += 1
            else: 
                current_block.append(deepcopy(c))

        self.block_indices[len(self.block_indices)] = nn.Sequential(*current_block)

        # some test to make sure the distribution is as expected
        assert len(self.block_indices) == 9, f"make sure the total number of blocks is '9'. Found: {len(self.block_indices)}"
        __lengths = [3, 3, 1]
        for l, (index_block, block) in zip(__lengths, [(6, self.block_indices[6]), (7, self.block_indices[7]), (8, self.block_indices[8])]):
            assert len(block) == l, f"Make sure the block with index {index_block} has length {l}. Found: {len(block)}"

    def __set_blocks(self):
        """
        This function maps each basic block to an index so that be accessed easily later on        
        """
        convolutional_block, avgpool, fc_block = list(self.__net.children())
        self.__set_convolutional_block(convolutional_block=convolutional_block)
        self.block_indices[5] = avgpool
        self.__set_fc_block(fc_block=fc_block)

    def __build_model_str(self, blocks: str) -> nn.Module:
        if blocks == self.__str_arguments[0]:
            return nn.Sequential(OrderedDict([(self.__index2block[block_index], block) for block_index, block in self.block_indices.items()]))
                    
        elif blocks == self.__str_arguments[1]:
            return nn.Sequential(OrderedDict([(self.__index2block[i], self.block_indices[i]) for i in range(5)]))
        
        elif blocks == self.__str_arguments[2]:
            return nn.Sequential(OrderedDict([(self.__index2block[i], self.block_indices[i]) for i in range(6)]))

    def _build_model(self, blocks: Union[str, List[str], int, List[int]]) -> nn.Sequential:
        if isinstance(blocks, str):
            return self.__build_model_str(blocks)
        
        if isinstance(blocks, int):
            return self.block_indices[blocks]
        
        if isinstance(blocks, List):
            if isinstance(blocks[0], str):
                # map each name to the index
                return nn.Sequential(OrderedDict([(b, self.block_indices[self.__block2index[b]]) for b in blocks]))
            
            return nn.Sequential(OrderedDict([(self.__index2block[b], self.block_indices[b]) for b in blocks]))

    def _freeze_model(self, 
                      frozen_blocks: Union[bool, str, List[str], int, List[int]]) -> None:
        if isinstance(frozen_blocks, bool):
            if not frozen_blocks:
                # if 'frozen_blocks' is set to False, the nothing to do... 
                return 
            # freeze all blocks
            for _, block in self.block_indices.items():
                for p in block.parameters():
                    p.requires_grad = False
        
            return 

        if isinstance(frozen_blocks, (str, int)):
            frozen_blocks = [frozen_blocks]

        for b in frozen_blocks:
            b = b if isinstance(b, int) else (self.__block2index[b])
            for p in self.block_indices[b].parameters():
                p.requires_grad = False
        
    def __init__(self, 
                model_blocks: Union[str, List[str], int, List[int]],
                frozen_model_blocks: Union[str, List[str], int, List[int]],
                *args, 
                **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # make sure to initialize the Alexnet as a field
        self.__net = alexnet(weights=AlexNet_Weights.DEFAULT)
        # save the default transform that comes with the AlexNet architecture
        self.transform = AlexNet_Weights.DEFAULT.transforms()
        self.block_indices = {}
        self.__set_blocks()
        
        # make sure the blocks are passed correctly
        self.__verify_blocks(blocks=model_blocks)    

        # time to freeze the model                    
        self.__verify_frozen_blocks(model_blocks=model_blocks, 
                                    frozen_blocks=frozen_model_blocks)
        
        # freeze the needed layers
        self._freeze_model(frozen_blocks=frozen_model_blocks)

        # build the model
        self.model = self._build_model(model_blocks)

        # delete the __net field and block_indices fields to reduce the size of the class
        del(self.__net)
        del(self.block_indices)


    def forward(self, x: torch.Tensor):
        # the forward function in the ResNet class simply calls the forward function
        # of each submodule consecutively: which is equivalent to saving all modules in a nn.Sequential module
        # and calling the forward method.
        return self.model.forward(x)

    def __str__(self):
        # the default __str__ function will display the self.__net module as well
        # which might be confusing as .__net is definitely not part of the forward pass of the model
        return self.model.__str__()
    
    def __repr__(self):
        return self.model.__repr__() 
    
    def children(self) -> Iterator[nn.Module]:
        # not overloading this method will return to an iterator with 2 elements: self.__net and self.feature_extractor
        return self.model.children()

    def modules(self) -> Iterator[nn.Module]:
        return self.model.modules()
    
    def named_children(self) -> Iterator[Tuple[str, nn.Module]]:
        return self.model.named_children()

if __name__ == '__main__':
    # m = AlexNetFeatureExtractor(model_blocks='all', frozen_model_blocks=['conv5', 'fc1', 'fc3'])
    # for n, b in m.named_children():
    #     print(n)
    #     for b in b.parameters():
    #         print(b.requires_grad)
    #     print("#" * 100)
    
    # for i in range(1, 6):
    #     eval(f'm.model.conv{i}')

    it = iter([1, 2, 3])
    a = next(it, None)
    while a is not None:
        print(a)
        a = next(it, None)       
