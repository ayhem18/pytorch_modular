"""
This script contains functionalities to test the code written in both
'layer specific' and 'dimension_analyser' scripts
"""

import os
import sys
import torch
import unittest

from pathlib import Path
from random import randint as ri

try:
    import src.pytorch_modular.random_generation.random_components as rc
except ModuleNotFoundError:
    h = os.getcwd()
    while 'src' not in os.listdir(h):
        print(h)
        h = Path(h).parent
    # at this point 'dir' should be the root of the repository
    # make sure to convert the path to a string as sys.path does not recognize Path objects
    sys.path.append(str(h))

import src.pytorch_modular.transfer_learning.resnetFeatureExtractor as res_fe
import src.pytorch_modular.random_generation.random_components as rc
import src.pytorch_modular.dimensions_analysis.dimension_analyser as da
from src.pytorch_modular.pytorch_utilities import get_default_device


class TestOutputAnalysis(unittest.TestCase):

    # def test_output_analyser_components(self):
    #     # device = get_default_device()
    #     analyser = da.DimensionsAnalyser(method=da._STATIC)
    #     cpu = 'cpu'
    #     for _ in range(50):
    #         in_channels, conv_block = rc.random_conv_block(return_in_c=True)
    #         # conv_block.to()
    #         conv_block.eval()
    #         # create the input shape
    #
    #         # settings the height and width to significantly large numbers
    #         # as the random settings of the convolutional block makes it likely for the dimensions to decrease
    #         # extremely fast
    #         random_height = ri(2000, 4000)
    #         random_width = ri(2000, 4000)
    #         random_batch = ri(1, 3)  # keeping the random_batch small for memory concerns (mainly with GPU)
    #
    #         input_shape = (random_batch, in_channels, random_height, random_width)
    #         input_tensor = torch.ones(size=input_shape)
    #         for m in conv_block:
    #             shape_computed = analyser.analyse_dimensions(input_shape, m)
    #             input_tensor = m.forward(input_tensor)
    #             input_shape = tuple(input_tensor.size())
    #
    #             self.assertEqual(input_shape, shape_computed), "THE OUTPUT SHAPES ARE DIFFERENT"
    #         # clear the GPU is needed
    #         torch.cuda.empty_cache()
    #         # deleting the unnecessary variable
    #         del conv_block
    #         del input_tensor

    # def test_output_with_random_network(self):
    #     # create a module analyser
    #     # make sure to use static
    #     analyser = da.DimensionsAnalyser(method=da._STATIC)
    #     # 50 test cases should be good enough
    #     for _ in range(50 ):
    #         in_channels, conv_block = rc.random_conv_block(return_in_c=True)
    #         conv_block.eval()
    #         # create the input shape
    #         random_batch = ri(1, 2)  # keeping the random_batch small for memory concerns (mainly with GPU)
    #         random_height = ri(1000, 2000)
    #         random_width = ri(1000, 2000)
    #
    #         input_shape = (random_batch, in_channels, random_height, random_width)
    #         computed_output_shape = analyser.analyse_dimensions(input_shape, conv_block)
    #
    #         input_tensor = torch.ones(size=input_shape)
    #         output_tensor = conv_block.forward(input_tensor)
    #
    #         self.assertEqual(tuple(output_tensor.size()), computed_output_shape), "THE OUTPUT SHAPES ARE DIFFERENT"
    #         # make sure to clear the gpu
    #         torch.cuda.empty_cache()
    #         # delete the variables
    #         del conv_block
    #         del input_tensor
    #         del output_tensor

    def test_output_pretrained_network(self):
        layers = list(range(1, 5))
        analyser = da.DimensionsAnalyser(method=da._STATIC)

        for v in layers:
            feature_extractor = res_fe.ResNetFeatureExtractor(num_blocks=v)
            feature_extractor.eval()

            for _ in range(50):
                # create the input shape
                random_batch = ri(1, 3)  # keeping the random_batch small for memory concerns (mainly with GPU)
                random_height = ri(1000, 2000)
                random_width = ri(1000, 2000)

                # resnet expected a usual image with 3 channels
                input_shape = (random_batch, 3, random_height, random_width)
                computed_output_shape = analyser.analyse_dimensions(input_shape, feature_extractor)

                input_tensor = torch.ones(size=input_shape)
                output_tensor = feature_extractor.forward(input_tensor)

                self.assertEqual(tuple(output_tensor.size()), computed_output_shape), "THE OUTPUT SHAPES ARE DIFFERENT"
                # make sure to clear the gpu
                torch.cuda.empty_cache()
                # delete the variables
                # del feature_extractor
                del input_tensor
                del output_tensor


if __name__ == '__main__':
    unittest.main()
