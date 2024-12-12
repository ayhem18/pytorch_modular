# Overview

mypt is short for My Pytorch. This is a simple library implementing routines tasks and subroutines often used in Deep Learning tasks.

## Subfolders: 

The library is under construction. As of the moment of writing this readme file, the following sections are implemented:

1. backbones: code to faciliate working and manipulating famous Feature Extractors such as Resnet and Alexnet
2. code_utilities: utility code such as file manipulation for visual tasks, ensuring reproducibility...
3. dimension_analysis: code to determine the shape of the output when passed through a given torch.nn.Module without specifically calling the module.
4. data: dataset classes for specific task and automating data-related code (e.g initializing a reproducible dataloader )
5. subroutines: K-nearest neighbors with Pytorch