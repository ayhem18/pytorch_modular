# Overview

The main goal of this module is to provide a simple and easy to use interface for building custom models on top of existing pretrained models.

## ResNet as a feature extractor.

Resnet is probably the most used backbone in computer vision. Let's break it down into its main components: 

1. The very first convolutional block
2. The "Layer" blocks
3. The average pooling layer
4. The fully connected layer


The first layer block is not particularly interesting. The core of the network is the "Layer" blocks. It is broken down into:
    - Layer
        - Bottleneck: 
            - a number of convolutional layers with a skip connection 


Depending on the architecture, a layer can have a different number of bottleneck blocks. 

The goal of the resnetFeatureExtractor is to extract any portion of the network and use it as a custom feature extractor. However, the smallest building block is a "bottleneck" block. In other words, it is possible to choose the network up until the n-th layer block, or the n-th bottleneck block (residual block).






