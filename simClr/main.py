from mypt.backbones.resnetFeatureExtractor import ResNetFeatureExtractor


if __name__ == '__main__':
    fe = ResNetFeatureExtractor(num_layers=-1)
    print(fe.transform)
