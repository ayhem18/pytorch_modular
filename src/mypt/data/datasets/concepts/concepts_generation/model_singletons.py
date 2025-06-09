"""
this script contains functionalities and models used across the different stages to prepare the
data to feed the Concept Bottleneck Models
"""

import clip
import torch

from typing import Union
from sentence_transformers import SentenceTransformer
from tenacity import retry, wait_random_exponential, stop_after_attempt, stop_after_delay

@retry(wait=wait_random_exponential(multiplier=2, min=2, max=15), stop=(stop_after_delay(20) | stop_after_attempt(10)))
def get_sentence_transformer(model_name: str = 'all-mpnet-base-v2'):
    return SentenceTransformer(model_name)


class CBM_SingletonInitializer(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(CBM_SingletonInitializer, cls).__new__(cls)
            # set the attributes of cls
            # set the 'device'
            cls.instance.__device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # initialize the mpnet model
            cls.instance.__mpnet_model = get_sentence_transformer()
            cls.instance.__clip_model, cls.instance.__clip_processor = clip.load("ViT-B/32",
                                                                                 device=cls.instance.__device)

        return cls.instance

    def get_device(self):
        return self.__device

    def get_mpnet_model(self):
        return self.__mpnet_model

    def get_clip_model(self):
        return self.__clip_model

    def get_clip_processor(self):
        return self.__clip_processor

    def set_mpnet_model_device(self, device: Union[str, torch.device]):
        return self.__mpnet_model.to(device)

    def set_clip_model_device(self, device: Union[str, torch.device]):
        return self.__clip_model.to(device)

    def set_clip_processor_device(self, device: Union[str, torch.device]):
        return self.__clip_processor.to(device)


