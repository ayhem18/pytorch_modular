"""
This script contains the main class responsible for generating concept-labels for a Concept Bottleneck model
"""
import torch
import clip

from torch import nn
from PIL import Image
from pathlib import Path
from typing import List, Union, Iterable


from .model_singletons import CBM_SingletonInitializer

class ClipLabelGenerator:
    def __init__(self, similarity_as_cosine: bool = True):
        """            
        similarity_as_cosine (bool, optional): if True, measure the similarity between image and text encodings with cosine similarity, otherwise dot product
        Defaults to True.
        """
        # Despite taking extra care to seed every conceivable source of randomness, the results with CBMs were almost never reproducible
        # this issue on github gave me hope: https://github.com/openai/CLIP/issues/13 
        # each call with the CLIP model will be carried twice. The first call is a warm up and hopefully the second one will be the same across runs

        self.clip_model, self.image_processor, self.device = (CBM_SingletonInitializer().get_clip_model(),
                                                              CBM_SingletonInitializer().get_clip_processor(),
                                                              CBM_SingletonInitializer().get_device())
        # make sure to put the clip_model and the image processor to the device 
        self.clip_model.to(self.device)

        # set clip_model to evaluation mode
        self.clip_model.eval()

        if hasattr(self.image_processor, 'to'):
            self.image_processor.to(self.device)

        # create a softmax torch layer to finalize the labels: convert them from cosine differences to probability-like values
        self.softmax_layer = nn.Softmax(dim=1).to('cpu')

        self.cosine = similarity_as_cosine

    def encode_concepts(self, concepts: List[str]) -> torch.Tensor:
        # tokenize the text
        concepts_tokens = clip.tokenize(concepts).to(self.device)

        # wrap up the encode_text
        with torch.no_grad():
            # encode the concepts: features
            concepts_clip = self.clip_model.encode_text(concepts_tokens).detach().cpu()

        # make sure the embeddings are of the expected shape
        batch_size, embedding_dim = concepts_clip.shape

        if batch_size != len(concepts):
            raise ValueError((f"Please make sure the batch size of the CLIP text embeddings match the number of concepts\n"
                              f"number of concepts: {len(concepts)}. batch size found: {batch_size}"))
        
        # make sure to keep only the concepts encodings in the 'gpu'
        concepts_tokens = concepts_tokens.detach().to('cpu')

        return concepts_clip


    def _embed_images(self, images: Union[Iterable[Union[str, Path, torch.Tensor]], str, Path, torch.Tensor]) -> torch.Tensor:
        # if only one image was passed, wrap it in a list
        if not isinstance(images, Iterable):
            images = [images]
        
        # if the images are passed as paths, read them
        if isinstance(images[0], (str, Path)):
            images = [Image.open(i) for i in images]

        # process the image: process each image with the CLIP processor (the CLIP.processor does not seem to support batching)
        # convert them to Tensors and stack them into a single tensor
        processed_images = torch.stack([self.image_processor(im) for im in images]).to(self.device)

        # if the data is given as a tensor, then compute the cosine difference
        image_embeddings = self.clip_model.encode_image(processed_images).detach().to(dtype=torch.float16, device='cpu')
        
        # move the processed images back to cpu
        processed_images = processed_images.to(device='cpu')

        return image_embeddings


    def _compute_cosine_similarity(self, 
                                   image_embeddings: torch.Tensor, 
                                   concepts_features: torch.Tensor) -> torch.Tensor:
        # normalize both the image and concepts embeddings
        image_embeddings /= torch.linalg.norm(image_embeddings, dim=-1, keepdim=True)
        concepts_features /= torch.linalg.norm(concepts_features, dim=-1, keepdim=True)

        num_concepts, emb_text_dim = concepts_features.size()
        num_images, emb_img_dim = image_embeddings.size() 

        if emb_text_dim != emb_img_dim:
            raise ValueError((f"In the current setting, image embddings do not match text embddings size-wise.\n"
                            f"Found: text dim: {emb_text_dim}. img dim: {emb_img_dim}"))

        cn = torch.linalg.norm(concepts_features, dim=1, dtype=torch.float32)
        imn = torch.linalg.norm(image_embeddings, dim=1, dtype=torch.float32)

        # make sure the tensors are normalized
        if not torch.allclose(cn, torch.ones(num_concepts, dtype=torch.float32), atol=10**-3):
            raise ValueError(f"The features are not normalized correctly")

        if not torch.allclose(imn, torch.ones(num_images, dtype=torch.float32), atol=10**-3):
            raise ValueError(f"The features are not normalized correctly")

        # return the cosine difference for every image, concept tuple
        cosine_diffs = image_embeddings @ concepts_features.T

        return cosine_diffs


    def _compute_dot_product(self, image_embeddings: torch.Tensor, concepts_features: torch.Tensor) -> torch.Tensor:
        # at this point, the similarity will be computed with the dot product 
        _, emb_text_dim = concepts_features.size()
        _, emb_img_dim = image_embeddings.size() 

        if emb_text_dim != emb_img_dim:
            raise ValueError((f"In the current setting, image embddings do not match text embddings size-wise.\n"
                    f"Found: text dim: {emb_text_dim}. img dim: {emb_img_dim}"))

        # return the dot product between every image, concept tuple
        dot_products = image_embeddings @ concepts_features.T

        return dot_products


    def generate_image_label(self,
                             images: Union[Iterable[Union[str, Path, torch.Tensor]], str, Path, torch.Tensor],
                             concepts_features: Union[torch.Tensor, List[str]],
                             apply_softmax: bool = True
                             ) -> torch.Tensor:

        images_embeddings = self._embed_images(images)

        if self.cosine:
            cosine_diffs = self._compute_cosine_similarity(images_embeddings, concepts_features).to(torch.float32)

            if apply_softmax:
                return self.softmax_layer(cosine_diffs)

            return cosine_diffs

        dot_products = self._compute_dot_product(images_embeddings, concepts_features).to(torch.float32)

        if apply_softmax:
            return self.softmax_layer(dot_products)

        return dot_products

