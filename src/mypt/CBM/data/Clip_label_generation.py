"""
This script contains the main class responsible for generating concept-labels for a Concept Bottleneck model
"""
from PIL import Image
import torch
import clip

from torch import nn
from typing import List, Union, Iterable
from pathlib import Path

from ...code_utilities.model_singletons import CBM_SingletonInitializer

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

    def encode_concepts(self, concepts: List[str], 
                        debug_memory: bool = False) -> torch.Tensor:
        # tokenize the text
        concepts_tokens = clip.tokenize(concepts).to(self.device)

        # wrap up the encode_text
        with torch.no_grad():
            # encode the concepts: features
            concepts_clip = self.clip_model.encode_text(concepts_tokens).detach().cpu()
            # repeat the call 
            # concepts_clip = self.clip_model.encode_text(concepts_tokens).detach().cpu()

        # make sure the embeddings are of the expected shape
        batch_size, embedding_dim = concepts_clip.shape

        if batch_size != len(concepts):
            raise ValueError((f"Please make sure the batch size of the CLIP text embeddings match the number of concepts\n"
                              f"number of concepts: {len(concepts)}. batch size found: {batch_size}"))
        
        # make sure to keep only the concepts encodings in the 'gpu'
        concepts_tokens = concepts_tokens.detach().to('cpu')

        if debug_memory:
            print(f"Concept Encoding: Memory before clearing: {round(torch.cuda.memory_allocated() / (1024 ** 2), 3)}  Mbs")
            torch.cuda.empty_cache()
            print(f"Concept Encoding: Memory after clearing: {round(torch.cuda.memory_allocated() / (1024 ** 2), 3)}  Mbs")

        return concepts_clip

    def generate_image_label(self,
                             images: Union[Iterable[Union[str, Path, torch.Tensor]], str, Path, torch.Tensor],
                             concepts_features: Union[torch.Tensor, List[str]],
                             debug_memory: bool = False, 
                             apply_softmax: bool = True
                             ) -> torch.Tensor:

        # if only one image was passed, wrap it in a list
        if not isinstance(images, Iterable):
            images = [images]

        # if the images are passed as paths, read them
        if isinstance(images[0], (str, Path)):
            images = [Image.open(i) for i in images]

        # process the image: process each image with the CLIP processor (the CLIP.processor does not seem to support batching)
        # convert them to Tensors and stack them into a single tensor
        processed_images = torch.stack([self.image_processor(im) for im in images]).to(self.device)

        # proceeding depending on the type of the passed 'concepts'
        if isinstance(concepts_features, List) and isinstance(concepts_features[0], str):
            # if the given concepts are in textual form then we can pass the data directly to the CLIP model
            logits_per_image, _ = self.clip_model(images, concepts_features)
            # logits_per_image, _ = self.clip_model(images, concepts_features)
            # as per the documentation of the CLIP model: https://github.com/openai/CLIP
            # logits_per_image represents the cosine difference between the embedding of the images with respect
            # to the given textual data
            return logits_per_image

        if debug_memory:
            print(f"Creating concept labels: GPU memory before passing images to CLIP: {torch.cuda.memory_allocated() / (10 ** 6)} MB")

        # if the data is given as a tensor, then compute the cosine difference
        image_embeddings = self.clip_model.encode_image(processed_images).detach().to(dtype=torch.float16, device='cpu')
        # call twice
        # image_embeddings = self.clip_model.encode_image(processed_images).detach().to(dtype=torch.float16, device='cpu')

        #NOTE: THE ONLY OPERATION USING THE GPU IS IMAGE ENCODING, EVERYTHING FROM THIS POINT ON IS ON CPU
        # make sure set the image_embeddings to cpu the moment they are produced as well as change their type to float32
        processed_images = processed_images.to(device='cpu')
        concepts_features = concepts_features.to(dtype=torch.float16, device='cpu')

        # empty the cache as everything is in cpu now,
        torch.cuda.empty_cache()
        
        if debug_memory:
            print(f"Creating concept labels: GPU memorey after clearing memory: {torch.cuda.memory_allocated() / (10 ** 6)} MB")

        if self.cosine:

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

            # return the cosine difference between every image, concept tuple
            cosine_diffs = image_embeddings @ concepts_features.T

            if apply_softmax:
                # the final step is to pass the cosine differences through the softmax layer. 
                labels = self.softmax_layer(cosine_diffs)

                # make sure the labels are all positive
                assert torch.all(labels >= 0), "Some of the final labels are negative"

                labels_sum = torch.sum(labels, dim=1, dtype=torch.float)
                ones = torch.ones(labels.size(dim=0), dtype=torch.float, device='cpu')
                assert torch.allclose(labels_sum, ones, atol=10 ** -3)

                assert labels.shape == (num_images, num_concepts), f"The shape of the labels is not as expected. Expected: {(num_images, num_concepts)}. Found: {labels.shape}"

                return labels

            return cosine_diffs

        # at this point, the similarity will be computed with the dot product 
        num_concepts, emb_text_dim = concepts_features.size()
        num_images, emb_img_dim = image_embeddings.size() 

        if emb_text_dim != emb_img_dim:
            raise ValueError((f"In the current setting, image embddings do not match text embddings size-wise.\n"
                    f"Found: text dim: {emb_text_dim}. img dim: {emb_img_dim}"))

        # return the cosine difference between every image, concept tuple
        dot_products = image_embeddings @ concepts_features.T
    
        if apply_softmax:
            # the final step is to pass the cosine differences through the softmax layer. 
            labels = self.softmax_layer(dot_products)

            # make sure the labels are all positive, sum up to one along the concept axis and with the correct shape
            assert torch.all(labels >= 0), "Some of the final labels are negative"
            assert torch.allclose(torch.sum(labels, dim=1, dtype=torch.float), torch.ones(labels.size(dim=0), dtype=torch.float, device='cpu'), atol=10 ** -3)
            assert labels.shape == (num_images, num_concepts), f"The shape of the labels is not as expected. Expected: {(num_images, num_concepts)}. Found: {labels.shape}"

            return labels

        return dot_products.to(torch.float32)

