"""
This script contains functionalities used to evaluate the encoding of the concepts
"""

import os, torch, random

import numpy as np
import pandas as pd

from functools import partial
from tqdm import tqdm
from pathlib import Path
from typing import Union
from torch.nn.functional import kl_div
from sklearn.metrics import jaccard_score

from ...code_utilities import directories_and_files as dirf
from ...code_utilities import pytorch_utilities as pu
from .Clip_label_generation import ClipLabelGenerator

def avg_max_pairwise_kl_distance(concepts_labels_dir: Union[str, Path], 
                               num_samples:int = None,
                               seed:int = 0,
                               verbose=False) -> np.ndarray:
    """ 
    This function computes the average distance and the maximum distance for each sample in the given concepts label directory    
    """

    concepts_labels_dir = dirf.process_path(concepts_labels_dir, 
                                          file_ok=False, 
                                          condition=lambda p: all([c.endswith('.pt') for c in os.listdir(p)]),
                                          error_message='The directory is expected to have only the concept labels saved as tensors'
                                          )
        
    cl = sorted(os.listdir(concepts_labels_dir))
    # avoid using the entire directory if the number of samples was specified 
    if num_samples is not None: 
        pu.seed_everything(seed=seed)
        # sample from the concept labels
        cl = random.sample(cl, min(num_samples,len(cl)))
    

    if verbose:
        all_data = torch.stack([torch.load(os.path.join(concepts_labels_dir, c)) for c in tqdm(cl, desc='loading all data in the concept label directory')])
    else:
        all_data = torch.stack([torch.load(os.path.join(concepts_labels_dir, c)) for c in cl])

    assert all_data.ndim == 2, f"the concepts labels are 1-dimensional !!. Found: {all_data.shape}"
    num_samples, dim = all_data.shape 

    avg_distances, max_distances = [], []
    for i in range(num_samples):
        sample_as_batch = torch.stack([all_data[i] for _ in range(num_samples)])
        assert sample_as_batch.shape == all_data.shape, f"the sample as batch does not have the correct dimensions {sample_as_batch.shape}"
        
        # calculate the kl divergence between the given sample and the rest of the class samples
        kl = (kl_div(input=torch.log(sample_as_batch), target=all_data, reduction="none") + 
              kl_div(input=torch.log(all_data), target=sample_as_batch, reduction="none")) 
        
        # sum up each row: the sum represents the distance between the two distributions
        kl = torch.sum(kl, dim=1)

        assert kl.shape == (num_samples, ), f"make sure the calculation of the kl distance is correct for each sample. Expected {(num_samples, )}. Found: {kl.shape}"

        sample_avg_distance, sample_max_distance = torch.mean(kl).item(), torch.max(kl).item()

        avg_distances.append(sample_avg_distance)
        max_distances.append(sample_max_distance)

    # convert both lists to numpy arrays
    avg_ds_np = np.asarray(avg_distances).reshape(-1, 1)
    max_ds_np = np.asarray(max_distances).reshape(-1, 1)
    
    res = np.concatenate([avg_ds_np, max_ds_np], axis=1)

    assert res.shape == (num_samples, 2), f"Make sure the final output represents the avg and max distance of each sample. Found shape: {res.shape}. Expected shape: {(num_samples, 2)}"

    return res

def avg_max_pairwise_binary_distance(concepts_labels_dir: Union[str, Path], 
                               num_samples:int = None,
                               seed:int = 0,
                               verbose=False) -> np.ndarray:
    """ 
    This function computes the average distance and the maximum distance for each sample in the given concepts label directory    
    """

    concepts_labels_dir = dirf.process_path(concepts_labels_dir, 
                                          file_ok=False, 
                                          condition=lambda p: all([c.endswith('.pt') for c in os.listdir(p)]),
                                          error_message='The directory is expected to have only the concept labels saved as tensors'
                                          )
        
    cl = os.listdir(concepts_labels_dir)
    # avoid using the entire directory if the number of samples was specified 
    if num_samples is not None: 
        pu.seed_everything(seed=seed)
        # sample from the concept labels
        cl = random.sample(cl, min(num_samples,len(cl)))
    
    # assert all([c.endswith('.pt') for c in os.listdir(concept_label_dir)]), "This directory contains files other than saved tensors"
    if verbose:
        all_data = torch.stack([torch.load(os.path.join(concepts_labels_dir, c)) for c in tqdm(cl, desc='loading all data in the concept label directory')])
    else:
        all_data = torch.stack([torch.load(os.path.join(concepts_labels_dir, c)) for c in sorted(os.listdir(concepts_labels_dir))])

    assert all_data.ndim == 2, f"the concepts labels are 1-dimensional !!. Found: {all_data.shape}"


    # assert all([c.endswith('.pt') for c in os.listdir(concept_label_dir)]), "This directory contains files other than saved tensors"
    all_data = torch.stack([torch.load(os.path.join(concepts_labels_dir, c)) for c in cl])

    assert all_data.ndim == 2, f"the concepts labels are 1-dimensional !!. Found: {all_data.shape}"
    num_samples, dim = all_data.shape 


    l1_distance = torch.nn.PairwiseDistance(p=1)
    avg_distances, max_distances = [], []
    
    loop = tqdm(range(num_samples), desc='iterating through the class samples') if verbose else range(num_samples)


    all_data_metric = all_data.T.numpy()

    for i in loop:
        sample_as_batch = torch.stack([all_data[i] for _ in range(num_samples)])
        assert sample_as_batch.shape == all_data.shape, f"the sample as batch does not have the correct dimensions {sample_as_batch.shape}"

        sample_distance_to_all = l1_distance.forward(sample_as_batch, all_data)
        # sample_distance_to_all = jaccard_score(sample_as_batch.T.numpy(), all_data_metric, average=None, zero_division=0)

        # sample_avg_distance, sample_max_distance = np.mean(sample_distance_to_all).item(), np.max(sample_distance_to_all).item()
        sample_avg_distance, sample_max_distance = torch.mean(sample_distance_to_all).item(), torch.max(sample_distance_to_all).item()

        avg_distances.append(sample_avg_distance)
        max_distances.append(sample_max_distance)

    # convert both lists to numpy arrays
    avg_ds_np = np.asarray(avg_distances).reshape(-1, 1)
    max_ds_np = np.asarray(max_distances).reshape(-1, 1)
    
    res = np.concatenate([avg_ds_np, max_ds_np], axis=1)

    assert res.shape == (num_samples, 2), f"Make sure the final output represents the avg and max distance of each sample. Found shape: {res.shape}. Expected shape: {(num_samples, 2)}"

    return res

def pairwise_inter_class_binary_distance(concepts_labels_dir1: Union[str, Path],
                           concepts_labels_dir2: Union[str, Path],
                           num_samples:int=None,
                           seed:int=0,
                           verbose=False) -> np.ndarray:
    # first of all, read the data from both directories
    concepts_labels_dir1 = dirf.process_path(concepts_labels_dir1, dir_ok=True,
                                           file_ok=False, 
                                           condition=lambda p: all([c.endswith('.pt') for c in os.listdir(p)]),
                                           error_message="This directory contains files other than saved tensors")

    concepts_labels_dir2 = dirf.process_path(concepts_labels_dir1, dir_ok=True,
                                           file_ok=False, 
                                           condition=lambda p: all([c.endswith('.pt') for c in os.listdir(p)]),
                                           error_message="This directory contains files other than saved tensors")
    
    cl1 = sorted(os.listdir(concepts_labels_dir1))
    # avoid using the entire directory if the number of samples was specified 
    if num_samples is not None: 
        pu.seed_everything(seed=seed)
        # sample from the concept labels
        cl1 = random.sample(cl1, min(num_samples,len(cl1)))

    cl2 = sorted(os.listdir(concepts_labels_dir2))
    # avoid using the entire directory if the number of samples was specified 
    if num_samples is not None: 
        pu.seed_everything(seed=seed)
        # sample from the concept labels
        cl2 = random.sample(cl2, min(num_samples,len(cl2)))

    
    c1_data = torch.stack([torch.load(os.path.join(concepts_labels_dir1, c)) for c in cl1])
    c2_data = torch.stack([torch.load(os.path.join(concepts_labels_dir2, c)) for c in cl2])
 
    assert c1_data.ndim == 2, f"the concepts labels are 1-dimensional !!. Found: {c1_data.shape}"
    assert c2_data.ndim == 2, f"the concepts labels are 1-dimensional !!. Found: {c2_data.shape}"

    n1, d1 = c1_data.shape
    n2, d2 = c2_data.shape

    assert d1 == d2, "The dimensions of concept labels must be the same"

    # make sure the data is binary
    assert sorted(torch.unique(c1_data).tolist()) == [0, 1], "The data of the first directory is not binary"
    assert sorted(torch.unique(c2_data).tolist()) == [0, 1], "The data of the second directory is not binary"

    min_cluster, max_cluster = (c1_data, c2_data) if n1 <= n2 else (c2_data, c1_data)
    min_n, max_n = len(min_cluster), len(max_cluster)

    l1_distance = torch.nn.PairwiseDistance(p=1, keepdim=True)
    
    distance_matrix = []

    loop = tqdm(range(min_n), desc='computing the inter class distance') if verbose else range(min_n)
    
    max_cluster_metric = max_cluster.T.numpy()
    for i in loop:
        sample_as_batch = torch.stack([min_cluster[i] for _ in range(max_n)])
        assert sample_as_batch.shape == max_cluster.shape, "the batch and the max cluster must be of the same shape"
        # calculate the distance between this one sample and all the samples in the other cluster
        sample_cluster_distance = l1_distance.forward(sample_as_batch, max_cluster).T.numpy()        
        # sample_cluster_distance = jaccard_score(sample_as_batch.T.numpy(), max_cluster_metric, average=None, zero_division=1).reshape((1, -1))
        # assert sample_cluster_distance.shape == (1, max_n), f"make sure the inter cluster distance is computed correctly. Expected: {(1, max_n)}. Found: {sample_cluster_distance.shape}"
        distance_matrix.append(sample_cluster_distance)


    distance_matrix = np.concatenate(distance_matrix, axis=0) 
    assert distance_matrix.shape == (min_n, max_n), f"Make sure the distance matrix is computed correctly. Expected: {(min_n, max_n)}. Found: {distance_matrix.shape}"
    return distance_matrix

def pairwise_inter_class_kl_distance(concepts_labels_dir1: Union[str, Path],
                           concepts_labels_dir2: Union[str, Path],
                           num_samples:int=None,
                           seed:int=0,
                           verbose=False) -> np.ndarray:

    # first of all, read the data from both directories
    concepts_labels_dir1 = dirf.process_path(concepts_labels_dir1, dir_ok=True,
                                           file_ok=False, 
                                           condition=lambda p: all([c.endswith('.pt') for c in os.listdir(p)]),
                                           error_message="This directory contains files other than saved tensors")

    concepts_labels_dir2 = dirf.process_path(concepts_labels_dir1, dir_ok=True,
                                           file_ok=False, 
                                           condition=lambda p: all([c.endswith('.pt') for c in os.listdir(p)]),
                                           error_message="This directory contains files other than saved tensors")
    
    cl1 = sorted(os.listdir(concepts_labels_dir1))
    # avoid using the entire directory if the number of samples was specified 
    if num_samples is not None: 
        pu.seed_everything(seed=seed)
        # sample from the concept labels
        cl1 = random.sample(cl1, min(num_samples,len(cl1)))

    cl2 = sorted(os.listdir(concepts_labels_dir2))
    # avoid using the entire directory if the number of samples was specified 
    if num_samples is not None: 
        pu.seed_everything(seed=seed)
        # sample from the concept labels
        cl2 = random.sample(cl2, min(num_samples,len(cl2)))

    
    c1_data = torch.stack([torch.load(os.path.join(concepts_labels_dir1, c)) for c in cl1])
    c2_data = torch.stack([torch.load(os.path.join(concepts_labels_dir2, c)) for c in cl2])
 
    assert c1_data.ndim == 2, f"the concepts labels are 1-dimensional !!. Found: {c1_data.shape}"
    assert c2_data.ndim == 2, f"the concepts labels are 1-dimensional !!. Found: {c2_data.shape}"

    n1, d1 = c1_data.shape
    n2, d2 = c2_data.shape

    assert d1 == d2, "The dimensions of concept labels must be the same"


    min_cluster, max_cluster = (c1_data, c2_data) if n1 <= n2 else (c2_data, c1_data)
    min_n, max_n = len(min_cluster), len(max_cluster)
   
    distance_matrix = []

    loop = tqdm(range(min_n), desc='computing the inter class distance') if verbose else range(min_n)
    for i in loop:
        sample_as_batch = torch.stack([min_cluster[i] for _ in range(max_n)])
        assert sample_as_batch.shape == max_cluster.shape, "the batch and the max cluster must be of the same shape"
        # calculate the distance between this one sample and all the samples in the other cluster
        
        sample_cluster_distance = (kl_div(input=torch.log(sample_as_batch), target=max_cluster, reduction="none") + 
              kl_div(input=torch.log(max_cluster), target=sample_as_batch, reduction="none"))

        sample_cluster_distance = torch.sum(sample_cluster_distance, dim=1, keepdim=True).T

        assert sample_cluster_distance.shape == (1, max_n), f"make sure the inter cluster distance is computed correctly. Expected: {(1, max_n)}. Found: {sample_cluster_distance.shape}"
        distance_matrix.append(sample_cluster_distance)


    distance_matrix = np.concatenate(distance_matrix, axis=0) 
    assert distance_matrix.shape == (min_n, max_n), f"Make sure the distance matrix is computed correctly. Expected: {(min_n, max_n)}. Found: {distance_matrix.shape}"
    return distance_matrix


# def evaluate_concepts_labels(directory: Union[str, Path],
#                              distance: str = 'KL',
#                              verbose: bool = False, 
#                              num_samples:int = 2 * 10 ** 3,
#                              seed: int = 0): 

#     if distance not in ['KL', 'binary']:
#         raise NotImplementedError(f"The function expects distance as {'KL' or 'binary'}. Found: {distance}")
    
#     # first extract the classes
#     cls = [c for c in os.listdir(directory) if c.endswith('_label')]
#     classes = [c[:c.find('_concept_label')] for c in cls]

#     intra_distance_function = (partial(avg_max_pairwise_kl_distance,verbose=verbose, num_samples=num_samples, seed=seed) 
#                                if distance == 'KL' else partial(avg_max_pairwise_binary_distance, verbose=verbose, num_samples=num_samples, seed=seed))
    
#     inter_distance_function = (partial(pairwise_inter_class_kl_distance, verbose=verbose, num_samples=num_samples, seed=seed) 
#                                if distance == 'KL' else partial(pairwise_inter_class_binary_distance, verbose=verbose, seed=seed, num_samples=num_samples))


#     intra_distances = {c[:c.find('_concept_label')]: intra_distance_function(os.path.join(directory, c)) 
#                        for c in tqdm(cls, desc='estimating intra class distances')}

#     # calculate the inter distances
#     inter_cluster_metrics = {}


#     for i in tqdm(range(len(classes)), desc='estimating the inter-class distances'):
#         loop2 = tqdm(range(len(classes)), desc=f'estimating the inter-class distances for the class: {classes[i]}') if verbose else range(len(classes))        
        
#         avg_dis_by_sample = intra_distances[classes[i]][:, [0]]
#         max_dis_by_sample = intra_distances[classes[i]][:, [1]]

#         # avg_dis_between_samples = np.mean(avg_dis_by_sample).item()
#         # max_dis_between_samples = np.mean(max_dis_by_sample).item()

#         for j in loop2:    
#             if j == i:
#                 continue

#             inter_cluster_distances = inter_distance_function(concepts_labels_dir1=os.path.join(directory, cls[i]),
#                                                               concepts_labels_dir2=os.path.join(directory, cls[j]))

#             avg_dis_by_sample_br = np.broadcast_to(avg_dis_by_sample, inter_cluster_distances.shape)
#             max_dis_by_sample_br = np.broadcast_to(max_dis_by_sample, inter_cluster_distances.shape)

#             _avg = np.mean(avg_dis_by_sample_br <= inter_cluster_distances, axis=1)
#             _max = np.mean(max_dis_by_sample_br <= inter_cluster_distances, axis=1)
            
#             # # sort the values values along each axis
#             # _avg = np.sort(_avg, axis=-1)
#             # # the idea is to consider the bottom and top k classes 
#             # avg_classes = _avg[:, :k]
#             # max_classes = _avg[:, -k:]

#             inter_cluster_metrics[(i, j)] = (np.mean(_avg).item(), np.mean(_max).item()) 
            
#     avg_metrics = np.zeros(shape=(len(classes), len(classes)))
#     max_metrics = np.zeros(shape=(len(classes), len(classes)))

#     for i in range(len(classes)):
#         for j in range(len(classes)):
#             if j == i:
#                 continue

#             res = inter_cluster_metrics[(i, j)]
#             a, m = res
#             avg_metrics[i, j] = m
#             avg_metrics[j, i] = m

#             max_metrics[i, j] = a
#             max_metrics[j, i] = a

#         max_metrics[i, i] = 0
#         avg_metrics[i, i] = 0

#     avg_metrics_df = pd.DataFrame(data=avg_metrics, index=classes, columns=classes)
#     max_metrics_df = pd.DataFrame(data=max_metrics, index=classes, columns=classes)

#     return avg_metrics_df, max_metrics_df

def evaluate_concepts_labels(directory: Union[str, Path],
                             distance: str = 'KL',
                             verbose: bool = False, 
                             num_samples:int = 2 * 10 ** 3,
                             seed: int = 0,
                             k: int = 5): 

    if distance not in ['KL', 'binary']:
        raise NotImplementedError(f"The function expects distance as {'KL' or 'binary'}. Found: {distance}")
    
    # first extract the classes
    cls = [c for c in os.listdir(directory) if c.endswith('_label')]
    classes = [c[:c.find('_concept_label')] for c in cls]

    intra_distance_function = (partial(avg_max_pairwise_kl_distance,verbose=verbose, num_samples=num_samples, seed=seed) 
                               if distance == 'KL' else partial(avg_max_pairwise_binary_distance, verbose=verbose, num_samples=num_samples, seed=seed))
    
    inter_distance_function = (partial(pairwise_inter_class_kl_distance, verbose=verbose, num_samples=num_samples, seed=seed) 
                               if distance == 'KL' else partial(pairwise_inter_class_binary_distance, verbose=verbose, seed=seed, num_samples=num_samples))


    intra_distances = {c[:c.find('_concept_label')]: intra_distance_function(os.path.join(directory, c)) 
                       for c in tqdm(cls, desc='estimating intra class distances')}

    # calculate the inter distances
    inter_cluster_metrics = {}


    clip_generator = ClipLabelGenerator(similarity_as_cosine=False)
    # encode each of the classes using the CLIP text encoder
    classes_encoded = clip_generator.encode_concepts(concepts=classes)
    
    # compute the similarities between the classes
    classes_sims = classes_encoded @ classes_encoded.T
    
    # choose the top classes for each class
    _, indices_close = torch.topk(input=classes_sims, k=k, dim=1, largest=True)
    _, indices_far = torch.topk(input=classes_sims, k=k, dim=1, largest=False)
    
    close_classes_per_cls = {classes[i]: indices_close[i, :].tolist() for i in range(len(classes))}
    far_classes_per_cls = {classes[i]: indices_far[i, :].tolist() for i in range(len(classes))}

    metrics_close = np.zeros(shape=(len(classes), k))
    metrics_far = np.zeros(shape=(len(classes), k))


    for i in tqdm(range(len(classes)), desc='estimating the inter-class distances'):
        # loop2 = tqdm(range(len(classes)), desc=f'estimating the inter-class distances for the class: {classes[i]}') if verbose else range(len(classes))        
        
        close_classes_i = close_classes_per_cls[classes[i]]
        far_classes_i = far_classes_per_cls[classes[i]]
        
        avg_dis_by_sample = intra_distances[classes[i]][:, [0]]

        for index, cci in enumerate(close_classes_i):
            inter_cluster_distances = inter_distance_function(concepts_labels_dir1=os.path.join(directory, cls[i]),
                                                              concepts_labels_dir2=os.path.join(directory, cls[cci]))

            avg_dis_by_sample_br = np.broadcast_to(avg_dis_by_sample, inter_cluster_distances.shape)
            _avg = np.mean(avg_dis_by_sample_br <= inter_cluster_distances, axis=1)    
            metrics_close[i][index] = np.mean(_avg).item()

        for index, fci in enumerate(far_classes_i):
            inter_cluster_distances = inter_distance_function(concepts_labels_dir1=os.path.join(directory, cls[i]),
                                                              concepts_labels_dir2=os.path.join(directory, cls[fci]))

            avg_dis_by_sample_br = np.broadcast_to(avg_dis_by_sample, inter_cluster_distances.shape)
            _avg = np.mean(avg_dis_by_sample_br <= inter_cluster_distances, axis=1)
            metrics_far[i][index] = np.mean(_avg).item()


    metrics_close_df = pd.DataFrame(data=metrics_close, index=classes, columns=[f'closest_class_{j}' for j in range(1, k + 1)])
    metrics_far_df = pd.DataFrame(data=metrics_far, index=classes, columns=[f'furthest_class_{j}' for j in range(1, k + 1)])

    return metrics_close_df, metrics_far_df
