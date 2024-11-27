"""
This script contains functionalities used to evaluate the encoding of the concepts
"""

import os
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial
from tqdm import tqdm
from pathlib import Path
from typing import Union, Dict, Tuple
from torch.nn.functional import kl_div
from sklearn.manifold import TSNE



from ...code_utilities import directories_and_files as dirf
from ...code_utilities import pytorch_utilities as pu

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
current=SCRIPT_DIR
while 'codebase' not in os.listdir(current):
    current = Path(current).parent

CODEBASE = os.path.join(current, 'codebase')
# the next step is to 


# the two functions below mainly compute the 
def KL_intra_cluster_distance(concept_label_dir: Union[str, Path], verbose=False) -> float:
    """This function calculates the KL divergence distance between the concepts labels of a given class (saved in a directory)
    Args:
        concept_label_dir (Union[str, Path]): a path to a directory of concept labels
    Returns:
        float: the average intra cluster distance
    """
    concept_label_dir = dirf.process_path(concept_label_dir, 
                                          file_ok=False, 
                                          condition=lambda p: all([c.endswith('.pt') for c in os.listdir(p)]),
                                          error_message='The directory is expected to have only the concept labels saved as tensors'
                                          )

    # assert all([c.endswith('.pt') for c in os.listdir(concept_label_dir)]), "This directory contains files other than saved tensors"
    all_data = torch.stack([torch.load(os.path.join(concept_label_dir, c)) for c in os.listdir(concept_label_dir)])

    assert all_data.ndim == 2, f"the concepts labels are 1-dimensional !!. Found: {all_data.shape}"
    num_samples, dim = all_data.shape 
    # difference
    difference = 0
    for i in range(num_samples):
        sample_as_batch = torch.stack([all_data[i] for _ in range(num_samples)])
        assert sample_as_batch.shape == all_data.shape, f"the sample as batch does not have the correct dimensions {sample_as_batch.shape}"
        
        # calculate the kl divergence between the given sample and the rest of the class samples
        kl = (kl_div(input=torch.log(sample_as_batch), target=all_data, reduction="none") + 
              kl_div(input=torch.log(all_data), target=sample_as_batch, reduction="none")) 
        difference += kl.sum().item() / 2

    # make sure to divide by the number of pairs: n * (n - 1) / 2
    num_pairs = (num_samples * (num_samples - 1)) // 2
    difference /= num_pairs
    return difference


def binary_intra_cluster_distance(concept_label_dir: Union[str, Path], verbose=False) -> float:
    """
    This function calculates the Jaccard similarity between the concept labels of a given class (saved in a directory)
    Args:
        concept_label_dir (Union[str, Path]): a path to a directory of concept labels
    Returns:
        float: the average intra-class distance
    """
    concept_label_dir = dirf.process_path(concept_label_dir, 
                                          file_ok=False, 
                                          condition=lambda p: all([c.endswith('.pt') for c in os.listdir(p)]),
                                          error_message='The directory is expected to have only the concept labels saved as tensors'
                                          )

    # assert all([c.endswith('.pt') for c in os.listdir(concept_label_dir)]), "This directory contains files other than saved tensors"
    if verbose:
        all_data = torch.stack([torch.load(os.path.join(concept_label_dir, c)) for c in tqdm(sorted(os.listdir(concept_label_dir)), desc='loading all data in the concept label directory')])
    else:
        all_data = torch.stack([torch.load(os.path.join(concept_label_dir, c)) for c in sorted(os.listdir(concept_label_dir))])

    assert all_data.ndim == 2, f"the concepts labels are 1-dimensional !!. Found: {all_data.shape}"

    # the function assume the data is binary
    assert sorted(torch.unique(all_data).tolist()) == [0, 1], "The binary_intra_cluster_distance function expects binary vectors." 

    num_samples, dim = all_data.shape 

    difference = 0
    l1_distance = torch.nn.PairwiseDistance(p=1)
    
    loop = tqdm(range(num_samples), desc='iterating through the class samples') if verbose else range(num_samples)
    for i in loop:
        sample_as_batch = torch.stack([all_data[i] for _ in range(num_samples)])
        assert sample_as_batch.shape == all_data.shape, f"the sample as batch does not have the correct dimensions {sample_as_batch.shape}"
        sample_distance_to_all = l1_distance.forward(sample_as_batch, all_data)
        difference += sample_distance_to_all.sum().item()

    # make sure to divide by the number of pairs: n * (n - 1) / 2
    num_pairs = (num_samples * (num_samples - 1))
    difference /= num_pairs
    return difference


def binary_inter_cluster_distance(concept_label_dir1: Union[str, Path],
                           concept_label_dir2: Union[str, Path], 
                           verbose=False) -> Tuple[float, float]:
    # first of all, read the data from both directories
    concept_label_dir1 = dirf.process_path(concept_label_dir1, dir_ok=True,
                                           file_ok=False, 
                                           condition=lambda p: all([c.endswith('.pt') for c in os.listdir(p)]),
                                           error_message="This directory contains files other than saved tensors")

    concept_label_dir2 = dirf.process_path(concept_label_dir2, dir_ok=True,
                                           file_ok=False, 
                                           condition=lambda p: all([c.endswith('.pt') for c in os.listdir(p)]),
                                           error_message="This directory contains files other than saved tensors")
    
    c1_data = torch.stack([torch.load(os.path.join(concept_label_dir1, c)) for c in os.listdir(concept_label_dir1)])
    c2_data = torch.stack([torch.load(os.path.join(concept_label_dir2, c)) for c in os.listdir(concept_label_dir2)])
 
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

    max_distance, average_distance = float('-inf'), 0 

    l1_distance = torch.nn.PairwiseDistance(p=1)
    for i in range(min_n):
        sample_as_batch = torch.stack([min_cluster[i] for _ in range(max_n)])
        assert sample_as_batch.shape == max_cluster.shape, "the batch and the max cluster must be of the same shape"

        # calculate the distance between this one sample and all the samples in the other cluster
        sample_cluster_distance = l1_distance.forward(sample_as_batch, max_cluster)        
        max_distance = max(torch.max(sample_cluster_distance).item(), max_distance)
        average_distance += sample_cluster_distance.sum().item()

    # average the distance: 
    average_distance /= (min_n * max_n)
    assert max_distance >= average_distance, "The maximum distance must be larger (or equal) than the average distance"
    return max_distance, average_distance


def KL_inter_cluster_distance(concept_label_dir1: Union[str, Path], 
                           concept_label_dir2: Union[str, Path], 
                           verbose:bool = False) -> Tuple[float, float]:
    """
    This function will compute the distance between two cluster of concept labels in 2 different ways, using the maximum distance between
    2 elements of the clusters and the average distance between 2 elements of the clusters

    Args:
        concept_label_dir1 (Union[str, Path]): a directory containing concept labels for the 1st class 
        concept_label_dir2 (Union[str, Path]): a directory containing concept labels for the 2nd class

    Returns:
        Tuple[float, float]: maximum distance, average distance
    """


    # first of all, read the data from both directories
    assert all([c.endswith('.pt') for c in os.listdir(concept_label_dir1)]), "This directory contains files other than saved tensors"
    assert all([c.endswith('.pt') for c in os.listdir(concept_label_dir2)]), "This directory contains files other than saved tensors"
    
    if verbose:
        c1_data = torch.stack([torch.load(os.path.join(concept_label_dir1, c)) for c in tqdm(sorted(os.listdir(concept_label_dir1)), desc='loading all samples from the first directory')])
        c2_data = torch.stack([torch.load(os.path.join(concept_label_dir2, c)) for c in tqdm(sorted(os.listdir(concept_label_dir2)), desc='loading all samples from the second directory')])
    else:
        c1_data = torch.stack([torch.load(os.path.join(concept_label_dir1, c)) for c in sorted(os.listdir(concept_label_dir1))])
        c2_data = torch.stack([torch.load(os.path.join(concept_label_dir2, c)) for c in sorted(os.listdir(concept_label_dir2))])


    assert c1_data.ndim == 2, f"the concepts labels are 1-dimensional !!. Found: {c1_data.shape}"
    assert c2_data.ndim == 2, f"the concepts labels are 1-dimensional !!. Found: {c2_data.shape}"
    
    n1, d1 = c1_data.shape
    n2, d2 = c2_data.shape

    assert d1 == d2, "The dimensions of concept labels must be the same"

    
    min_cluster, max_cluster = (c1_data, c2_data) if n1 <= n2 else (c2_data, c1_data)
    min_n, max_n = len(min_cluster), len(max_cluster)

    max_distance, average_distance = float('-inf'), 0 

    loop = tqdm(range(min_n), desc='iterating through the smaller directory') if verbose else range(min_n)
    for i in loop:
        sample_as_batch = torch.stack([min_cluster[i] for _ in range(max_n)])
        assert sample_as_batch.shape == max_cluster.shape, "the batch and the max cluster must be of the same shape"

        # calculate the distance between this one sample and all the samples in the other cluster
        sample_cluster_distance = (kl_div(input=torch.log(sample_as_batch), target=max_cluster, reduction='none') + 
                                   kl_div(input=torch.log(max_cluster), target=sample_as_batch, reduction='none')) / 2
        
        max_distance = max(torch.max(sample_cluster_distance.sum(dim=1)).item(), max_distance)
        average_distance += sample_cluster_distance.sum().item()

    # average the distance: 
    average_distance /= (min_n * max_n)

    assert max_distance >= average_distance, "The maximum distance must be larger (or equal) than the average distance"

    return max_distance, average_distance


def evaluate_encoding(directory: Union[str, Path],
                      distance: str = 'KL',
                      verbose: bool = False
                      ) -> Union[Dict, pd.DataFrame, pd.DataFrame]:
    if distance not in ['KL', 'binary']:
        raise NotImplementedError(f"The function expects distance as {'KL' or 'binary'}. Found: {distance}")
    
    # first extract the classes
    classes = [c for c in os.listdir(directory) if c.endswith('_label')]
    
    intra_distance_function = partial(KL_intra_cluster_distance,verbose=verbose) if distance == 'KL' else partial(binary_intra_cluster_distance, verbose=verbose)
    inter_distance_function = partial(KL_inter_cluster_distance, verbose=verbose) if distance == 'KL' else partial(binary_inter_cluster_distance, verbose=verbose)

    intra_distances = {c: intra_distance_function(os.path.join(directory, c)) for c in tqdm(classes, desc='estimating intra class distances')}

    # calculate the inter distances
    inter_distances = {}

    loop1 = tqdm(range(len(classes)), desc='estimating the inter-class distances') if verbose else range(len(classes))
    for i in loop1:
        loop2 = tqdm(range(i + 1, len(classes)), desc=f'estimating the inter-class distances for the class: {classes[i]}') if verbose else range(i + 1, len(classes))
        for j in loop2:
            inter_distances[(classes[i], classes[j])] = inter_distance_function(concept_label_dir1=os.path.join(directory, classes[i]),
                                                                            concept_label_dir2=os.path.join(directory, classes[j]))

    max_kl_distances = np.zeros(shape=(len(classes), len(classes)))
    avg_kl_distances = np.zeros(shape=(len(classes), len(classes)))

    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            res = inter_distances[(classes[i], classes[j])]
            m, a = res
            max_kl_distances[i, j] = m
            max_kl_distances[j, i] = m

            avg_kl_distances[i, j] = a
            avg_kl_distances[j, i] = a

        max_kl_distances[i, i] = 0
        avg_kl_distances[i, i] = 0

    max_kl_distances_df = pd.DataFrame(data=max_kl_distances, index=classes, columns=classes)
    avg_kl_distances_df = pd.DataFrame(data=avg_kl_distances, index=classes, columns=classes)

    return intra_distances, max_kl_distances_df, avg_kl_distances_df


def visualize_concepts_labels(directory: Union[str, Path],
                              vis_title: str,  
                              num_total_samples: int = 10 ** 4,
                              seed: int = 0,
                              ):
    # seed everything for reproducibility
    pu.seed_everything(seed=seed)

    # first extract the classes
    classes = [c for c in os.listdir(directory) if c.endswith('_label')]
    num_classes = len(classes)
    avg_per_cls = num_total_samples // num_classes

    all_samples = torch.cat([torch.stack([torch.load(os.path.join(directory, c, f)) for f in os.listdir(os.path.join(directory, c))[:avg_per_cls]], 
                                         dim=0)
                            for c in classes], 
                        dim=0)

    samples_per_cls = {}
    for i, c in enumerate(classes):
        samples_per_cls[i] = min(len(os.listdir(os.path.join(directory, c))), avg_per_cls)

    # create the TSNE class
    samples_embedded = TSNE(n_components=2,
        random_state=seed,
        learning_rate='auto', 
        init='pca' # init=pca to preserve the global structure as mentioned in the user guide: 
        # https://scikit-learn.org/stable/modules/manifold.html#t-sne
        ).fit_transform(all_samples.numpy())

    plt.figure(figsize=(20, 20))

    total = 0
    for i, num_samples in samples_per_cls.items():
        cls_embeddedings = samples_embedded[total: total + num_samples, :]
        total = total + num_samples
        plt.scatter(cls_embeddedings[:, 0], cls_embeddedings[:, 1], label=f'{classes[i]}')
        

    plt.legend()
    plt.xlabel("embeddings first dim")
    plt.ylabel("embeddings second dim")
    plt.title(vis_title)

    vis_dir = os.path.join(CODEBASE, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    plt.savefig(os.path.join(vis_dir, f'{vis_title}.png'))
    plt.show()

def _binary_vector_distribution(concept_label_dir: Union[str, Path]) -> np.ndarray:
    concept_label_dir = dirf.process_path(concept_label_dir, 
                                        file_ok=False, 
                                        condition=lambda p: all([c.endswith('.pt') for c in os.listdir(p)]),
                                        error_message='The directory is expected to have only the concept labels saved as tensors'
                                        )
    # TODO: rewrite the function to account for having a very large number of files in a given directory...
    all_data = torch.stack([torch.load(os.path.join(concept_label_dir, c)) for c in os.listdir(concept_label_dir)])
    res = all_data.mean(dim=0).numpy()
    return res

def evaluate_binary_vector_distribution(directory: Union[str, Path]) -> pd.DataFrame:
    # first extract the classes
    classes = [c for c in os.listdir(directory) if c.endswith('_label')]
    
    # compute the distributions
    distributions = torch.from_numpy(np.stack([_binary_vector_distribution(os.path.join(directory, c)) for c in classes]))

    n = len(classes)

    distances = torch.zeros(size=(n, n))

    for i in range(n):
        sample_as_batch = torch.stack([distributions[i] for _ in range(n)])
        sample_as_batch = distributions[[i], :]
        assert sample_as_batch.shape == (1, distributions.shape[1]), f"shape mismatch, found: {sample_as_batch.shape}. expected: {(1, distributions.shape[1])}"

        # calculate the distance between this one sample and all the samples in the other cluster
        sample_cluster_distance = (kl_div(input=torch.log(sample_as_batch), target=distributions, reduction='none') + 
                                   kl_div(input=torch.log(distributions), target=sample_as_batch, reduction='none')) / 2

        distances[i, :] = sample_cluster_distance

    res = sample_cluster_distance.numpy()
    return pd.DataFrame(data=res, columns=classes, index=classes)


# let's have a better mechanism to evaluate the concepts
def evaluate_concepts_labels(directory: Union[str, Path], 
                             distance: str = 'KL',
                             verbose: bool = False, 
                             num_samples:int = 2 * 10 ** 3,
                             seed: int = 0): 
    
    if distance not in ['KL', 'binary']:
        raise NotImplementedError(f"The function expects distance as {'KL' or 'binary'}. Found: {distance}")

    # first extract the classes
    cl_dirs = [c for c in os.listdir(directory) if c.endswith('_label')]
    classes = [c for c in os.listdir(directory) if not c.endswith('_label')]


    # determine the function used to compute 
    intra_distance_function = (partial(KL_intra_cluster_distance,verbose=verbose, num_samples=num_samples) 
                            if distance == 'KL' else partial(binary_intra_cluster_distance, verbose=verbose, num_samples=num_samples))
    
    inter_distance_function = (partial(KL_inter_cluster_distance, verbose=verbose, num_samples=num_samples) 
                               if distance == 'KL' else partial(binary_inter_cluster_distance, verbose=verbose, num_samples=num_samples))

    # compute the intra-class distances
    intra_distances = {c[:c.find("_concept_label")]: intra_distance_function(os.path.join(directory, c)) for c in tqdm(cl_dirs, desc='estimating intra class distances')}

    # instead of just computing the inter distances

    inter_distances = {}

    loop1 = tqdm(range(len(classes)), desc='estimating the inter-class distances') if verbose else range(len(classes))
    for i in loop1:
        loop2 = tqdm(range(i + 1, len(classes)), desc=f'estimating the inter-class distances for the class: {classes[i]}') if verbose else range(i + 1, len(classes))
        for j in loop2:
            inter_distances[(classes[i], classes[j])] = inter_distance_function(concept_label_dir1=os.path.join(directory, classes[i]),
                                                                            concept_label_dir2=os.path.join(directory, classes[j]))
