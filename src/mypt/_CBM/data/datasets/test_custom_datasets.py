"""
This script contains several tests to make sure any custom dataset (for the concepts) is correct: 
1. consistent: the same index returns always the same set (sample, concept_label, class_label)
2. one-to-one: no 2 indices possibly map to the same sample 
3. the concepts labels saved in torch are match the ones produced on the fly
"""

import os, sys
from pathlib import Path
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

import torch
import json
import itertools 
import shutil

from pathlib import Path
from tqdm import tqdm
from typing import Union
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

from . import conceptDataset as cd
from .uniformConceptDataset import UniformConceptDataset
from .binaryConceptDataset import BinaryConceptDataset

from .. import Clip_label_generation as clg
from ....code_utilities import directories_and_files as dirf
from ....backbones.resnetFeatureExtractor import ResNetFeatureExtractor


def test_concept_dataset(concepts_path: Union[Path, str], 
                         dataset_path: Union[str, Path]):
    # first read the concepts
    with open(concepts_path) as f:
        cs = json.load(f)

    concepts = list(set(itertools.chain(*cs.values())))

    # make sure the dataset path contains only directories
    dataset_path = dirf.process_path(dataset_path,
                            file_ok=False,
                            dir_ok=True,
                            # make sure that all the sub files are indeed directories
                            condition=lambda x: all([os.path.isdir(os.path.join(x, p))
                                for p in os.listdir(x)]),
                            error_message=f'The root directory is expected to have only directories as'
                            f' inner files.')
    
    g_cosine = clg.ClipLabelGenerator()
    g_dot = clg.ClipLabelGenerator(similarity_as_cosine=False)

    # use the transform of the ResnetFeature extactor
    transform = ResNetFeatureExtractor(num_layers=2).transform

    # TEST1 : consistency #################
    for g in [g_cosine, g_dot]:
        try:
            # let's make 2 copies of the directory
            a1, a2 = os.path.join(Path(dataset_path).parent, 'copy1'), os.path.join(Path(dataset_path).parent, 'copy2')

            # copy amazon_cbm to both directories
            dirf.copy_directories(src_dir=dataset_path, des_dir=a1, copy=True)
            dirf.copy_directories(src_dir=dataset_path, des_dir=a2, copy=True)

            d1 = cd.ConceptDataset(root=a1, 
                                label_generator=g,
                                concepts=concepts, 
                                label_generation_batch_size=10, 
                                image_transform=transform, 
                                remove_existing=True, 
                                debug=False)

            d2 = cd.ConceptDataset(root=a2,
                                label_generator=g,
                                concepts=concepts, 
                                label_generation_batch_size=10, 
                                image_transform=transform, 
                                remove_existing=True, 
                                debug=False)

            assert len(d1) == len(d2), "The 2 datasets have different lengths"

            for i in tqdm(range(len(d1))):
                img1, cl1, l1 = d1[i]
                img2, cl2, l2 = d2[i]

                assert (img1 == img2).all().item(), "the 2 images are different"
                assert (cl1 == cl2).all().item(), "The concept labels are different"
                assert l1 == l2, "The concept labels are different"

        finally: 
            shutil.rmtree(a1)
            shutil.rmtree(a2)

    # TEST2: one to one correspondance
    try: 
        a = os.path.join(Path(dataset_path).parent, 'copy')
        dirf.copy_directories(src_dir=dataset_path, des_dir=a, copy=True)
        d = cd.ConceptDataset(root=a, 
                            label_generator=None,
                            concepts=concepts, 
                            label_generation_batch_size=20, 
                            image_transform=transform, 
                            remove_existing=True, 
                            debug=False)
        
        for i in tqdm(range(len(d))):
            for j in range(i + 1, len(d)):
                p1, _  = d.idx2path(i)
                p2, _ = d.idx2path(j)
                # extract the class and the file name
                c1, f1 = os.path.basename(Path(p1).parent), os.path.basename(p1)
                c2, f2 = os.path.basename(Path(p2).parent), os.path.basename(p2)

                if c1 == c2 and f1 == f2:
                    raise ValueError(f"2 indices map to the same sample. found: {(c1, f1)} and {(c2, f2)}")

    finally: 
        shutil.rmtree(a)

def test_dataloader_randomness(concepts_path: Union[str, Path], 
                               dataset_path: Union[str, Path]): 
    
    seed_everything(69, workers=True)

    # first read the concepts
    with open(concepts_path) as f:
        cs = json.load(f)

    concepts = list(set(itertools.chain(*cs.values())))

    # make sure the dataset path contains only directories
    dataset_path = dirf.process_path(dataset_path,
                            file_ok=False,
                            dir_ok=True,
                            # make sure that all the sub files are indeed directories
                            condition=lambda x: all([os.path.isdir(os.path.join(x, p))
                                for p in os.listdir(x)]),
                            error_message=f'The root directory is expected to have only directories as'
                            f' inner files.')
    
    g_cosine = clg.ClipLabelGenerator()
    cs_cosine = g_cosine.encode_concepts(concepts)

    # use the transform of the ResnetFeature extactor
    transform = ResNetFeatureExtractor(num_layers=2).transform

    try: 
        a = os.path.join(Path(dataset_path).parent, 'copy')
        dirf.copy_directories(src_dir=dataset_path, des_dir=a, copy=True)
        d = cd.ConceptDataset(root=a, 
                            label_generator=None,
                            concepts=concepts, 
                            label_generation_batch_size=20, 
                            image_transform=transform, 
                            remove_existing=True, 
                            debug=False)
        # set the generator
        gen = torch.Generator()
        gen.manual_seed(69)

        dl = DataLoader(dataset=d, 
                        batch_size=1, 
                        shuffle=True, 
                        pin_memory=True, 
                        generator=gen)        
        iter = dl.__iter__()
        indices1 = []
        try:
            while True:
                indices1.extend(iter._next_index())
        except:
            pass 

        gen = torch.Generator()
        gen.manual_seed(69)

        dl = DataLoader(dataset=d, 
                        batch_size=1, 
                        shuffle=True, 
                        pin_memory=True, 
                        generator=gen)        
        iter = dl.__iter__()
        indices2 = []
        try:
            while True:
                indices2.extend(iter._next_index())
        except:
            pass 
        
        count = sum([i1 == i2 for i1, i2 in zip(indices1, indices2)])

    finally:
        shutil.rmtree(a)

    assert count == len(d)

def test_unifrom_concept_dataset(concepts_path:Union[str, Path], 
                                 dataset_path: Union[str, Path]):
    # let's start working on these ideas
    # first read the concepts
    with open(concepts_path) as f:
        concepts = json.load(f)

    # make sure the dataset path contains only directories
    dataset_path = dirf.process_path(dataset_path,
                            file_ok=False,
                            dir_ok=True,
                            # make sure that all the sub files are indeed directories
                            condition=lambda x: all([os.path.isdir(os.path.join(x, p))
                                for p in os.listdir(x)]),
                            error_message=f'The root directory is expected to have only directories as'
                            f' inner files.')
    
    g_cosine = clg.ClipLabelGenerator()
    g_dot = clg.ClipLabelGenerator(similarity_as_cosine=False)

    # use the transform of the ResnetFeature extactor
    transform = ResNetFeatureExtractor(num_layers=2).transform

    for g in [g_cosine, g_dot]:
        try:
            # let's make 2 copies of the directory
            a1, a2 = os.path.join(Path(dataset_path).parent, 'copy1'), os.path.join(Path(dataset_path).parent, 'copy2')

            # copy amazon_cbm to both directories
            dirf.copy_directories(src_dir=dataset_path, des_dir=a1, copy=True)
            dirf.copy_directories(src_dir=dataset_path, des_dir=a2, copy=True)

            d1 = UniformConceptDataset(root=a1, 
                                label_generator=g,
                                concepts=concepts, 
                                label_generation_batch_size=10, 
                                image_transform=transform, 
                                remove_existing=True, 
                                debug=False)

            d2 = UniformConceptDataset(root=a2,
                                label_generator=g,
                                concepts=concepts, 
                                label_generation_batch_size=10, 
                                image_transform=transform, 
                                remove_existing=True, 
                                debug=False)

            assert len(d1) == len(d2), "The 2 datasets have different lengths"

            for i in tqdm(range(len(d1))):
                img1, cl1, l1 = d1[i]
                img2, cl2, l2 = d2[i]

                assert (img1 == img2).all().item(), "the 2 images are different"
                assert (cl1 == cl2).all().item(), "The concept labels are different"
                assert l1 == l2, "The concept labels are different"

        finally: 
            shutil.rmtree(a1)
            shutil.rmtree(a2)

def test_binary_concept_dataset(concepts_path:Union[str, Path], 
                                 dataset_path: Union[str, Path]):
    with open(concepts_path) as f:
        concepts = json.load(f)

    # make sure the dataset path contains only directories
    dataset_path = dirf.process_path(dataset_path,
                            file_ok=False,
                            dir_ok=True,
                            # make sure that all the sub files are indeed directories
                            condition=lambda x: all([os.path.isdir(os.path.join(x, p))
                                for p in os.listdir(x)]),
                            error_message=f'The root directory is expected to have only directories as'
                            f' inner files.')
    # use the transform of the ResnetFeature extactor
    transform = ResNetFeatureExtractor(num_layers=2).transform
    
    for k in range(1, 3):
            try:
                # let's make 2 copies of the directory
                a1, a2 = os.path.join(Path(dataset_path).parent, 'copy1'), os.path.join(Path(dataset_path).parent, 'copy2')

                # copy amazon_cbm to both directories
                dirf.copy_directories(src_dir=dataset_path, des_dir=a1, copy=True)
                dirf.copy_directories(src_dir=dataset_path, des_dir=a2, copy=True)

                d1 = BinaryConceptDataset(root=a1, 
                                    concepts=concepts,
                                    similarity='cosine', 
                                    top_k=k,
                                    label_generation_batch_size=10, 
                                    image_transform=transform, 
                                    remove_existing=True)

                d2 = BinaryConceptDataset(root=a2,
                                    concepts=concepts,
                                    similarity='cosine', 
                                    top_k=k,
                                    label_generation_batch_size=10, 
                                    image_transform=transform, 
                                    remove_existing=True)

                assert len(d1) == len(d2), "The 2 datasets have different lengths"

                for i in tqdm(range(len(d1)), desc='testing items'):
                    img1, cl1, l1 = d1[i]
                    img2, cl2, l2 = d2[i]

                    assert (img1 == img2).all().item(), "the 2 images are different"
                    assert (cl1 == cl2).all().item(), "The concept labels are different"
                    assert l1 == l2, "The concept labels are different"

            finally: 
                shutil.rmtree(a1)
                shutil.rmtree(a2)

