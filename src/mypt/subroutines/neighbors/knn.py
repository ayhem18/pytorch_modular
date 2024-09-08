"""
This script implements a K Nearest Neighbors Classifier Based on the outputs of a model
"""

import torch, warnings
import torchvision.transforms as tr
import numpy as np

from pathlib import Path
from typing import Union, Optional, Tuple, List
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter,defaultdict
from functools import partial

from ...code_utilities import pytorch_utilities as pu
from ...data.dataloaders.standard_dataloaders import initialize_val_dataloader
from ...shortcuts import str2distance


class KNN:
    __supported_str_measures = list(str2distance.keys())

    @classmethod
    def _batch_sizes(cls, batch_size: Union[str, float], dataset: Dataset) -> int:
        if isinstance(batch_size, float):
            assert 0 < batch_size <= 1.0, "passing a float batch size requires it to be a portion of the dataset size: between 0 and 1"
            batch_size = len(dataset) * batch_size

        if batch_size == 1:
            warnings.warn(f"a batch size of value 1 is ambiguous. The default behavior is to consider the entire dataset in this case")
            batch_size = len(dataset)
        
        return batch_size

    @classmethod
    def _measures(cls, 
                measure: Union[str, callable, torch.nn.Module] = 'cosine_sim',
                measure_init_kargs: dict = None,
                ):
        
        mik = measure_init_kargs if measure_init_kargs is not None else {}

        if isinstance(measure, str):
            if measure not in cls.__supported_str_measures:
                raise NotImplementedError(f"The current Knn implementation does not support {measure} out of the box. Please pass it as a callable with 2 arguments. supported measures: {cls.__supported_str_measures}")
            measure_str = measure

            try:
                    # the corresponding callable can be either a class or a function
                    # try the first option: a class
                    m = str2distance[measure_str](**mik) # this line of code should throw an error if the callable is a function and not a class
            except Exception as e1:
                    try:
                        m = str2distance[measure_str]
                    except Exception as e2:
                        raise ValueError((f"calling the measure: {measure_str} raised the following error: {str(e2)}."
                                            f"Check the measure_str and the initialization keyword arguments: {measure_init_kargs}"))          

        return m
        

    def _load_model(self) -> None:
        if self.ckpnt is None:
            # assume the model is ready for inference and return it as it is
            self.model.eval()
            self.model = self.model.to(self.inference_device)
            return 
        
        if isinstance(self.ckpnt, (Path, str)):
            try:
                self.model.load_state_dict(torch.load(self.ckpnt)['model_state_dict'])
            except KeyError:
                raise ValueError((f"the model_ckpnt dir was passed as a path (of type str, Path). In this case, load the model requires the ckpnt to include a key: 'model_state_dict'." 
                                 f"Please pass a callable to properly load the model that modifies the model in place"))
            
            self.model.eval()
            self.model.to(self.inference_device)
            return self.model

        # this leaves only the callable case
        self.ckpnt(self.model)
        self.model.eval()
        self.model = self.model.to(self.inference_device)


    def __init__(self, 
                train_ds: Dataset,
                train_ds_inference_batch_size:Union[int, float], 
                model: torch.nn.Module,
                
                process_item_ds: Optional[callable]=None,
                process_model_output: Optional[callable]=None,

                model_ckpnt: Optional[Union[str, Path, callable]]=None, 
                inference_device:Optional[str]=None,
                ) -> None:

        # the train dataset
        if not hasattr(train_ds, '__len__'):
            raise AttributeError(f"The KnnClassifier expects the train dataset to have the __len__ attribute")

        if len(train_ds) == 0:
            raise ValueError(f"Make sure not to pass an empty dataset. The dataset passed is of length: {len(train_ds)}")

        if not hasattr(train_ds, "__getitem__"):
            raise AttributeError(f"The KnnClassifier class requires the train dataset to have the __getitem__ attribute as it enables parallelism with dataloaders during inference ")

        self.train_ds = train_ds

        self.tbs = self._batch_sizes(batch_size=train_ds_inference_batch_size, dataset=self.train_ds)

        # the model
        self.model = model.cpu() # the model might have been in the gpu, move it to gpu first (the model should be moved to gpu only during inference)
        self.model.eval()


        # processing a sample before passing it to the model
        if process_item_ds is None:
            process_item_ds = lambda item: item
        
        self.process_item_ds = process_item_ds

        # processing the output of a model
        if process_model_output is None:
            process_model_output = lambda model, x: model(x)

        self.process_model_output = process_model_output

        if inference_device is None:
            inference_device = pu.get_default_device()

        self.inference_device = inference_device 
        
        self.ckpnt = model_ckpnt

        if callable(self.ckpnt):
            try:
                self._load_model()            
            except Exception as e:
                raise ValueError(f"The passed checkpoint raised the following error: {e}")
            
    
    def __build_candidates(self, 
                          train_dl: DataLoader, 
                          val_dl: DataLoader,
                          num_neighbors:int,
                          msr: callable,
                          measure_as_similarity:bool,
                          val_process_item_ds: callable,
                          ) -> Tuple[dict, dict]:

        nearest_neighbors_distances = {}
        nearest_neighbors_indices = {}

        ref_count = 0

        for _, ref_b in tqdm(enumerate(train_dl), desc="iterating over train_ds for inference"):
            # the model is already loaded and ready for inference
            ref_b = self.process_item_ds(ref_b)
            with torch.no_grad():
                ref_b_embs = self.process_model_output(self.model, ref_b.to(self.inference_device))

            inf_count = 0

            for _ , inf_b in enumerate(val_dl):           
                inf_b = val_process_item_ds(inf_b)
                with torch.no_grad():
                    inf_b_embs = self.process_model_output(self.model, inf_b.to(self.inference_device))

                    distances2ref = msr(inf_b_embs, ref_b_embs)

                # find the closest samples for the current batch
                values, local_indices  = torch.topk(distances2ref, 
                                                    k=min(num_neighbors, len(distances2ref[0])), 
                                                    dim=-1, 
                                                    largest=measure_as_similarity)
                # the indices should be convert to global indices with respect to the training dataset
                global_indices = (local_indices + ref_count).cpu().numpy()

                values = values.cpu().numpy()                             

                for i in range(len(inf_b)):

                    if inf_count + i not in nearest_neighbors_distances:

                        nearest_neighbors_distances[inf_count + i] = values[[i], :]
                        nearest_neighbors_indices[inf_count + i] = global_indices[[i], :] 

                    else:

                        nearest_neighbors_distances[inf_count + i] = np.concatenate([nearest_neighbors_distances[inf_count + i], 
                                                                                     values[[i], :]
                                                                                     ], # extracting the values as an np.array with shape [1, width] 
                                                                                     axis=1)
                        
                        nearest_neighbors_indices[inf_count + i] = np.concatenate([nearest_neighbors_indices[inf_count + i], 
                                                                                   global_indices[[i], :]
                                                                                   ], 
                                                                                   axis=1)


                # make sure to increase the 'inf_count' variable
                inf_count += len(inf_b)
        
            # assert len(nearest_neighbors_distances) == len(nearest_neighbors_indices)
            ref_count += len(ref_b)

        return nearest_neighbors_distances, nearest_neighbors_indices


    def __filter_candidates(self, 
                            nearest_neighbors_distances: dict[int, np.ndarray], 
                            nearest_neighbors_indices: dict[int, np.ndarray],
                            num_neighbors: int, 
                            measure_as_similarity:bool,
                            batch_size:int, 
                            num_samples:int
                            ) -> Tuple[np.ndarray, np.ndarray]:
        # first extract the candidates
        values_res = None
        indices_res = None

        for batch_index_start in range(0, num_samples, batch_size):
            vs = np.concatenate([nearest_neighbors_distances[batch_index_start + i] 
                                 for i in 
                                 range(min(batch_size, num_samples - batch_index_start))
                                ], 
                                axis=0)
            
            gis = np.concatenate([nearest_neighbors_indices[batch_index_start + i] for i in range(min(batch_size, num_samples - batch_index_start))], axis=0)

            batch_best_values, intermediate_indices  = torch.topk(torch.from_numpy(vs).to(self.inference_device), 
                                                                  k=num_neighbors, 
                                                                  dim=-1, 
                                                                  largest=measure_as_similarity)

            # batch_best_indices = gis[:, ]
            batch_best_indices = gis[[[i] for i in range(len(gis))], intermediate_indices.cpu().tolist()]

            batch_best_values = batch_best_values.cpu().numpy()

            if values_res is None:
                values_res = batch_best_values
                indices_res = batch_best_indices
            else:
                values_res = np.concatenate([values_res, batch_best_values], axis=0)
                indices_res = np.concatenate([indices_res, batch_best_indices], axis=0)

        # These functions are not supposed to be called directly, the model won't be moved back to the cpu...

        return values_res, indices_res


    def _find_neighbors(self, 
                        val_ds: DataLoader,
                        num_neighbors:int,
                        msr: callable,
                        measure_as_similarity:bool,
                        val_process_item_ds: callable,
                        val_batch_size:int,
                        num_workers:int=2
                        ):

        train_dl = initialize_val_dataloader(self.train_ds, 
                                             seed=0, 
                                             batch_size=self.tbs, 
                                             num_workers=num_workers,
                                             warning=False)
        
        val_dl = initialize_val_dataloader(val_ds, 
                                           seed=0, 
                                           batch_size=val_batch_size, 
                                           num_workers=num_workers,
                                           warning=False)

        if isinstance(msr, torch.nn.Module):
            msr = msr.to(self.inference_device)

        nearest_neighbors_distances, nearest_neighbors_indices = self.__build_candidates(train_dl=train_dl, 
                                                                                         val_dl=val_dl,
                                                                                         num_neighbors=num_neighbors, 
                                                                                         msr=msr, 
                                                                                         measure_as_similarity=measure_as_similarity, 
                                                                                         val_process_item_ds=val_process_item_ds,
                                                                                        )
        
        values_res, indices_res = self.__filter_candidates(nearest_neighbors_distances=nearest_neighbors_distances,
                                        nearest_neighbors_indices=nearest_neighbors_indices, 
                                        num_neighbors=num_neighbors,
                                        measure_as_similarity=measure_as_similarity,
                                        batch_size=val_batch_size,
                                        num_samples=len(val_ds))

        # move the model back to cpu
        self.model = self.model.to('cpu')
        if isinstance(msr, torch.nn.Module):
            msr = msr.cpu()

        return values_res, indices_res


    def predict(self, 
                val_ds: Dataset,
                inference_batch_size: Union[int, float],
                num_neighbors:int,

                measure: Union[str, callable, torch.nn.Module],
                measure_as_similarity,
                measure_init_kargs: dict = None,

                process_item_ds: Optional[callable]=None,
                process_model_output: Optional[callable]=None,
                num_workers:int=2,
                ) -> Tuple[np.ndarray, np.ndarray]:

        msr = self._measures(measure, measure_init_kargs)

        # process the batch size
        ibs = self._batch_sizes(inference_batch_size, val_ds)

        if process_item_ds is None:
            # use the training function
            process_item_ds = self.process_item_ds
        
        if process_model_output is None:
            # use the training function
            process_model_output = self.process_model_output

        self._load_model()
        
        res = self._find_neighbors( 
                        val_ds=val_ds,
                        num_neighbors=num_neighbors,
                        msr=msr,
                        measure_as_similarity=measure_as_similarity,
                        val_process_item_ds=process_item_ds,
                        val_batch_size=ibs,
                        num_workers=num_workers,
                        )

        # move the model back to cpu
        self.model = self.model.to('cpu')

        return res


class KnnClassifier(KNN):
    def __init__(self,
                train_ds: Dataset,
                train_ds_inference_batch_size:Union[int, float], 
                model: torch.nn.Module,
                
                process_item_ds: Optional[callable]=None, 
                process_item_ds_class: Optional[callable]=None,
                process_model_output: Optional[callable]=None,

                model_ckpnt: Optional[Union[str, Path, callable]]=None, 
                inference_device:Optional[str]=None):

        super().__init__(
                        train_ds=train_ds,
                        train_ds_inference_batch_size=train_ds_inference_batch_size, 
                        model=model,

                        process_item_ds=process_item_ds,
                        process_model_output=process_model_output,

                        model_ckpnt=model_ckpnt, 
                        inference_device=inference_device)
        
        if process_item_ds_class is None:
            warnings.warn("the 'process_item_ds_class' is not Passed. The class assumes that the dataset is a classification dataset where each item is a tuple of an image and a classification label")
            process_item_ds_class = lambda ds, index: ds[index][1] #  

        self.process_item_ds_class = process_item_ds_class


    def __predict_per_sample(self, 
                             sample_distances: Union[List[float]], 
                             sample_classes: Union[List[int]], 
                             measure_as_similarity: bool) -> int:

        counter = Counter(sample_classes)
        max_freq = max([v for _, v in counter.items()])
        modes = [k for k, v in counter.items() if v == max_freq]

        if len(modes) == 1:
            return modes[0]

        msr_dict = defaultdict(lambda : [])

        if measure_as_similarity:
            for index,c in enumerate(sample_classes):
                if c in modes:
                    msr_dict[c].append(sample_distances[index])
            label = max(msr_dict, key=lambda x: np.mean(msr_dict[x]))
            return label

        for index,c in enumerate(sample_classes):
            if c in modes:
                msr_dict[c].append(sample_distances[index])

        label = min(msr_dict, key=lambda x: np.mean(msr_dict[x]))
        return label


    def predict(
            self, 
            val_ds: Dataset,
            inference_batch_size: Union[int, float],
            num_neighbors:int,

            measure: Union[str, callable, torch.nn.Module],
            measure_as_similarity: bool,
            measure_init_kargs: dict = None,

            process_item_ds: Optional[callable]=None,
            process_model_output: Optional[callable]=None,
            num_workers:int=2) -> np.ndarray:
        # let's see how it goes
        distances_res, indices_res =super().predict(val_ds=val_ds,
                                                inference_batch_size=inference_batch_size,
                                                num_neighbors=num_neighbors,
                                                measure=measure,
                                                measure_as_similarity=measure_as_similarity,
                                                measure_init_kargs=measure_init_kargs,
                                                process_item_ds=process_item_ds,
                                                process_model_output=process_model_output,
                                                num_workers=num_workers                        
                                                )   

        classes = np.asarray([[self.process_item_ds_class(self.train_ds, index) for index in arr_indices] for arr_indices in indices_res])

        predictions = np.asarray([self.__predict_per_sample(distances_res[i, :].tolist(), 
                                                            classes[i, :].tolist(), 
                                                            measure_as_similarity) 
                                    for i in range(len(classes))
                                ])

        return predictions
