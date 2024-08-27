"""
This script implements a K Nearest Neighbors Classifier Based on the outputs of a model
"""

import torch, warnings
import torchvision.transforms as tr
import numpy as np

from pathlib import Path
from typing import Union, Optional, Tuple
from torch.utils.data import Dataset
from tqdm import tqdm

from ...code_utilities import pytorch_utilities as pu
from ...data.dataloaders.standard_dataloaders import initialize_val_dataloader
from ...shortcuts import str2distance


class KnnClassifier:
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
        

    def __init__(self, 
                train_ds: Dataset,
                train_ds_inference_batch_size:Union[int, float], 
                model: torch.nn.Module,
                
                process_sample_ds: Optional[callable]=None,
                process_model_output: Optional[callable]=None,

                model_ckpnt: Optional[Union[str, Path, callable]]=None, 
                inference_device:Optional[str]=None,
                debug_mode: bool = False
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
        self.model = model.cpu() # move the model to cpuat first
        self.model.eval()


        # processing a sample before passing it to the model
        if process_sample_ds is None:
            process_sample_ds = tr.ToTensor()
        
        self.process_sample_ds = process_sample_ds

        # processing the output of a model
        if process_model_output is None:
            process_model_output = lambda model, x: model(x)

        self.process_model_output = process_model_output

        if inference_device is None:
            inference_device = pu.get_default_device()

        self.inference_device = inference_device 
        
        self.ckpnt = model_ckpnt

        self.debug_mode = debug_mode

    def _load_model(self) -> None:
        if self.ckpnt is None:
            # assume the model is ready for inference and return it as it is
            self.model.eval()
            return 
        
        if isinstance(self.ckpnt, (Path, str)):
            try:
                self.model.load_state_dict(torch.load(self.ckpnt)['model_state_dict'])
            except KeyError:
                raise ValueError((f"the model_ckpnt dir was passed as a path (of type str, Path). In this case, load the model requires the ckpnt to include a key: 'model_state_dict'." 
                                 f"Please pass a callable to properly load the model that modifies the model in place"))

            self.model.eval()
            return self.model

        # this leaves only the callable case
        self.ckpnt(self.model)
        self.model.eval()
    
    def predict(self, 
                val_ds: Dataset,
                inference_batch_size: Union[int, float],
                num_neighbors:int,

                measure: Union[str, callable, torch.nn.Module] = 'cosine_sim',
                measure_as_similarity:bool=True,
                measure_init_kargs: dict = None,

                process_sample_ds: Optional[callable]=None,
                process_model_output: Optional[callable]=None,
                num_workers:int=2,
                ) -> Tuple[np.ndarray, np.ndarray]:
        # process the batch size
        ifs = self._batch_sizes(inference_batch_size)
        
        if process_sample_ds is None:
            process_sample_ds = tr.ToTensor()
        
        # processing the output of a model
        if process_model_output is None:
            process_model_output = lambda model, x: model(x)


        train_dl = initialize_val_dataloader(val_ds, 
                                             seed=0, 
                                             batch_size=ifs, 
                                             num_workers=num_workers,
                                             warning=False)
        
        val_dl = initialize_val_dataloader(val_ds, 
                                           seed=0, 
                                           batch_size=ifs, 
                                           num_workers=num_workers,
                                           warning=False)

        msr = self._measures(measure, measure_init_kargs)

        self._load_model()

        nearest_neighbors_distances = {}
        nearest_neighbors_indices = {}


        self.model = self.model.to(self.inference_device)

        ref_count = 0

        for _, ref_b in tqdm(enumerate(train_dl), desc="iterating over train_ds for inference"):
            # the model is already loaded and ready for inference
            with torch.no_grad():
                ref_b_embs = self.process_model_output(self.model, self.process_sample_ds(ref_b).to(self.inference_device)).cpu()

            inf_count = 0

            for _ , inf_b in enumerate(val_dl):           
                # process samples
                inf_b = process_sample_ds(inf_b)

                with torch.no_grad():
                    inf_b_embs = self.process_model_output(self.model, self.process_sample_ds(inf_b).to(self.inference_device)).cpu()

                    distances2ref = msr(inf_b_embs, ref_b_embs)

                # find the closest samples for the current batch
                values, local_indices  = torch.topk(distances2ref, k=num_neighbors, dim=-1, largest=measure_as_similarity)
                # the indices should be convert to global indices with respect to the training dataset
                global_indices = (local_indices + ref_count).numpy()

                values = values.numpy()                             

                for i in range(len(inf_b)):

                    if inf_count not in nearest_neighbors_distances:

                        nearest_neighbors_distances[inf_count + i] = values[[i], :]
                        nearest_neighbors_indices[inf_count + i] = global_indices[[i], :] 

                    else:

                        nearest_neighbors_distances[inf_count + i] = np.concatenate([nearest_neighbors_distances[inf_count + i], 
                                                                                     values[[i], :]
                                                                                     ], # extracting the values as an np.array with shape [1, width] 
                                                                                     axis=1)
                        
                        nearest_neighbors_indices[inf_count + i] = np.concatenate([nearest_neighbors_distances[inf_count + i], 
                                                                                   global_indices[[i], :]
                                                                                   ], 
                                                                                   axis=1)
                        

            inf_count += len(inf_b)

            ref_count += len(ref_b)

        assert len(nearest_neighbors_distances) == len(nearest_neighbors_indices) == len(val_ds), "The length of the mappings do not match the size of the dataset !!"


        values_res = None
        indices_res = None

        for batch_index_start in range(0, len(val_ds), ifs):
            vs = np.concatenate([nearest_neighbors_distances[batch_index_start + i] 
                                 for i in 
                                 range(min(ifs, len(val_ds) - batch_index_start))
                                ], 
                                axis=0)
            
            gis = np.concatenate([nearest_neighbors_indices[batch_index_start + i] for i in range(min(ifs, len(val_ds) - batch_index_start))], axis=0)

            batch_best_values, intermediate_indices  = torch.topk(torch.from_numpy(vs), k=num_neighbors, dim=-1, largest=measure_as_similarity)
            batch_best_indices = gis[intermediate_indices.numpy()]

            batch_best_values = batch_best_values.numpy()

            if values_res is None:
                values_res = batch_best_values
                indices_res = batch_best_indices
            else:
                values_res = np.concatenate([values_res, batch_best_values], axis=0)
                indices_res = np.concatenate([indices_res, batch_best_indices], axis=0)

        return values_res, indices_res
        