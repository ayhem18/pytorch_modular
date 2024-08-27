import torchvision.transforms as tr 
import os, torch, pickle

from typing import Union, Optional, Dict, List, Tuple
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

from ...code_utilities import pytorch_utilities as pu
from ...code_utilities import directories_and_files as dirf 
from ...shortcuts import str2distance

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def _top_k_nearest_model(dataset: Dataset,
                        get_sample_callable: callable,
                        
                        model: torch.nn.Module,
                        
                        batch_size: int, 
                        k_neighbors: int,
                  
                        measure: Union[callable, torch.nn.Module] ,
                        measure_as_similarity:bool,
                        device:str,
                        ) -> Dict[int, Tuple[List[int], List[int]]]:
      results = {}

      for id1 in tqdm(range(0, len(dataset), batch_size), desc='finding the nearest neighbors'):
            ref_batch = torch.stack([get_sample_callable(dataset, i) for i in range(id1, min(id1 + batch_size, len(dataset)))]).to(device)

            # pass it through the model 
            try:
                  ref_batch = model(ref_batch)  
            except:
                  raise ValueError(f"This function expects the __call__ function to return the final output of the model: only one tensor")

            distances2ref = None
      
            for id2 in range(0, len(dataset), batch_size):

                  batch = torch.stack([get_sample_callable(dataset, j) for j in range(id2, min(id2 + batch_size, len(dataset)))]).to(device)

                  # pass through the model (no need for the try catch clause, as the first clause would raise an error in case of unexpected input !!)
                  batch = model(batch)

                  # compute the distance
                  batch_dis = measure(ref_batch, batch).cpu()

                  if distances2ref is None:
                        distances2ref = batch_dis
                  else:
                        distances2ref = torch.concat([distances2ref, batch_dis], dim=1)

            values, indices  = torch.topk(distances2ref, k=k_neighbors + 1, dim=-1, largest=measure_as_similarity)

            # make sure to ignore the first element at each row as it should be the element itself
            indices = indices[:, 1:]
            values = values[:, 1:]

            # save the results for the current batch
            batch_res = {id1 + i : list(zip(indices[i, :].squeeze().tolist(), values[i, :].squeeze().tolist())) # associating each index with a list of tuples (index, measure) 
                         for i in range(len(indices))}
            results.update(batch_res)

      return results

def topk_nearest_model_ckpnt(
                        results_directory: Union[str, Path],

                        dataset: Dataset,
                        sample_processing: Union[callable, torch.nn.Module, tr.Compose],

                        model: torch.nn.Module,
                        model_ckpnt: Optional[Union[str, Path]],

                        batch_size: int, 
                        k_neighbors: int,

                        measure: Union[str, callable, torch.nn.Module] = 'cosine_sim',
                        measure_as_similarity:bool=True,
                        measure_init_kargs: Dict = None,

                        res_file_name: str = None) -> Dict:
      ########################################## set the model ##########################################

      device = pu.get_default_device()

      # load the model if needed 
      if model_ckpnt is not None:
            model.load_state_dict(torch.load(model_ckpnt)['model_state_dict'])
            # otherwise consider the model ready to use
            
      # set the model to the validation model and then to the device
      model.eval()
      model = model.to(device)

      
      ########################################## process the Dataset Object  ##########################################

      if not (hasattr(dataset, 'load_sample') or hasattr(dataset, '__getitem__')):
            raise ValueError((f"For flexibility this function expects the dataset object to include a function 'load_sample' or 'dataset__'"
                              "to load the sample, given its index"))          
   
      valid_ds = False
      processed_item = None

      get_sample_callable = None

      if hasattr(dataset, '__getitem__'):
            item = dataset[0]
            try: 
                  processed_item = sample_processing(item)
            except Exception as e:
                  raise ValueError(f"The item returned by the dataset.__getitem__() raised an error: {e}")

            # the case of tuple and list types were added after using a standard image classification dataset
            if isinstance(processed_item, (Tuple, List)):
                  valid_ds = isinstance(processed_item[0], torch.Tensor)                   
                  get_sample_callable = lambda d, index: sample_processing(d[index][0])
            else:
                  valid_ds = isinstance(processed_item, torch.Tensor)
                  get_sample_callable = lambda d, index: sample_processing(d[index])


      if not valid_ds and hasattr(dataset, 'load_sample'):
            item = dataset.load_sample(0)
            try: 
                  processed_item = sample_processing(item)
            except Exception as e:
                  raise ValueError(f"The item returned by the dataset.load_sample() raised an error: {e}")

            if isinstance(processed_item, (Tuple, List)):
                  valid_ds = isinstance(processed_item[0], torch.Tensor)                   
                  get_sample_callable = lambda d, index: sample_processing(d.load_sample(index)[0]) 
            else:
                  valid_ds = isinstance(processed_item, torch.Tensor)
                  get_sample_callable = lambda d, index: sample_processing(d.load_sample(index)) 

      if not valid_ds: 
            raise ValueError(f"The processed item does not return a Pytorch Tensor. it returns an object of type:  {type(processed_item)}")


      ########################################## Process the distance measure ##########################################
      
      measure_init_kargs = measure_init_kargs if measure_init_kargs is not None else {}

      if isinstance(measure, str):
            measure_str = measure
            try:
                  # the corresponding callable can be either a class or a function
                  # try the first option: a class
                  measure = str2distance[measure_str](**measure_init_kargs) # this line of code should throw an error if the callable is a function and not a class
            except Exception as e1:
                  try:
                        measure = str2distance[measure_str]
                  except Exception as e2:
                        raise ValueError((f"calling the measure: {measure_str} raised the following error: {str(e2)}."
                                          f"Check the measure_str and the initialization keyword arguments: {measure_init_kargs}"))          


      # make sure to map the measure callable to the device if it inherits the nn.Module
      if isinstance(measure, torch.nn.Module):
            measure = measure.to(device)

      # quick check for the type of the 'measure' callable output
      first_obj = torch.stack([get_sample_callable(dataset, 0), get_sample_callable(dataset, 0)], dim=0).to(device)

      # pass two single-item batches to the model and then pass the model's outputs to the loss function
      x, y = model(first_obj), model(first_obj)

      measure_output = measure(x,y)
      if not isinstance(measure_output, (torch.Tensor)):
            raise ValueError(f"the output of the measure callable is expected to a pytorch Tensor !!!!. Found an output of type: {type(measure_output)}")

      
      ########################################## find the neighbors ##########################################
      results = _top_k_nearest_model(dataset=dataset,
                        get_sample_callable=get_sample_callable,
                        model=model,
                        
                        batch_size=batch_size, 
                        k_neighbors=k_neighbors,
                  
                        measure=measure,
                        measure_as_similarity=measure_as_similarity,
                        device=device
                        )

      ########################################## Save the results on the file system ##########################################
      
      if res_file_name is None:
            if model_ckpnt is not None:
                  res_file_name = os.path.splitext(os.path.basename(model_ckpnt))[0] + "_results.obj"
            else:
                  res_file_name = "results.obj"

      res_path = dirf.process_path(results_directory, file_ok=False)
      res_path = os.path.join(results_directory, res_file_name)

      with open(res_path, 'wb') as f:
            pickle.dump(results, f)

      return results
