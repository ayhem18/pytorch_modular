import torchvision.transforms as tr 
import os, torch, pickle

from typing import Union, Optional, Dict
from pathlib import Path
from torch.utils.data import Dataset

from ...code_utilities import pytorch_utilities as pu
from ...code_utilities import directories_and_files as dirf 
from ...shortcuts import str2distance

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def _top_k_nearest_model(dataset: Dataset,
                        sample_processing: Union[callable, torch.nn.Module, tr.Compose],

                        model: torch.nn.Module,
                        
                        batch_size: int, 
                        k_neighbors: int,
                  
                        measure: Union[callable, torch.nn.Module] ,
                        measure_as_similarity:bool,
                        device:str,
                        ) -> Dict:
      results = {}

      for id1 in range(0, len(dataset), batch_size):
            
            try:
                  ref_batch = torch.stack([sample_processing(dataset.load_sample(i)) for i in range(id1, min(id1 + batch_size, len(dataset)))]).to(device)
            except AttributeError:
                  ref_batch = torch.stack([sample_processing(dataset[i]) for i in range(id1, min(id1 + batch_size, len(dataset)))]).to(device)

            # pass it through the model 
            _, ref_batch = model.forward(ref_batch)
            distances2ref = None
      
            for id2 in range(0, len(dataset), batch_size):
                  # ignore the same samples in the comparison process
                  if id2 == id1:
                        continue
                  try:
                        batch = torch.stack([sample_processing(dataset.load_sample(j)) for j in range(id2, min(id2 + batch_size, len(dataset)))]).to(device)
                  except AttributeError:
                        batch = torch.stack([sample_processing(dataset[j]) for j in range(id2, min(id2 + batch_size, len(dataset)))]).to(device)

                  # pass through the model
                  _, batch = model.forward(batch)

                  # compute the distance
                  batch_dis = measure(ref_batch, batch).cpu()

                  if distances2ref is None:
                        distances2ref = batch_dis
                  else:
                        distances2ref = torch.concat([distances2ref, batch_dis], dim=1)

            _, indices  = torch.topk(distances2ref, k=k_neighbors, dim=-1, largest=measure_as_similarity)
      
            # save the results for the current batch
            batch_res = {id1 + i : indices[i, :].squeeze().tolist() for i in range(len(indices))}
            results.update(batch_res)

      return results

def topk_nearest_model_ckpnt(results_directory: Union[str, Path],
                        
                             dataset: Dataset,
                             sample_processing: Union[callable, torch.nn.Module, tr.Compose],

                             model: torch.nn.Module,
                             model_ckpnt: Optional[Union[str, Path]],

                             batch_size: int, 
                             k_neighbors: int,
                        
                             measure: Union[str, callable, torch.nn.Module] = 'cosine_sim',
                             measure_as_similarity:bool=True
                        
                             ) -> Dict:
      
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

      if not (hasattr(dataset, 'load_sample') or hasattr(dataset, '__get_item__')):
            raise ValueError((f"For flexibility this function expects the dataset object to include a function 'load_sample' or '__get_item__' 
                              to load the sample, given its index"))          
   
      valid_ds = False
      if hasattr(dataset, '__get_item__'):
            item = dataset[0]
            try: 
                  processed_item = sample_processing(item)
            except Exception as e:
                  raise ValueError(f"The item returned by the dataset.__get_item__() raised an error: {e}")

            valid_ds = isinstance(processed_item, torch.Tensor)

      if not valid_ds and hasattr(dataset, 'load_sample'):
            item = dataset.load_sample(0)
            try: 
                  processed_item = sample_processing(item)
            except Exception as e:
                  raise ValueError(f"The item returned by the dataset.load_sample() raised an error: {e}")

            valid_ds = isinstance(processed_item, torch.Tensor)

      if not valid_ds: 
            raise ValueError(f"The processed item does not return a Pytorch Tensor. it returns an object of type:  {type(processed_item)}")


      ########################################## Process the distance measure ##########################################
      if isinstance(measure, str):
            # map the measure name to the correct callable
            measure = str2distance[measure]
      
      # make sure to map the measure callable to the device if it inherits the nn.Module
      if isinstance(measure, torch.nn.Module):
            measure = measure.to(device)

      # quick check for the type of the 'measure' callable output
      try:
            first_obj = dataset[0]
      except:
            first_obj = dataset.load_sample(0)

      measure_output = measure(torch.unsqueeze(first_obj, dim=0), torch.unsqueeze(first_obj, dim=0))
      if not isinstance(measure_output, (torch.Tensor)):
            raise ValueError(f"the output of the measure callable is expected to a pytorch Tensor !!!!. Found an output of type: {type(measure_output)}")

      
      ########################################## find the neighbors ##########################################
      results = _top_k_nearest_model(dataset=dataset,
                        sample_processing=sample_processing,
                        model=model,
                        
                        batch_size=batch_size, 
                        k_neighbors=k_neighbors,
                  
                        measure=measure,
                        measure_as_similarity=measure,
                        device=device
                        )

      ########################################## Save the results on the file system ##########################################
      
      if model_ckpnt is not None:
            res_file_name = os.path.splitext(os.path.basename(model_ckpnt))[0] + "_results.obj"
      else:
            res_file_name = "results.obj"

      res_path = dirf.process_path(results_directory, file_ok=False)
      res_path = os.path.join(results_directory, res_file_name)

      with open(res_path, 'wb') as f:
            pickle.dump(results, f)

      return results




