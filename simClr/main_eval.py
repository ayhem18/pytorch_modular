import os
import numpy as np

import torchvision.transforms as tr
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Union, Dict

from mypt.code_utilities import directories_and_files as dirf
from mypt.models.simClr.simClrModel import ResnetSimClr
from mypt.subroutines.neighbors import model_embs as me

from model_train.training import run_pipeline, OUTPUT_SHAPE 
from model_train.evaluating import evaluate_model, _set_data_classification_data
from mypt.subroutines.neighbors.knn import KNN

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

_OUTPUT_DIM = 128


# def visualize_neighbors(res: Union[str, Path, Dict], num_images: int = 5):
#     ds = STL10(root=os.path.join(SCRIPT_DIR ,'data', 'stl10', 'train'), 
#                transform=tr.ToTensor(), )

#     if isinstance(res, (str, Path)):
#         import pickle
#         with open(res, 'rb') as f:
#             res = pickle.load(f)        

#     for i, ilist in list(res.items())[:num_images]:
#         x, y = ds[i]
#         x = np.moveaxis(x.numpy(), 0,-1)

#         fig = plt.figure() 
#         fig.add_subplot(1, 1 + len(ilist), 1) 
#         plt.imshow(x)
#         plt.title("original image")

#         for rank, (index, measure) in enumerate(ilist):
#             n, _ = ds[index] 
#             n = np.moveaxis(n.numpy(), 0, -1)
#             fig.add_subplot(1, 1 + len(ilist), 2 + rank)
#             plt.imshow(n)
#             plt.title(f"nearest neighbor: {rank + 1}\nsimilarity: {round(measure, 4)}")

#         plt.show()



def evaluate(model, 
             model_ckpnt, 
             train_per_cls:int, 
             val_per_cls: int,

             ):
    train_folder = os.path.join(SCRIPT_DIR, 'data', 'food101', 'train')
    val_folder = os.path.join(SCRIPT_DIR, 'data', 'food101', 'val')

    train_ds, val_ds = _set_data_classification_data(train_folder, val_folder, OUTPUT_SHAPE[1:], train_per_cls, val_per_cls)
    
    knn = KNN(train_ds=train_ds, 
                train_ds_inference_batch_size=5000, # change this value depending on the computation resources
                process_model_output=lambda m, x: m(x)[0], # the forward call returns a tuple 
                model=model,
                model_ckpnt=model_ckpnt,
                process_item_ds=lambda i:i[0], # ignore the class label from the dataset during training
                ) 

    metrics, indices = knn.predict(val_ds, 
                                inference_batch_size=5000, # change this value depending on the computation resources  
                                num_neighbors=5, 
                                measure='cosine_sim',
                                measure_as_similarity=True
                                )

    neighbor_labels = np.asarray([
                                    [train_ds[index][1] for index in indices[i, :]]
                                    for i in range(len(indices))
                                   ]
                                  )

    # find the neighbors and stuff
    labels = np.asarray([val_ds[i][1] for i in range(len(val_ds))])
    

    res = {}
    for i in range(len(labels)):
        res[i] = {"label": int(labels[i]), 
                  "neighbor_labels": neighbor_labels[i, :].tolist(), 
                  "neighbor_indices": indices[i, :].tolist(), 
                  "neighbor_metrics": metrics[i, :].tolist()}

    results_directory = os.path.join(SCRIPT_DIR, 'eval_res')

    res_file_name = os.path.splitext(os.path.basename(model_ckpnt))[0] + "_results.obj"

    res_path = dirf.process_path(results_directory, file_ok=False)
    res_path = os.path.join(results_directory, res_file_name)

    import pickle
    with open(res_path, 'wb') as f:
        pickle.dump(res, f)

    return res


if __name__ == '__main__':
    model = ResnetSimClr(input_shape=OUTPUT_SHAPE,  
                         output_dim=_OUTPUT_DIM, 
                         num_fc_layers=4, 
                         freeze=False, 
                         architecture=101)

    ckpnt = os.path.join(SCRIPT_DIR, 'logs', 'train_logs', 'iteration_1', 'ckpnt_train_loss-4.5096_epoch-91.pt')

    evaluate(model=model, 
             model_ckpnt=ckpnt, 
             train_per_cls=100, 
             val_per_cls=100)

