import os
import numpy as np

import torchvision.transforms as tr
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Union, Dict
from torchvision.datasets import STL10

from mypt.code_utilities import pytorch_utilities as pu
from mypt.models.simClr.simClrModel import AlexnetSimClr
from model_train.training import run_pipeline


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# train_data_folder = os.path.join(SCRIPT_DIR, 'data', 'fashionMnist', 'train')

# model = AlexnetSimClr(input_shape=(3, 96, 96), output_dim=128, num_fc_layers=3, freeze=False)

# run_pipeline(model=model, 
#       train_data_folder=train_data_folder, 
#       val_data_folder=None,
#       learning_rates=[0.01, 0.1],
#       output_shape=(96, 96),
#       ckpnt_dir=os.path.join(SCRIPT_DIR, 'logs', 'train_logs'),
#       num_epochs=3, 
#       batch_size=128, 
#       temperature=0.5, 
#       seed=0, 
#       use_wandb=False)

from mypt.subroutines.topk_nearest_neighbors import model_embs as me
from model_train.ds_wrapper import STL10Wrapper
from model_train.training import _DEFAULT_DATA_AUGS, _UNIFORM_DATA_AUGS


def evaluate_ckpnt(model, ckpnt): 

    ds = STL10Wrapper(root_dir=os.path.join(SCRIPT_DIR ,'data', 'stl10', 'train'), 
                                train=True,
                                length=256, 
                                output_shape=(96,96),
                                augs_per_sample=2, 
                                sampled_data_augs=_DEFAULT_DATA_AUGS,
                                uniform_data_augs=_UNIFORM_DATA_AUGS)

    res = me.topk_nearest_model_ckpnt(results_directory=os.path.join(SCRIPT_DIR, 'eval_res'),
                                      dataset=ds,
                                      sample_processing=lambda x: x, # the dataset returns the object in the expected format already
                                      model=model,
                                      model_ckpnt=ckpnt,
                                      k_neighbors=5,
                                      batch_size=128)

def visualize_neighbors(res: Union[str, Path, Dict]):
    ds = STL10(root=os.path.join(SCRIPT_DIR ,'data', 'stl10', 'train'), 
               transform=tr.ToTensor(), )

    if isinstance(res, (str, Path)):
        import pickle
        with open(res, 'rb') as f:
            res = pickle.load(f)        

    for i, ilist in list(res.items())[:3]:
        x, y = ds[i]
        x = np.moveaxis(x.numpy(), 0,-1)

        fig = plt.figure() 
        fig.add_subplot(1, 1 + len(ilist), 1) 
        plt.imshow(x)
        plt.title("original image")

        for rank, index in enumerate(ilist):
            n, _ = ds[index] 
            n = np.moveaxis(n.numpy(), 0, -1)
            fig.add_subplot(1, 1 + len(ilist), 2 + rank)
            plt.imshow(n)
            plt.title(f"nearest neighbor: {rank + 1}")

        plt.show()
                

if __name__ == '__main__':
    # model = AlexnetSimClr(input_shape=(3, 96, 96), output_dim=128, num_fc_layers=6, freeze=False)
    # ckpnt = os.path.join(SCRIPT_DIR, 'logs', 'tune_logs', 'sweep_7', 'ckpnt_train_loss-5.0265_epoch-49.pt') 
    # evaluate_ckpnt(model, ckpnt)
    visualize_neighbors(res=os.path.join(SCRIPT_DIR, 'eval_res', 'ckpnt_train_loss-5.0265_epoch-49_results.obj'))
