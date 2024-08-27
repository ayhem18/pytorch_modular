import os
import numpy as np

import torchvision.transforms as tr
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Union, Dict
from torchvision.datasets import STL10

from mypt.code_utilities import pytorch_utilities as pu
from mypt.models.simClr.simClrModel import ResnetSimClr
from mypt.subroutines.neighbors import model_embs as me

from model_train.training import run_pipeline, _DEFAULT_DATA_AUGS, _UNIFORM_DATA_AUGS 
from model_train.ds_wrapper import Food101Wrapper 

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

_BATCH_SIZE = 16

def evaluate_ckpnt(model, ckpnt): 
    ds = STL10(root=os.path.join(SCRIPT_DIR ,'data', 'stl10', 'train'), 
               transform=tr.ToTensor(), )

    res = me.topk_nearest_model_ckpnt(results_directory=os.path.join(SCRIPT_DIR, 'eval_res'),
                                      dataset=ds,
                                      sample_processing=lambda x: x, # the dataset returns the object in the expected format already
                                      model=model,
                                      model_ckpnt=ckpnt,
                                      k_neighbors=5,
                                      batch_size=4)

def visualize_neighbors(res: Union[str, Path, Dict], num_images: int = 5):
    ds = STL10(root=os.path.join(SCRIPT_DIR ,'data', 'stl10', 'train'), 
               transform=tr.ToTensor(), )

    if isinstance(res, (str, Path)):
        import pickle
        with open(res, 'rb') as f:
            res = pickle.load(f)        

    for i, ilist in list(res.items())[:num_images]:
        x, y = ds[i]
        x = np.moveaxis(x.numpy(), 0,-1)

        fig = plt.figure() 
        fig.add_subplot(1, 1 + len(ilist), 1) 
        plt.imshow(x)
        plt.title("original image")

        for rank, (index, measure) in enumerate(ilist):
            n, _ = ds[index] 
            n = np.moveaxis(n.numpy(), 0, -1)
            fig.add_subplot(1, 1 + len(ilist), 2 + rank)
            plt.imshow(n)
            plt.title(f"nearest neighbor: {rank + 1}\nsimilarity: {round(measure, 4)}")

        plt.show()
                
def train_main(model):
    train_data_folder = os.path.join(SCRIPT_DIR, 'data', 'food101', 'train')

    ckpnt_dir_parent = os.path.join(SCRIPT_DIR, 'logs', 'train_logs') 
    return run_pipeline(model=model, 
        train_data_folder=train_data_folder, 
        val_data_folder=None,
        # initial_lr=0.01,
        learning_rates=0.01,
        output_shape=(200, 200),
        ckpnt_dir=os.path.join(ckpnt_dir_parent, f'iteration_{len(os.listdir(ckpnt_dir_parent)) + 1}'),
        num_epochs=64, 
        batch_size=_BATCH_SIZE, 
        temperature=0.5, 
        seed=0, 
        use_wandb=True,
        batch_stats=True)



if __name__ == '__main__':
    data = os.path.join(SCRIPT_DIR, 'data', 'food101', 'train')

    # model = ResnetSimClr(input_shape=(3, 200, 200), output_dim=128, num_fc_layers=4, freeze=False)
    # train_main(model)
    # ckpnt = os.path.join(SCRIPT_DIR, 'logs', 'train_logs', 'ckpnt_train_loss-4.0917_epoch-123.pt')
    # evaluate_ckpnt(model, ckpnt)
    # visualize_neighbors(res=os.path.join(SCRIPT_DIR, 'eval_res', 'ckpnt_train_loss-4.0917_epoch-123_results.obj'), num_images=25)


    # ds = Food101Wrapper(root_dir=data, 
    #                     output_shape=224,
    #                     augs_per_sample=2, 
    #                     sampled_data_augs=_DEFAULT_DATA_AUGS,
    #                     uniform_data_augs=_UNIFORM_DATA_AUGS,
    #                     train=True, 
    #                     samples_per_cls=1000)    

    # from bisect import bisect
    # from collections import Counter

    # occs = Counter()

    # for i in range(3150):
    #     stop_points = sorted(list(ds.samples_per_cls_map.keys()))

    #     index = bisect(stop_points, i)
        
    #     if index == len(stop_points):
    #         index -= 1

    #     sp = stop_points[index]

    #     _, c = ds._ds[ds.samples_per_cls_map[sp] + i - sp]

    #     occs[c] += 1
    
    # print(occs)
