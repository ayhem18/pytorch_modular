"""
This script contains the implementation of the main training process 
"""

import os

from mypt.code_utilities import directories_and_files as dirf, pytorch_utilities as pu
from model_train_pl.train import ResnetSimClrWrapper, evaluate_augmentations, calculate_weights, train_simClr
from model_train_pl.constants import TRACK_PROJECT_NAME
from model_train_pl.set_ds import _DEFAULT_DATA_AUGS

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def train_main():
    dataset_name = 'imagenette'
    train_data_folder = os.path.join(SCRIPT_DIR, 'data', dataset_name, 'train')
    val_data_folder = os.path.join(SCRIPT_DIR, 'data', dataset_name, 'val')
    
    ckpnt_dir_parent = dirf.process_path(os.path.join(SCRIPT_DIR, 'logs', 'train_logs'), 
                                         dir_ok=True, 
                                         file_ok=False)

    # delete any empty folders in the 'train_logs' folder
    dirf.clear_directory(ckpnt_dir_parent, condition=lambda x: os.path.isdir(x) and len(os.listdir(x)) == 0)
    n = len(os.listdir(ckpnt_dir_parent))

    run_name = f'{TRACK_PROJECT_NAME}_iteration_{n + 1}'

    train_simClr(
                train_data_folder=train_data_folder,
                val_data_folder=val_data_folder,
                dataset=dataset_name,
                log_dir=os.path.join(ckpnt_dir_parent, f'{dataset_name}_iteration_{n + 1}'),
                num_epochs=5, 
                train_batch_size=512,
                # set to 512 since the dataset size will be just 70 * 10 = 700 and larger values will raise an exception
                # at the point of dataloader initialization
                val_batch_size=512,  
                seed=0, 
                use_logging=True,
                num_warmup_epochs=0,
                val_per_epoch=2,
                num_train_samples_per_cls=70,
                num_val_samples_per_cls=70,
                run_name=run_name,     
                debug_augmentations=_DEFAULT_DATA_AUGS                   
                )

if __name__ == '__main__':
    train_main()   

    # ckpnt_dir_parent = dirf.process_path(os.path.join(SCRIPT_DIR, 'logs', 'train_logs'), 
    #                                      dir_ok=True, 
    #                                      file_ok=False)

    # # delete any empty folders in the 'train_logs' folder
    # dirf.clear_directory(ckpnt_dir_parent, condition=lambda x: os.path.isdir(x) and len(os.listdir(x)) == 0)
    # n = len(os.listdir(ckpnt_dir_parent))
    # log_dir=os.path.join(ckpnt_dir_parent, f'imagenette_iteration_{n}')

    # ckpnt = os.path.join(log_dir, 'val-epoch=19-val_epoch_loss=6.413670.ckpt')

    # wrapper = ResnetSimClrWrapper.load_from_checkpoint(ckpnt)

    # import torch
    # aug_scores =torch.load(ckpnt)['augmentation_scores']
    # augs_per_difficulty = evaluate_augmentations(
    #                                     augmentations=_DEFAULT_DATA_AUGS,
    #                                     get_augmentation_name=pu.get_augmentation_name, 
    #                                     augmentation_scores=aug_scores,
    #                                     num_categories=3)

    # from model_train_pl.set_ds import _set_imagenette_ds_debug

    # train_data_folder = os.path.join(SCRIPT_DIR, 'data', 'imagenette', 'train')

    # dataloader = _set_imagenette_ds_debug(train_data_folder=train_data_folder, 
    #                                     output_shape=(200, 200), 
    #                                     batch_size=1024, 
    #                                     num_train_samples_per_cls=None)
    
    # sample_weights = calculate_weights(augmentations_per_difficulty=augs_per_difficulty, 
    #                                    dataloader=dataloader, 
    #                                    model=wrapper.model, 
    #                                    process_model_output=lambda x: x[1])




    # from torchvision.datasets import MNIST
    # import torchvision.transforms as tr
    # data_path = os.path.join(SCRIPT_DIR, 'data', 'mnist')

    # from torch.utils.data import TensorDataset
    # import torch
    
    # values = torch.arange(10).unsqueeze(dim=-1) ## basically a tensor of values [1], [2], [3]... [10]
    # weights = [1, 2, 3, 4, 5, 1, 1, 1, 1, 1]
    # ds = TensorDataset(values)

    # # ds = MNIST(root=data_path, train=False, transform=tr.ToTensor())
    
    # from mypt.data.dataloaders.standard_dataloaders import initialize_train_dataloader
    # dl = initialize_train_dataloader(dataset_object=ds, seed=0, 
    #                                  batch_size=len(values), 
    #                                  num_workers=2, 
    #                                  weights=weights,
    #                                  drop_last=False)

    # for v in dl:
    #     print(v)
