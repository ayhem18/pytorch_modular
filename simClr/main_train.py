"""
This script contains the implementation of the main training process 
"""

import os

from mypt.code_utilities import directories_and_files as dirf, pytorch_utilities as pu
from model_train_pl.train import train_simClr_wrapper, ResnetSimClrWrapper, evaluate_augmentations, calculate_weights
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


    train_simClr_wrapper(
                        train_data_folder=train_data_folder,
                        val_data_folder=val_data_folder,
                        dataset=dataset_name,
                        log_dir=os.path.join(ckpnt_dir_parent, f'{dataset_name}_iteration_{n + 1}'),
                        num_epochs=2, 
                        train_batch_size=512,
                        val_batch_size=1600, 
                        seed=0, 
                        use_logging=False,
                        num_warmup_epochs=0,
                        val_per_epoch=1,
                        num_train_samples_per_cls=None,
                        num_val_samples_per_cls=None,
                        run_name=run_name,     
                        debug_augmentations=_DEFAULT_DATA_AUGS                   
                        )

if __name__ == '__main__':
    # train_main()   

    ckpnt_dir_parent = dirf.process_path(os.path.join(SCRIPT_DIR, 'logs', 'train_logs'), 
                                         dir_ok=True, 
                                         file_ok=False)

    # delete any empty folders in the 'train_logs' folder
    dirf.clear_directory(ckpnt_dir_parent, condition=lambda x: os.path.isdir(x) and len(os.listdir(x)) == 0)
    n = len(os.listdir(ckpnt_dir_parent))
    log_dir=os.path.join(ckpnt_dir_parent, f'imagenette_iteration_{n}')

    ckpnt = os.path.join(log_dir, 'val-epoch=00-val_epoch_loss=6.731850.ckpt')

    wrapper = ResnetSimClrWrapper.load_from_checkpoint(ckpnt)

    import torch
    aug_scores =torch.load(ckpnt)['augmentation_scores']
    augs_per_difficulty = evaluate_augmentations(
                                        augmentations=_DEFAULT_DATA_AUGS,
                                        get_augmentation_name=pu.get_augmentation_name, 
                                        augmentation_scores=aug_scores,
                                        num_categories=3)


    

    # # the next step woulw
    # sample_weights = calculate_weights(augmentations_per_difficulty=augs_per_difficulty,
    #                                    dataset=)
