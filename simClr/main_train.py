"""
This script contains the implementation of the main training process 
"""

import os

from mypt.code_utilities import directories_and_files as dirf, pytorch_utilities as pu
from model_train_pl.train import ResnetSimClrWrapper, train_simClr
from model_train_pl.constants import TRACK_PROJECT_NAME
from model_train_pl.set_ds import _DEFAULT_DATA_AUGS

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def train_main():
    dataset_name = 'food101'
    train_data_folder = os.path.join(SCRIPT_DIR, 'data', dataset_name, 'train')
    val_data_folder = os.path.join(SCRIPT_DIR, 'data', dataset_name, 'val')
    
    ckpnt_dir_parent = dirf.process_path(os.path.join(SCRIPT_DIR, 'logs', 'train_logs'), 
                                         dir_ok=True, 
                                         file_ok=False)

    # delete any empty folders in the 'train_logs' folder
    dirf.clear_directory(ckpnt_dir_parent, 
                         condition=lambda x: os.path.isdir(x) and 
                         len([y for y in os.listdir(x) 
                              if not (os.path.isdir(os.path.join(x, y)) and len(os.listdir(os.path.join(x, y))) == 0)
                              ]
                              ) 
                         == 0)
    
    n = len(os.listdir(ckpnt_dir_parent))

    run_name = f'{TRACK_PROJECT_NAME}_iteration_{n + 1}'

    
    train_simClr(
                train_data_folder=train_data_folder,
                val_data_folder=val_data_folder,
                dataset=dataset_name,
                log_dir=os.path.join(ckpnt_dir_parent, f'{dataset_name}_iteration_{n + 1}'),
                num_epochs=10, 
                train_batch_size=256,
                val_batch_size=256,  
                seed=0, 
                use_logging=True,
                num_warmup_epochs=0,
                val_per_epoch=3,
                num_train_samples_per_cls=10,
                num_val_samples_per_cls=10,
                run_name=run_name,     
                debug_augmentations=_DEFAULT_DATA_AUGS                   
                )

if __name__ == '__main__':
    train_main()   
    # define the wrapper
    # wrapper = ResnetSimClrWrapper(
    #                               # model parameters
    #                               input_shape=(3,) + (200, 200), 
    #                               output_dim=256,
    #                               num_fc_layers=4,

    #                               #logging parameters
    #                               logger=None,
    #                               log_per_batch=3,
    #                               val_per_epoch=3,
    #                               # loss parameters 
    #                               temperature=0.5,
    #                               debug_loss=True,

    #                               # learning 
    #                               lrs=0.6,
    #                               num_epochs=3,
    #                               num_warmup_epochs=0,

    #                               debug_embds_vs_trans=True, 
    #                               debug_augmentations=[],

    #                               # more model parameters
    #                               dropout=0.2,
    #                               architecture=50,
    #                               freeze=False, 
    #                               save_hps=False, # no need to save the hyperparameters, this way the model will only load the weights
    #                               )    

    # import torch

    # initial_checkpoint = os.path.join(SCRIPT_DIR, 'logs', 'train_logs', 'food101_iteration_2', 'round_1', 'val-epoch=05-val_epoch_loss=4.848253.ckpt')
    # wrapper.model.load_state_dict(torch.load(initial_checkpoint)['state_dict'])
    # # let's see if we can load the state of optimizers and schedulers as well
    # wrapper.optimizers().optimizer.load_state_dict(torch.load(initial_checkpoint)['optimizer'])
    # wrapper.lr_schedulers().scheduler.load_state_dict(torch.load(initial_checkpoint)['lr_scheduler'])

