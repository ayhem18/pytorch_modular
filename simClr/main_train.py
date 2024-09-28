"""
This script contains the implementation of the main training process 
"""

import os
from mypt.code_utilities import directories_and_files as dirf

from model_train_pl.train import train_simClr_wrapper
from model_train_pl.constants import TRACK_PROJECT_NAME

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def train_main():
    dataset_name = 'imagenette'
    train_data_folder = os.path.join(SCRIPT_DIR, 'data', dataset_name, 'train')
    val_data_folder = os.path.join(SCRIPT_DIR, 'data', dataset_name, 'val')
    
    ckpnt_dir_parent = dirf.process_path(os.path.join(SCRIPT_DIR, 'logs', 'train_logs'), dir_ok=True, file_ok=False)

    # delete any empty folders in the 'train_logs' folder
    dirf.clear_directory(ckpnt_dir_parent, condition=lambda x: os.path.isdir(x) and len(os.listdir(x)) == 0)
    n = len(os.listdir(ckpnt_dir_parent))

    run_name = f'{TRACK_PROJECT_NAME}_iteration_{n + 1}'

    train_simClr_wrapper(
                        train_data_folder=train_data_folder,
                        val_data_folder=val_data_folder,
                        dataset=dataset_name,
                        log_dir=os.path.join(ckpnt_dir_parent, f'{dataset_name}_iteration_{n + 1}'),
                        num_epochs=5, 
                        batch_size=16, 
                        seed=0, 
                        use_logging=False,
                        num_warmup_epochs=0,
                        val_per_epoch=1,
                        num_train_samples_per_cls=100,
                        num_val_samples_per_cls=100,
                        run_name=run_name,                        
                        )

if __name__ == '__main__':
    # train_main()
    train_data_folder = os.path.join(SCRIPT_DIR, 'data', 'imagenette_tune', 'train')
    val_data_folder = os.path.join(SCRIPT_DIR, 'data', 'imagenette_tune', 'val')

    from model_train_pl.set_ds import _set_data_tune
    train_dl, val_dl = _set_data_tune(train_data_folder, val_data_folder, 
                        output_shape=(200, 200),
                        batch_size=10, 
                        dataset='imagenette')

    print(len(train_dl))
    print(len(val_dl))

    # from mypt.data.datasets.parallel_augmentation import parallel_aug_dir as pad
    # from torchvision import transforms as tr
    # ds = pad.ParallelAugDirDs(train_data_folder, 
    #                      output_shape=(200, 200), 
    #                      augs_per_sample=2, 
    #                      sampled_data_augs=[tr.AutoAugment(), tr.CenterCrop(size=(150, 150)), tr.Grayscale()],
    #                      uniform_augs_after=[],
    #                      uniform_augs_before=[])
    
    # print(len(ds))
    
    # _set_data_tune()

