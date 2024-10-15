"""
This script contains the implementation of the main training process 
"""

import os
from mypt.code_utilities import directories_and_files as dirf, pytorch_utilities as pu

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
    from model_train_pl.constants import _OUTPUT_DIM, _OUTPUT_SHAPE
    from mypt.models.simClr.simClrModel import ResnetSimClr
    import torch

    model = ResnetSimClr(input_shape=_OUTPUT_SHAPE,  
                         output_dim=_OUTPUT_DIM, 
                         num_fc_layers=4, 
                         freeze=False, 
                         architecture=101)

    # ckpnt = os.path.join(SCRIPT_DIR, 'logs', 'train_logs', 'food101_iteration_1', 'ckpnt_val_loss_4.6184_epoch_49.pt')

    # # load the model
    # print(torch.load(ckpnt)['model_state_dict'])

    # let's get this out of the way already

    from model_train_pl.simClrWrapper import cosine_sims_model_embeddings_with_transformations

    from mypt.data.datasets.genericFolderDs import GenericFolderDS
    from mypt.data.dataloaders.standard_dataloaders import initialize_val_dataloader
    from torchvision import transforms as tr

    food101_ds_path = os.path.join(SCRIPT_DIR, 'data', 'food101', 'train/food-101/images')


    model.forward(torch.randn(size=(10, 3, 200, 200)))

    ds = GenericFolderDS(root=food101_ds_path, transforms=[tr.ToTensor(), 
                                                           tr.Resize(size=(200, 200))])

    dl = initialize_val_dataloader(ds, seed=0, batch_size=100, num_workers=0, warning=False)

    for b in dl:    
        b = b.to(device)
        cosine_sims_model_embeddings_with_transformations(model=model, 
                                                          batch=b, 
                                                          transformations=[tr.ColorJitter(),
                                                                           tr.GaussianBlur(kernel_size=(5, 5))]
                                                        )
        break
