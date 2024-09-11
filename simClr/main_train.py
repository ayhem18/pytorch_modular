import os
from mypt.code_utilities import directories_and_files as dirf
from mypt.models.simClr.simClrModel import ResnetSimClr
from mypt.subroutines.neighbors import model_embs as me

from model_train.training import run_pipeline, OUTPUT_SHAPE 

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

_BATCH_SIZE = 320

_OUTPUT_DIM = 128


def train_main(model):
    train_data_folder = os.path.join(SCRIPT_DIR, 'data', 'food101', 'train')
    val_data_folder = os.path.join(SCRIPT_DIR, 'data', 'food101', 'val')
    
    ckpnt_dir_parent = dirf.process_path(os.path.join(SCRIPT_DIR, 'logs', 'train_logs'), dir_ok=True, file_ok=False)

    return run_pipeline(model=model, 
        train_data_folder=train_data_folder, 
        val_data_folder=val_data_folder,
        dataset='imagenette',
        output_shape=OUTPUT_SHAPE[1:],
        ckpnt_dir=os.path.join(ckpnt_dir_parent, f'iteration_{len(os.listdir(ckpnt_dir_parent)) + 1}'),
        num_epochs=3, 
        batch_size=_BATCH_SIZE, 
        temperature=0.5, 
        seed=0, 
        use_wandb=False,
        batch_stats=False,
        debug_loss=True, # get as much insight on the loss as possible
        num_train_samples_per_cls=10,
        num_val_samples_per_cls=10,
        num_warmup_epochs=0,
        )


if __name__ == '__main__':

    model = ResnetSimClr(input_shape=OUTPUT_SHAPE,  
                         output_dim=_OUTPUT_DIM, 
                         num_fc_layers=4, 
                         freeze=False, 
                         architecture=50)
    train_main(model)
