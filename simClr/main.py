import os

from mypt.code_utilities import pytorch_utilities as pu

from mypt.models.simClr.simClrModel import AlexnetSimClr
from model_train.training import run_pipeline


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    train_data_folder = os.path.join(SCRIPT_DIR, 'data', 'fashionMnist', 'train')

    model = AlexnetSimClr(input_shape=(3, 96, 96), output_dim=128, num_fc_layers=3, freeze=False)

    run_pipeline(model=model, 
          train_data_folder=train_data_folder, 
          val_data_folder=None,
          learning_rates=[0.01, 0.1],
          output_shape=(96, 96),
          ckpnt_dir=os.path.join(SCRIPT_DIR, 'logs', 'train_logs'),
          num_epochs=3, 
          batch_size=128, 
          temperature=0.5, 
          seed=0, 
          use_wandb=False)

