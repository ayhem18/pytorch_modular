"""
"""

import os

from typing import Union, Dict
from pathlib import Path

from model_train import tuning as tn

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def sanity_check_tuning(sanity_train: Union[str, Path],
                        temperature: float,
                        lr_options: Dict,
                        num_layers_options: Dict,
                        epochs_per_sweeps:int,
                        sweep_count:int
                        ):
    # the idea here is to find a set of hyperparameters enabling the model to overfit to the training data
    # the model with the lowest training loss will be chosen for further training
     
    log_dir = os.path.join(SCRIPT_DIR, 'logs', 'tune_logs')
    tn.tune(train_data_folder=sanity_train, 
            val_data_folder=None,
            log_dir=log_dir,
            temperature=temperature,
            lr_options=lr_options,
            num_layers_options=num_layers_options,
            objective='train_loss',
            tune_method='bayes',
            epochs_per_sweeps=epochs_per_sweeps,
            sweep_count=sweep_count
            )


def set_the_initial_model():
    # let's start with lr_options: 
    # the learning rate of the feature extractor will be between 10 ** -3 and 10 ** -1
    lr_options = {"max": -1.0, "min":-3.0}
    num_fc_layers = {"values": list(range(2, 7))}

    sanity_train = os.path.join(SCRIPT_DIR, 'data', 'stl10', 'train')
    sanity_check_tuning(sanity_train=sanity_train, 
                        lr_options=lr_options, 
                        num_layers_options=num_fc_layers,
                        epochs_per_sweeps=50, 
                        sweep_count=20, 
                        temperature=0.5)
    

if __name__ == '__main__':
    # let's start with something simple
    set_the_initial_model()

