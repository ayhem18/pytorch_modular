import wandb, os

from pathlib import Path
from typing import Union

from mypt.utilities.pytorch_utilities import seed_everything, get_default_device, get_module_device

import train
from train import set_model

def sanity_check(sweep_config, 
                 seed: int, 
                 log_dir: Union[str, Path]):
    sweep_id = wandb.sweep(sweep=sweep_config, 
                    project=train.WANDB_PROJECT_NAME)

    def sweep_function():
        seed_everything(seed)
        # first define the sub log dir
        log_subdir = os.path.join(log_dir, f'log_{len(os.listdir(log_dir)) + 1}')

        # initialize wanbd 
        wandb.init(project=train.WANDB_PROJECT_NAME,
                    config=sweep_config,
                    name=os.path.basename(log_dir))

        

        # let's create the configuration really quick
        config = {}
        for c, _ in sweep_config['parameters'].items():
            config[c] = eval(f'wandb.config.{c}')

        # create a model using the wandb.configuration
        cbm, _ = initialize_resnet_cbmw_from_config(config=config, seed=seed)
        
        best_ckpnt = train_cbm_wrapper(cbm_wrapper=cbm,
                            train_dir=train_dir,
                            concepts=concepts,
                            val_dir=val_dir,
                            log_dir=log_subdir, 
                            run_name=os.path.basename(log_dir), 
                            wandb_logger=wandb_logger, 
                            remove_existing=False, 
                            num_epochs=epoch_per_sweep, 
                            representation=config['representation'], 
                            similarity=config['similarity'], 
                            seed=seed, 
                            repeat_config=True,
                            top_k=topk_rep3)
        
        # initialize the best model from this config
        best_cbm = ResnetCbmWrapper.load_from_checkpoint(best_ckpnt)

        cbm_accuracy = cbm_predict(best_cbm, 
                        concepts=concepts, 
                        data_dir=target_dir)

        config['target_accuracy'] = cbm_accuracy
        # save the config as json
        with open(os.path.join(log_subdir, 'config.json'), 'w') as fp: 
            json.dump(config, fp, indent=2)

    # run the sweep
    wandb.agent(sweep_id, 
            function=sweep_function,
            count=sweep_count)


def sweep_function(configuration_path):
    pass    
