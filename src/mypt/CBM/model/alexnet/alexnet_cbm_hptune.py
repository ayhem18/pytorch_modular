import os, json, wandb, shutil

from typing import Union, Dict, List, Optional
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from functools import partial


from .alexnet_cbm_wrapper import initialize_alexnet_cbmw_from_config, AlexnetCbmWrapper
from ..abstract_cbm_wrapper import train_cbm_wrapper, cbm_predict
from ....code_utilities import directories_and_files as dirf
from ....code_utilities import pytorch_utilities as pu


WANDB_PROJECT_NAME = 'CBM-UDA'
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

_BATCH_SIZE = 512


def _select_configurations(sweep_dir: Union[str, Path], 
                          top_k: int = 2) -> Dict[str, Union[float, List[float]]]:    
    # create a list to save val losses
    sweeps_val_loss = []
    sweep_dir = dirf.process_path(sweep_dir)

    for log in os.listdir(sweep_dir):
        sweep_log_dir = os.path.join(sweep_dir, log)    
        files = os.listdir(sweep_log_dir)
        if len(files) == 0:
            continue    
        min_val_loss = float('inf')        
        for ckpnt_org in os.listdir(sweep_log_dir):
            if not ckpnt_org.endswith('.ckpt'):
                continue 
            ckpnt, _ = os.path.splitext(ckpnt_org)
            _, _, val_loss = ckpnt.split('-')
            ckpnt, _ = os.path.splitext(ckpnt_org)
            _, val_loss = val_loss.split('=')
            val_loss = float(val_loss)
            min_val_loss = min(val_loss, min_val_loss)
        
        sweeps_val_loss.append((log, min_val_loss))

    # pick the sweeps with the lowest val loss
    sweeps_val_loss = sorted(sweeps_val_loss, key=lambda x: x[1])[:top_k]
    
    configs_val_loss = [x[0] for x in sweeps_val_loss]

    # the next step is to remove all unnecessary sweeps
    dirf.clear_directory(sweep_dir, lambda p: (os.path.isdir(p) and os.path.basename(p) not in configs_val_loss))

    return [os.path.join(sweep_dir, l) for l in configs_val_loss]


def _sweep_function(log_dir: Union[Path, str], 
                    sweep_config: Dict, 
                    seed: int,
                    dir_train: Union[Path,str], 
                    dir_val: Union[str, Path],
                    concepts,
                    epoch_per_sweep:int, 
                    topk_rep3:int 
                    ):
    # first define the sub log dir
    log_subdir = os.path.join(log_dir, f'log_{len(os.listdir(log_dir)) + 1}')

    # initialize wanbd 
    wandb.init(project=WANDB_PROJECT_NAME,
                config=sweep_config,
                name=os.path.basename(log_dir))

    wandb_logger = WandbLogger(project=WANDB_PROJECT_NAME,
                    log_model=False, 
                    save_dir=log_subdir, 
                    name=os.path.basename(log_dir))


    # let's create the configuration really quick
    config = {}
    for c, _ in sweep_config['parameters'].items():
        config[c] = eval(f'wandb.config.{c}')

    # create a model using the wandb.configuration
    cbm, _ = initialize_alexnet_cbmw_from_config(config=config, seed=seed)
    
    best_ckpnt = train_cbm_wrapper(cbm_wrapper=cbm,
                        train_dir=dir_train,
                        concepts=concepts,
                        val_dir=dir_val,
                        log_dir=log_subdir, 
                        run_name=os.path.basename(log_dir), 
                        wandb_logger=wandb_logger, 
                        remove_existing=False, 
                        num_epochs=epoch_per_sweep, 
                        representation=config['representation'], 
                        similarity=config['similarity'], 
                        seed=seed, 
                        repeat_config=True,
                        top_k=topk_rep3, 
                        num_workers=2, 
                        batch_size=_BATCH_SIZE)
    
    # # initialize the best model from this config
    # best_cbm = AlexnetCbmWrapper.load_from_checkpoint(best_ckpnt)

    # cbm_accuracy = cbm_predict(best_cbm, 
    #                             seed=seed,
    #                             concepts=concepts, 
    #                             data_dir=dir_target, 
    #                             batch_size=256) # increasing the batch size should not be an issue since the gradient calculation is turned off

    # config['target_accuracy'] = cbm_accuracy
    # save the config as json
    with open(os.path.join(log_subdir, 'config.json'), 'w') as fp: 
        json.dump(config, fp, indent=2)

    wandb.finish()
    

def hypertune_alexnet_cbm(cbm_configuration_path: Union[str, Path], 
                         num_concept_layers_options: Dict,
                         loss_coefficient_options: Dict,
                         dropout_options: Dict,
                         data_dir: Union[str, Path],
                         concepts: Union[Dict[str, List[str]], List[str]],
                         log_dir: Union[str, Path],
                         top_k: int = 1,
                         sweep_count: int = 50, 
                         epoch_per_sweep: int = 10,
                         seed: int = 69, 
                         topk_rep3: int = 1,
                         tune_method: str = 'bayes',
                         ):

    with open(cbm_configuration_path) as f:
        cbm_config = json.load(f)

    # before proceeding, make sure to replace any num_epochs field with the 'epoch_per_sweep' value
    for k, v in cbm_config.items():
        if k == 'num_epochs': 
            cbm_config[k] = epoch_per_sweep
        
        if isinstance(v, Dict):
            for _k, _v in v.items():
                if _k == 'num_epochs':
                    v[_k] = epoch_per_sweep

    # make sure to convert the wandb format
    for k, v in cbm_config.items():
        cbm_config[k] = {"value": v}

    # add the options for the major hyperparameters
    cbm_config["num_concept_layers"] = num_concept_layers_options
    cbm_config['loss_coefficient'] = loss_coefficient_options
    cbm_config['concept_projection_dropout'] = dropout_options

    sweep_config = {
        "name": "cbm_sweep", 
        "metric": {"name": "val_loss", "goal": "minimize"}, 
        "method": tune_method, 
        "parameters": cbm_config,
    }

    data_dir = dirf.process_path(os.path.join(data_dir), 
                                 file_ok=False, 
                                 condition=lambda p: {'train', 'val'}.issubset(set(os.listdir(p))))
    # process the paths
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')    

    log_dir = dirf.process_path(os.path.join(log_dir, f'sweep_{os.path.splitext(os.path.basename(cbm_configuration_path))[0]}_{os.path.basename(data_dir)}'))
    
    sweep_id = wandb.sweep(sweep=sweep_config, 
                    project=WANDB_PROJECT_NAME)


    # let's define the object that the sweep agent will run
    sweep_function_object = partial(_sweep_function, 
                                    log_dir=log_dir, 
                                    sweep_config=sweep_config,
                                    seed=seed,
                                    dir_train=train_dir, 
                                    dir_val=val_dir, 
                                    concepts=concepts, 
                                    epoch_per_sweep=epoch_per_sweep,
                                    topk_rep3=topk_rep3, 
                                    )

    # run the sweep
    wandb.agent(sweep_id, 
            function=sweep_function_object,
            count=sweep_count)
    
    # make sure to select the results
    logs_val_loss =  _select_configurations(sweep_dir=log_dir, 
                          top_k=top_k)
     
    # move them to the folder of the configuration
    config_folder = Path(cbm_configuration_path).parent

    new_configs_val_loss = []

    for i, l in enumerate(logs_val_loss):
        config_path = os.path.join(l, 'config.json')
        des_path = os.path.join(config_folder, f'{os.path.splitext(os.path.basename(cbm_configuration_path))[0]}_val_loss{i}.json')
        new_configs_val_loss.append(des_path)
        shutil.copy(config_path, des_path)

    return new_configs_val_loss


DEFAULT_NUM_LAYERS = {'values': [2, 3, 4, 5]}
DEFAULT_LOSS_COEFF =  {'min': 0.1, 'max': 1.0}
DEFAULT_DROPOUT = {'min': 0.1, 'max': 1.0}
