import os, json, wandb, shutil

from typing import Union, Dict, List, Optional
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from functools import partial

from .resnet_classifier_wrapper import initialize_resnet_classifier_from_config, ResnetClassifierWrapper
from ..abstract_classifier_wrapper import train_classifier_wrapper, classifier_predict
from ...code_utilities import directories_and_files as dirf
from ...code_utilities import pytorch_utilities as pu


WANDB_PROJECT_NAME = 'CBM-UDA'
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def _select_configurations(sweep_dir: Union[str, Path], 
                          top_k: int = 2) -> List[Union[str, Path]]:
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
                    dir_target: Optional[Union[str, Path]],
                    epoch_per_sweep:int, 
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
    classifier, _ = initialize_resnet_classifier_from_config(config=config, seed=seed)
    
    best_ckpnt = train_classifier_wrapper(classifier=classifier,
                        train_dir=dir_train,
                        val_dir=dir_val,
                        log_dir=log_subdir, 
                        run_name=os.path.basename(log_dir), 
                        num_workers=2, 
                        batch_size=64,
                        num_epochs=epoch_per_sweep, 
                        wandb_logger=wandb_logger, 
                        seed=seed, 
                        repeat_config=True,
                        )
    
    # initialize the best model from this config
    best_classifier = ResnetClassifierWrapper.load_from_checkpoint(best_ckpnt)

    acc = classifier_predict(best_classifier, 
                                data_dir=dir_target, 
                                seed=seed,
                                batch_size=256) # increasing the batch size should not be an issue since the gradient calculation is turned off

    config['target_accuracy'] = acc
    # save the config as json
    with open(os.path.join(log_subdir, 'config.json'), 'w') as fp: 
        json.dump(config, fp, indent=2)

    wandb.finish()
    

def hypertune_resnet_classifier(classifier_configuration_path: Union[str, Path], 
                         num_classification_layers_options: Dict,
                         dropout_options: Dict,
                         data_dir: Union[str, Path],
                         target_dir: Union[Path, str], 
                         log_dir: Union[str, Path],
                         top_k: int = 1,
                         sweep_count: int = 50, 
                         epoch_per_sweep: int = 10,
                         seed: int = 69, 
                         tune_method: str = 'bayes',
                         ):

    with open(classifier_configuration_path) as f:
        classifier_config = json.load(f)

    # before proceeding, make sure to replace any num_epochs field with the 'epoch_per_sweep' value
    for k, v in classifier_config.items():
        if k == 'num_epochs': 
            classifier_config[k] = epoch_per_sweep
        
        if isinstance(v, Dict):
            for _k, _v in v.items():
                if _k == 'num_epochs':
                    v[_k] = epoch_per_sweep

    # make sure to convert the wandb format
    for k, v in classifier_config.items():
        classifier_config[k] = {"value": v}

    # add the options for the major hyperparameters
    classifier_config["num_classification_layers"] = num_classification_layers_options
    classifier_config['dropout'] = dropout_options

    sweep_config = {
        "name": "classifier_sweep", 
        "metric": {"name": "val_loss", "goal": "minimize"}, 
        "method": tune_method, 
        "parameters": classifier_config,
    }

    data_dir = dirf.process_path(os.path.join(data_dir), 
                                 file_ok=False, 
                                 condition=lambda p: {'train', 'val'}.issubset(set(os.listdir(p))))
    # process the paths
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')    
    target_dir = dirf.process_path(target_dir)

    log_dir = dirf.process_path(os.path.join(log_dir, f'sweep_{os.path.splitext(os.path.basename(classifier_configuration_path))[0]}_{os.path.basename(data_dir)}'))
    
    sweep_id = wandb.sweep(sweep=sweep_config, 
                    project=WANDB_PROJECT_NAME)


    # let's define the object that the sweep agent will run
    sweep_function_object = partial(_sweep_function, 
                                    log_dir=log_dir, 
                                    sweep_config=sweep_config,
                                    seed=seed,
                                    dir_train=train_dir, 
                                    dir_val=val_dir, 
                                    dir_target=target_dir, 
                                    epoch_per_sweep=epoch_per_sweep,
                                    )

    # run the sweep
    wandb.agent(sweep_id, 
            function=sweep_function_object,
            count=sweep_count)
    
    # make sure to select the results
    logs_val_loss =  _select_configurations(sweep_dir=log_dir, 
                          top_k=top_k)
     
    # move them to the folder of the configuration
    config_folder = Path(classifier_configuration_path).parent

    new_configs_val_loss = []

    for i, l in enumerate(logs_val_loss):
        config_path = os.path.join(l, 'config.json')
        des_path = os.path.join(config_folder, f'{os.path.splitext(os.path.basename(classifier_configuration_path))[0]}_{os.path.basename(data_dir)}_val_loss{i}.json')
        new_configs_val_loss.append(des_path)
        shutil.copy(config_path, des_path)

        # remove the 'target_accuracy' field from the copied configurations
        with open(des_path, 'r') as f:
            p = json.load(f)            
        del(p['target_accuracy'])
        with open(des_path, 'w') as f: 
            json.dump(p, f, indent=2)    
    
    return new_configs_val_loss

DEFAULT_NUM_LAYERS = {'values': [1, 2, 3, 4]}
DEFAULT_DROPOUT = {'min': 0.1, 'max': 0.5}
