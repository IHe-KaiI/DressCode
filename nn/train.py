from distutils import dir_util
from pathlib import Path
import argparse
import numpy as np
import torch.nn as nn
import yaml

# My modules
import customconfig
import data
import nets
from trainer import Trainer
from experiment import ExperimentWrappper

import warnings
warnings.filterwarnings('ignore')  # , category='UserWarning'


def get_values_from_args():
    """command line arguments to control the run for running wandb Sweeps!"""
    # https://stackoverflow.com/questions/40001892/reading-named-command-arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', '-c', help='YAML configuration file', type=str, default='./models/att/att.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    return config 


def get_old_data_config(in_config):
    """Shortcut to control data configuration
        Note that the old experiment is HARDCODED!!!!!"""
    # get data stats from older runs to save runtime
    old_experiment = ExperimentWrappper({'experiment': in_config['old_experiment']}, system_info['wandb_username'])
    # NOTE data stats are ONLY correct for a specific data split, so these two need to go together
    split, _, data_config = old_experiment.data_info()

    # Use only minimal set of settings
    # NOTE: you can remove elements for which the in_config should be a priority
    #       from the list below
    data_config = {
        'standardize': data_config['standardize'],
        'max_pattern_len': data_config['max_pattern_len'],
        'max_panel_len': data_config['max_panel_len'],
        'max_num_stitches': data_config['max_num_stitches'],  # the rest of the info is not needed here
        'max_datapoints_per_type': data_config['max_datapoints_per_type'] if 'max_datapoints_per_type' in data_config else None,  # keep the numbers too
        'panel_classification': data_config['panel_classification'],
        'filter_by_params': data_config['filter_by_params'],
        'mesh_samples': data_config['mesh_samples'],
        'obj_filetag': data_config['obj_filetag'],
        'point_noise_w': data_config['point_noise_w'] if 'point_noise_w' in data_config else 0
    }
    # update with freshly configured values
    in_config.update(data_config)
    
    print(split)

    return split, in_config


if __name__ == "__main__":

    np.set_printoptions(precision=4, suppress=True)  # for readability

    config = get_values_from_args()
    system_info = customconfig.Properties('./system.json')
    
    experiment = ExperimentWrappper(
        config,  # set run id in cofig to resume unfinished run!
        system_info['wandb_username'],
        no_sync=True)   


    # Dataset Class
    data_class = getattr(data, config['dataset']['class'])
    dataset = data_class(Path(system_info['datasets_path']), Path(system_info['caption_path']),config['dataset'], gt_caching=True, feature_caching=True)

    # --- Trainer --- 
    trainer = Trainer(
        config['trainer'], experiment, dataset, config['data_split'], 
        with_norm=True, with_visualization=config['trainer']['with_visualization'], resume=config['experiment']['run_id'])  # only turn on visuals on custom garment data

    # --- Model ---
    trainer.init_randomizer()
    model_class = getattr(nets, config['NN']['model'])
    model = model_class()


    # --- TRAIN --- 
    trainer.fit(model)  # Magic happens here

    # --- Final evaluation ----
    # On the best-performing model
    try:
        model.load_state_dict(experiment.get_best_model()['model_state_dict'])
    except BaseException as e:  # not the best to catch all the exceptions here, but should work for most of cases foe now
        print(e)
        print('Train::Warning::Proceeding to evaluation with the current (final) model state')

