from argparse import Namespace
import json
import numpy as np
import random
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset

# My modules
from data.utils import BalancedBatchSampler


# ---------------------- Main Wrapper ------------------
class DatasetWrapper(object):
    """Resposible for keeping dataset, its splits, loaders & processing routines.
        Allows to reproduce earlier splits
    """
    def __init__(self, in_dataset, known_split=None, batch_size=None, shuffle_train=True):
        """Initialize wrapping around provided dataset. If splits/batch_size is known """

        self.dataset = in_dataset
        self.data_section_list = ['full', 'train', 'validation', 'test']

        self.training = in_dataset
        self.validation = None
        self.test = None
        self.full_per_datafolder = None

        self.batch_size = None

        self.loaders = Namespace(
            full=None,
            full_per_data_folder=None,
            train=None,
            test=None,
            test_per_data_folder=None,
            validation=None,
            valid_per_data_folder=None
        )

        self.split_info = {
            'random_seed': None, 
            'valid_per_type': None, 
            'test_per_type': None
        }

        if known_split is not None:
            self.load_split(known_split)
        if batch_size is not None:
            self.batch_size = batch_size
            self.new_loaders(batch_size, shuffle_train)
    
    def get_loader(self, data_section='full'):
        """Return loader that corresponds to given data section. None if requested loader does not exist"""
        try:
            return getattr(self.loaders, data_section)
        except AttributeError:
            raise ValueError('DataWrapper::requested loader on unknown data section {}'.format(data_section))
        

    def new_loaders(self, batch_size=None, shuffle_train=True):
        """Create loaders for current data split. Note that result depends on the random number generator!
        
            if the data split was not specified, only the 'full' loaders are created
        """
        if batch_size is not None:
            self.batch_size = batch_size
        if self.batch_size is None:
            raise RuntimeError('DataWrapper:Error:cannot create loaders: batch_size is not set')

        self.loaders.full = DataLoader(self.dataset, self.batch_size)
        if self.full_per_datafolder is None:
            self.full_per_datafolder = self.dataset.subsets_per_datafolder()
        self.loaders.full_per_data_folder = self._loaders_dict(self.full_per_datafolder, self.batch_size)

        if self.validation is not None and self.test is not None:
            # we have a loaded split!
            try:
                self.dataset.config['balanced_batch_sampling'] = True
                # indices IN the training set breakdown per type
                _, train_indices_per_type = self.dataset.indices_by_data_folder(self.training.indices)
                batch_sampler = BalancedBatchSampler(train_indices_per_type, batch_size=self.batch_size)
                self.loaders.train = DataLoader(self.training, batch_sampler=batch_sampler)
            except (AttributeError, NotImplementedError) as e:  # cannot create balanced batches
                print('{}::Warning::Failed to create balanced batches for training. Using default sampling'.format(self.__class__.__name__))
                self.dataset.config['balanced_batch_sampling'] = False
                self.loaders.train = DataLoader(self.training, self.batch_size, shuffle=shuffle_train)
            # no need for breakdown per datafolder for training -- for now

            self.loaders.validation = DataLoader(self.validation, self.batch_size)
            self.loaders.valid_per_data_folder = self._loaders_dict(self.validation_per_datafolder, self.batch_size) 

            self.loaders.test = DataLoader(self.test, self.batch_size)
            self.loaders.test_per_data_folder = self._loaders_dict(self.test_per_datafolder, self.batch_size)

        return self.loaders.train, self.loaders.validation, self.loaders.test

    def _loaders_dict(self, subsets_dict, batch_size, shuffle=False):
        """Create loaders for all subsets in dict"""
        loaders_dict = {}
        for name, subset in subsets_dict.items():
            loaders_dict[name] = DataLoader(subset, batch_size, shuffle=shuffle)
        return loaders_dict

    # -------- Reproducibility ---------------
    def new_split(self, valid, test=None, random_seed=None):
        """Creates train/validation or train/validation/test splits
            depending on provided parameters
            """
        self.split_info['random_seed'] = random_seed if random_seed else int(time.time())
        self.split_info.update(valid_per_type=valid, test_per_type=test, type='count')
        
        return self.load_split()

    def load_split(self, split_info=None, batch_size=None):
        """Get the split by provided parameters. Can be used to reproduce splits on the same dataset.
            NOTE this function re-initializes torch random number generator!
        """
        if split_info:
            self.split_info = split_info

        if 'random_seed' not in self.split_info or self.split_info['random_seed'] is None:
            self.split_info['random_seed'] = int(time.time())
        # init for all libs =)
        torch.manual_seed(self.split_info['random_seed'])
        random.seed(self.split_info['random_seed'])
        np.random.seed(self.split_info['random_seed'])

        # if file is provided
        if 'filename' in self.split_info and self.split_info['filename'] is not None:
            print('DataWrapper::Loading data split from {}'.format(self.split_info['filename']))
            with open(self.split_info['filename'], 'r') as f_json:
                split_dict = json.load(f_json)

            self.training, self.validation, self.test, self.training_per_datafolder, self.validation_per_datafolder, self.test_per_datafolder = self.dataset.split_from_dict(
                split_dict, 
                with_breakdown=True)
        else:
            keys_required = ['test_per_type', 'valid_per_type', 'type']
            if any([key not in self.split_info for key in keys_required]):
                raise ValueError('Specified split information is not full: {}. It needs to contain: {}'.format(split_info, keys_required))
            print('DataWrapper::Loading data split from split config: {}: valid per type {} / test per type {}'.format(
                self.split_info['type'], self.split_info['valid_per_type'], self.split_info['test_per_type']))
            self.training, self.validation, self.test, self.training_per_datafolder, self.validation_per_datafolder, self.test_per_datafolder = self.dataset.random_split_by_dataset(
                self.split_info['valid_per_type'], 
                self.split_info['test_per_type'],
                self.split_info['type'],
                with_breakdown=True)

        if batch_size is not None:
            self.batch_size = batch_size
        if self.batch_size is not None:
            self.new_loaders()  # s.t. loaders could be used right away

        print('DatasetWrapper::Dataset split: {} / {} / {}'.format(
            len(self.training) if self.training else None, 
            len(self.validation) if self.validation else None, 
            len(self.test) if self.test else None))
        self.split_info['size_train'] = len(self.training) if self.training else 0
        self.split_info['size_valid'] = len(self.validation) if self.validation else 0
        self.split_info['size_test'] = len(self.test) if self.test else 0
        
        self.print_subset_stats(self.training_per_datafolder, len(self.training), 'Training')
        self.print_subset_stats(self.validation_per_datafolder, len(self.validation), 'Validation')
        self.print_subset_stats(self.test_per_datafolder, len(self.test), 'Test')

        return self.training, self.validation, self.test

    def print_subset_stats(self, subset_breakdown_dict, total_len, subset_name='', log_to_config=True):
        """Print stats on the elements of each datafolder contained in given subset"""
        # gouped by data_folders
        if not total_len:
            print('{}::Warning::Subset {} is empty, no stats printed'.format(self.__class__.__name__, subset_name))
            return
        self.split_info[subset_name] = {}
        message = ''
        for data_folder, subset in subset_breakdown_dict.items():
            if log_to_config:
                self.split_info[subset_name][data_folder] = len(subset)
            message += '{} : {:.1f}%;\n'.format(data_folder, 100 * len(subset) / total_len)
        
        print('DatasetWrapper::{} subset breakdown::\n{}'.format(subset_name, message))

    def save_to_wandb(self, experiment):
        """Save current data info to the wandb experiment"""
        # Split
        experiment.add_config('data_split', self.split_info)
        # save serialized split s.t. it's loaded to wandb
        split_datanames = {}
        split_datanames['training'] = [self.dataset.datapoints_names[idx] for idx in self.training.indices]
        split_datanames['validation'] = [self.dataset.datapoints_names[idx] for idx in self.validation.indices]
        split_datanames['test'] = [self.dataset.datapoints_names[idx] for idx in self.test.indices]
        with open(experiment.local_wandb_path() / 'data_split.json', 'w') as f_json:
            json.dump(split_datanames, f_json, indent=2, sort_keys=True)

        # data info
        self.dataset.save_to_wandb(experiment)

    # ---------- Standardinzation ----------------
    def standardize_data(self):
        """Apply data normalization based on stats from training set"""
        self.dataset.standardize(self.training)

    # --------- Managing predictions on this data ---------
    def predict(self, model, save_to, sections=['test'], single_batch=False, orig_folder_names=False):
        """Save model predictions on the given dataset section"""
        # Main path
        prediction_path = save_to / ('nn_pred_' + datetime.now().strftime('%y%m%d-%H-%M-%S'))
        prediction_path.mkdir(parents=True, exist_ok=True)

        device = model.device_ids[0] if hasattr(model, 'device_ids') else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        # turn on att weights saving during prediction!
        model.module.save_att_weights = True  # model that don't have this poperty will just ignore it

        for section in sections:
            # Section path
            section_dir = prediction_path / section
            section_dir.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                loader = self.get_loader(section)
                if loader:
                    for batch in loader:
                        features_device = batch['features'].to(device)
                        preds = model(features_device)
                        self.dataset.save_prediction_batch(
                            preds, batch['name'], batch['data_folder'], section_dir, features=batch['features'].numpy(), 
                            model=model, orig_folder_names=orig_folder_names)
                        
                        if single_batch:  # stop after first iteration
                            break
            
        # Turn of to avoid wasting time\memory diring other operations
        model.module.save_att_weights = False

        return prediction_path

