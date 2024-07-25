import copy
import numpy as np
from pathlib import Path
import random

import torch
# import meshplot  # when uncommented, could lead to problems with wandb run syncing

# My modules
from data.pattern_converter import NNSewingPattern, InvalidPatternDefError
from data.datasets import Garment3DPatternFullDataset


# -------------- Sampler ----------
class BalancedBatchSampler():
    """ Sampler creates batches that have the same class distribution as in given subset"""
    # https://stackoverflow.com/questions/66065272/customizing-the-batch-with-specific-elements
    def __init__(self, ids_by_type, batch_size=10, drop_last=True):
        """
            * ids_by_type provided as dictionary of torch.Subset() objects
            * drop_last is True by default to better guarantee that all batches are well-balanced
        """
        if len(ids_by_type) > batch_size:
            raise NotImplementedError('{}::Error::Creating batches that are smaller then total number of data classes is not implemented!'.format(
                self.__class__.__name__
            ))

        print('{}::Using custom balanced batch data sampler'.format(self.__class__.__name__))

        # represent as lists of ids for simplicity
        self.data_ids_by_type = dict.fromkeys(ids_by_type)
        for data_class in self.data_ids_by_type:
            self.data_ids_by_type[data_class] = ids_by_type[data_class].tolist()

        self.class_names = list(self.data_ids_by_type.keys())
        self.batch_size = batch_size
        self.data_size = sum(len(self.data_ids_by_type[i]) for i in ids_by_type)
        self.num_full_batches = self.data_size // batch_size  # int division
        
        # extra batch left?
        last_batch_len = self.data_size - self.batch_size * self.num_full_batches
        self.drop_last = drop_last or last_batch_len == 0  # by request or because there is no batch with leftovers 
        
        # num of elements per type in each batch
        self.batch_len_per_type = dict.fromkeys(ids_by_type)
        for data_class in self.class_names:
            self.batch_len_per_type[data_class] = int((len(ids_by_type[data_class]) / self.data_size) * batch_size)  # always rounds down
        
        if sum(self.batch_len_per_type.values()) > self.batch_size:
            raise('BalancedBatchSampler::Error:: Failed to evaluate per-type length correctly')


    def __iter__(self):
        ids_by_type = copy.deepcopy(self.data_ids_by_type)

        # shuffle
        for data_class in ids_by_type:
            random.shuffle(ids_by_type[data_class])

        batches = []
        for _ in range(self.num_full_batches):
            batch = []
            for data_class in self.class_names:
                
                for _ in range(self.batch_len_per_type[data_class]):
                    if not len(ids_by_type[data_class]):  # exausted
                        break
                    batch.append(ids_by_type[data_class].pop())

            # Fill the rest of the batch randomly if needed
            diff = self.batch_size - len(batch)
            for _ in range(diff):
                non_empty_class_names = [name for name in self.class_names if len(ids_by_type[name])]
                batch.append(ids_by_type[random.choice(non_empty_class_names)].pop())
            
            random.shuffle(batch)  # to avoid grouping by type in case it matters
            batches.append(batch)

        if not self.drop_last:  
            # put the rest of elements in the last batch
            batch = []
            for ids_list in ids_by_type.values():
                batch += ids_list

            random.shuffle(batch)  # to avoid grouping by type in case it matters
            batches.append(batch)
        
        return iter(batches)

    def __len__(self):
        return self.num_full_batches + (not self.drop_last)


# ------------------------- Utils for non-dataset examples --------------------------

def save_garments_prediction(predictions, save_to, data_config=None, datanames=None, stitches_from_stitch_tags=False):
    """ 
        Saving arbitrary sewing pattern predictions that
        
        * They do NOT have to be coming from garmet dataset samples.
    """

    save_to = Path(save_to)
    batch_size = predictions['outlines'].shape[0]

    if datanames is None:
        datanames = ['pred_{}'.format(i) for i in range(batch_size)]
    
    for idx, name in enumerate(datanames):
        # "unbatch" dictionary
        prediction = {}
        for key in predictions:
            prediction[key] = predictions[key][idx]

        if data_config is not None and 'standardize' in data_config:
            # undo standardization  (outside of generinc conversion function due to custom std structure)
            gt_shifts = data_config['standardize']['gt_shift']
            gt_scales = data_config['standardize']['gt_scale']
            for key in gt_shifts:
                if key == 'stitch_tags' and not data_config['explicit_stitch_tags']:  
                    # ignore stitch tags update if explicit tags were not used
                    continue
                prediction[key] = prediction[key].cpu().numpy() * gt_scales[key] + gt_shifts[key]

        # stitch tags to stitch list
        if stitches_from_stitch_tags:
            stitches = Garment3DPatternFullDataset.tags_to_stitches(
                torch.from_numpy(prediction['stitch_tags']) if isinstance(prediction['stitch_tags'], np.ndarray) else prediction['stitch_tags'],
                prediction['free_edges_mask']
            )
        else:
            stitches = None

        pattern = NNSewingPattern(view_ids=False)
        pattern.name = name
        try:
            pattern.pattern_from_tensors(
                prediction['outlines'], prediction['rotations'], prediction['translations'], 
                stitches=stitches,
                padded=True)   
            # save
            print('[Saving!]')
            pattern.serialize(save_to, to_subfolder=True)
        except (RuntimeError, InvalidPatternDefError, TypeError) as e:
            print(e)
            print('Saving predictions::Skipping pattern {}'.format(name))
            pass
