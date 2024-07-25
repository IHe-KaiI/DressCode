import numpy as np
import torch


# ------------------ Transforms ----------------
def _dict_to_tensors(dict_obj):  # helper
    """convert a dictionary with numeric values into a new dictionary with torch tensors"""
    new_dict = dict.fromkeys(dict_obj.keys())
    for key, value in dict_obj.items():
        if value is None:
            new_dict[key] = torch.Tensor()
        elif isinstance(value, dict):
            new_dict[key] = _dict_to_tensors(value)
        elif isinstance(value, str):  # no changes for strings
            new_dict[key] = value
        elif isinstance(value, np.ndarray):
            new_dict[key] = torch.from_numpy(value)

            # TODO more stable way of converting the types (or detecting ints)
            if value.dtype not in [np.int, np.int64, np.bool]:
                new_dict[key] = new_dict[key].float()  # cast all doubles and ofther stuff to floats
        else:
            new_dict[key] = torch.tensor(value)  # just try directly, if nothing else works
    return new_dict


# Custom transforms -- to tensor
class SampleToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):        
        return _dict_to_tensors(sample)


class FeatureStandartization():
    """Normalize features of provided sample with given stats"""
    def __init__(self, shift, scale):
        self.shift = torch.Tensor(shift)
        self.scale = torch.Tensor(scale)
    
    def __call__(self, sample):
        updated_sample = {}
        updated_sample = sample # TODO: No normalize for captions
        # for key, value in sample.items():
        #     if key == 'features':
        #         updated_sample[key] = (sample[key] - self.shift) / self.scale
        #     else: 
        #         updated_sample[key] = sample[key]

        return updated_sample


class GTtandartization():
    """Normalize features of provided sample with given stats
        * Supports multimodal gt represented as dictionary
        * For dictionary gts, only those values are updated for which the stats are provided
    """
    def __init__(self, shift, scale):
        """If ground truth is a dictionary in itself, the provided values should also be dictionaries"""
        
        self.shift = _dict_to_tensors(shift) if isinstance(shift, dict) else torch.Tensor(shift)
        self.scale = _dict_to_tensors(scale) if isinstance(scale, dict) else torch.Tensor(scale)
    
    def __call__(self, sample):
        gt = sample['ground_truth']
        if isinstance(gt, dict):
            new_gt = dict.fromkeys(gt.keys())
            for key, value in gt.items():
                new_gt[key] = value
                if key in self.shift:
                    new_gt[key] = new_gt[key] - self.shift[key]
                if key in self.scale:
                    new_gt[key] = new_gt[key] / self.scale[key]
                # if shift and scale are not set, the value is kept as it is
        else:
            new_gt = (gt - self.shift) / self.scale

        # gather sample
        updated_sample = {}
        for key, value in sample.items():
            updated_sample[key] = new_gt if key == 'ground_truth' else sample[key]

        return updated_sample
