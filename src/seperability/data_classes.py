from typing import List, Tuple, Union
from copy import deepcopy

import torch
import numpy as np
import pandas as pd
import wandb
from welford_torch import Welford

######################################################################################
# Data Store Classes
######################################################################################

class RunDataItem:
    """ Data class for storing data from a single run.
    """

    # Keys for data stored in RunDataItem
    keys_loss_data = ['loss', 'log_loss']
    keys_accuracy = ['topk', 'topk_skip', 'base', 'skip']
    keys_areas = keys_accuracy
    keys_deletions = ['ff_del', 'ff_threshold', 'attn_del', 'attn_threshold']
    keys_deletions_per_layer = ['ff', 'attn']
    keys_raw = ['ff_raw', 'attn_raw']
    keys = {
        'loss_data': keys_loss_data,
        'accuracy': keys_accuracy,
        'areas': keys_areas,
        'deletions': keys_deletions,
        'deletions_per_layer': keys_deletions_per_layer,
        'raw': keys_raw,
    }

    def __init__(self, datasets: List[str] = None):
        if datasets is None:
            datasets = ['pile', 'code']
        self.datasets = datasets

        # Cross Entropy Loss
        self.loss_data = {
            dataset: {key: 0 for key in self.keys_loss_data} for dataset in datasets
        }

        # Prediction Accuracy percentage
        self.accuracy = {
            dataset: {key: 0 for key in self.keys_accuracy} for dataset in datasets
        }
        self.areas = {
            dataset: {key: 0 for key in self.keys_areas} for dataset in datasets
        }

        # Information about neurons that were deleted
        self.deletions = {key: 0 for key in self.keys_deletions}
        self.deletions_per_layer = {key: [] for key in self.keys_deletions_per_layer}

        # Raw Activations
        self.raw = {key: [] for key in self.keys_raw}

    def update(self, data):
        """ Update data in DataItem.
        Possible Keys: loss_data, accuracy, deletions, raw, deletions_per_layer
        """
        for key, value in data.items():
            getattr(self, key).update(value)
        return self

    def summary(self):
        return {
            'loss_data': self.loss_data,
            'accuracy': self.accuracy,
            'deletions': self.deletions,
            'deletions_per_layer': self.deletions_per_layer,
        }

    def flat_summary(self):
        dataset_loss = {}
        dataset_accuracy = {}
        for dataset in self.datasets:
            for key in self.keys['loss_data']:
                dataset_loss[dataset+'_'+key] = self.loss_data[dataset][key]
            for key in self.keys['accuracy']:
                dataset_accuracy[dataset+'_'+key] = self.accuracy[dataset][key]
        areas = {f'area_{key}': value for key, value in self.areas.items()}

        return { **dataset_loss, **dataset_accuracy, **self.deletions, **areas }

######################################################################################
# Run Data History Class
######################################################################################

class RunDataHistory:
    def __init__(self, use_wandb: bool = True):
        self.history : List[RunDataItem] = []
        self.df = pd.DataFrame()
        self.use_wandb = use_wandb

    def add(self, item: Union[RunDataItem, dict]):
        # Add RunDataItem to history
        if not isinstance(item, RunDataItem) and isinstance(item, dict):
            item = RunDataItem().update(item)
        self.history.append(item)

        # Calculate EXTRACT prediction areas
        self.calculate_areas()
        item = self.history[-1]

        # Log to wandb
        if self.use_wandb:
            wandb.log( self.history[-1].summary() )

        # save to pandas DataFrame
        self.df_append(item)

    def df_append(self, item: RunDataItem):
        new_data = pd.DataFrame({ k:[v] for k,v in item.flat_summary().items() })
        self.df = pd.concat([ self.df, new_data ], ignore_index=True )
        return self.df

    def calculate_areas(self):
        keys = RunDataItem.keys_accuracy
        areas = {}
        for k in keys:
            pile_data = [ run.accuracy['pile'][k] for run in self.history ]
            code_data = [ run.accuracy['code'][k] for run in self.history ]

            total_area = 0.5 * pile_data[0] * code_data[0]
            area = 0

            prev_pile, prev_code = 0, 0
            for (pile, code) in reversed(list(zip(pile_data, code_data))):
                area += 0.5 * (pile - prev_pile) * (code + prev_code)
                prev_pile, prev_code = pile, code

            areas[k] = area / total_area
        self.history[-1].areas = areas

######################################################################################
# Activation Collector Data Class
######################################################################################

class ActivationCollector:
    """ Class for collecting data from model.

    Collects data on activations:
    - count of positive activations
    - mean and variance of activations
    - positive and negative mass / variance
    """
    def __init__(self,
            shape: Tuple[int],
            device: str,
            collect_raw: bool = False,
            dtype: torch.dtype = torch.float64,
        ):
        self.shape       = shape
        self.collect_raw = collect_raw
        self.device      = device
        self.dtype       = dtype
        self.n_points    = 0

        # Welford for calculating mean and variance
        self.all : Welford = Welford(dtype=self.dtype).detach()
        self.pos = Welford(dtype=self.dtype).detach()
        self.neg = Welford(dtype=self.dtype).detach()

        # Count number of times each activation is positive
        self.pos_counter = \
            torch.zeros(shape, device=device, dtype=torch.int32).detach()

        # Optional: collect raw activations
        self.raw = None
        if self.collect_raw:
            self.raw = []

    def add(self, data_point):
        # Add mean and variance of data_point to all_activation
        self.n_points += 1
        self.all.add(data_point)

        # Get information about positive and negative activations
        pos_points = (data_point>0)
        self.pos.add(data_point * pos_points )
        self.neg.add(data_point * pos_points.logical_not() )

        # Add number of positive activations to pos_counter
        self.pos_counter += pos_points

        # Add raw activations to raw if collect_raw
        if self.collect_raw:
            self.raw.append(data_point.detach().cpu())

    def get_raw(self):
        if not self.collect_raw:
            raise ValueError('Raw activations not collected'
                           + ' ActivationCollector.collect_raw=False' )
        if self.n_points == 0:
            raise ValueError('No data points added to ActivationCollector')
        return torch.stack(self.raw)

    def summary(self, dtype: torch.dtype = torch.float32):
        if self.n_points == 0:
            raise ValueError('No data points added to ActivationCollector')

        return {
            'mean': self.all.mean.to(dtype=dtype),
            'std': self.all.var_s.to(dtype=dtype),
            'pos_mass': self.pos.mean.to(dtype=dtype),
            'pos_var': self.pos.var_s.to(dtype=dtype),
            'neg_mass': self.neg.mean.to(dtype=dtype),
            'neg_var': self.neg.var_s.to(dtype=dtype),
            'pos_count': self.pos_counter / self.n_points,
        }
