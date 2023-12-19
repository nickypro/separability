from typing import List, Dict, Tuple, Union, Optional, Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
import pandas as pd
import wandb
from welford_torch import Welford

######################################################################################
# Functional Conversion Data Classes
######################################################################################

# Class for storing dtype as string
class DtypeMap():
    def __init__(self, str_dtype=None, torch_dtype=None):
        self.str_dtype = str_dtype
        self.torch_dtype = torch_dtype

    @property
    def _dtype(self):
        # Manual Override
        if self.torch_dtype is not None:
            return self.torch_dtype

        # Auto type from string
        dtype_map = {
            "int4": torch.float16,
            "int8": torch.float16, #torch.qint8 ?
            "fp16": torch.float16,
            "fp32": torch.float32,
            "fp64": torch.float64,
        }
        return dtype_map[self.str_dtype]

    @property
    def _dtype_args(self):
        dtype_key = "torch_dtype"

        # Manual Override
        if self.torch_dtype is not None:
            return {dtype_key: self.torch_dtype}

        # Auto type from string
        args = {
            "int4": {dtype_key: self._dtype, "load_in_4bit": True},
            "int8": {dtype_key: self._dtype, "load_in_8bit": True},
            "fp16": {dtype_key: self._dtype},
            "fp32": {dtype_key: self._dtype},
            "fp64": {dtype_key: self._dtype},
        }
        return args[self.str_dtype]

    @property
    def is_low_precision(self):
        return self.str_dtype in ["int4", "int8"]

######################################################################################
# Evaluation Data Classes
######################################################################################

def _set_empty_attrs_to_dict(__class):
    for attr in __class.__dataclass_fields__:
        __attr = getattr(__class, attr)
        if __attr is None:
            setattr(__class, attr, {})

@dataclass
class EvalConfig:
    dataset_name: str
    dataset_repo: str = None
    dataset_subset: str = None
    dataset_type: str = "prediction" # ["prediction", "generation", "mmlu"]
    dataset_text_key: str = "text"
    dataset_filter: Optional[Callable] = None
    dataset_has_test_split: bool = True
    dataset_split: str = None # "test", "train"
    streaming: bool = True # Do not download the whole dataset by default
    sample_size: int = 1e5
    skip_token_strings: Optional[List[str]] = None
    skip_token_ids: Tensor = None
    topk: int = 10
    start_index: int = 0
    num_texts_to_skip:  int = 0
    num_tokens_to_skip: int = 1e5
    num_top_tokens: int = 50 # count tokens, and show this number of top tokens
    loading_bar_desc: str = "Acc"
    verbose: bool = False
    # Custom Eval Parameters
    is_train_mode: bool = False
    dataset_image_key: str = "image"
    dataset_image_label_key: str = "label"
    n_shot: int = 0
    masked_model: bool = False
    masked_token_str: str = "<mask>"
    masked_token_id: int = None
    masked_frac_chosen: float            = 0.15
    masked_frac_chosen_masked: float     = 0.8
    masked_frac_chosen_randomized: float = 0.1
    masked_frac_chosen_unchanged: float  = 0.1
    mmlu_subsets: Optional[Union[str, List[str]]] = None
    sliding_window_buffer_size = 1024
    sliding_window_step_size = 512
    generated_text_prompt: str = None
    generated_text_include_prompt: bool = False
    generated_text_num_samples: int = 1
    generated_text_length: int = 50
    generated_text_temperature: float = None
    # Membership Inference Attack
    mia_retain: str = None
    mia_retain_split: str = None
    mia_forget: str = None
    mia_forget_split: str = None
    mia_test: str = None
    mia_test_split: str = None
    misc: Optional[Dict[str, any]] = None

    def to_dict(self):
        _dict = {}
        for attr in self.__dataclass_fields__:
            _dict[attr] = getattr(self, attr)
        return _dict


@dataclass
class EvalOutput:
    loss_data: dict = None
    percent: dict = None
    misc: dict = None

    def __post_init__(self):
        _set_empty_attrs_to_dict(self)

@dataclass
class EvalAllOutput:
    loss_data: dict = None
    accuracy: dict = None
    misc: dict = None

    def __post_init__(self):
        _set_empty_attrs_to_dict(self)

    def add(self, dataset: str, data: EvalOutput):
        self.loss_data[dataset] = data.loss_data
        self.accuracy[dataset]  = data.percent
        self.misc[dataset]      = data.misc

    def to_dict(self):
        _dict = {}
        for attr in self.__dataclass_fields__:
            _dict[attr] = getattr(self, attr)
        return _dict


# Raw Data Collector for %Accuracy
@dataclass
class RawAccuracyData:
    num_predictions: int = 0
    num_accurate: int = 0
    num_topk_accurate: int = 0
    num_skip_predictions: int = 0
    num_skip_accurate: int = 0
    num_topk_skip_accurate: int = 0
    token_counts: Optional[np.ndarray] = None

    # Make addition just the sum of attributes
    def __add__(self, other):
        if self.token_counts is None and other.token_counts is None:
            new_token_counts = None
        elif self.token_counts is None:
            new_token_counts = other.token_counts
        elif other.token_counts is None:
            new_token_counts = self.token_counts
        else:
            new_token_counts = self.token_counts + other.token_counts

        return RawAccuracyData(
            num_predictions=self.num_predictions + other.num_predictions,
            num_accurate=self.num_accurate + other.num_accurate,
            num_topk_accurate=self.num_topk_accurate + other.num_topk_accurate,
            num_skip_predictions=self.num_skip_predictions + other.num_skip_predictions,
            num_skip_accurate=self.num_skip_accurate + other.num_skip_accurate,
            num_topk_skip_accurate=self.num_topk_skip_accurate + other.num_topk_skip_accurate,
            token_counts=new_token_counts,
        )

    def to_dict(self):
        _dict = {}
        for attr in self.__dataclass_fields__:
            _dict[attr] = getattr(self, attr)
        return _dict

    def get_percentages( self, as_string: bool = False):
        # Print top1 prediction accuracy
        pred       = self.num_predictions
        skip_pred  = self.num_skip_predictions
        pred      += int(pred==0)
        skip_pred += int(skip_pred==0)
        percent = {
            "base"      : (100 * self.num_accurate / pred),
            "topk"      : (100 * self.num_topk_accurate / pred),
            "skip"      : (100 * self.num_skip_accurate / skip_pred),
            "topk_skip" : (100 * self.num_topk_skip_accurate / skip_pred),
        }
        if as_string:
            percent = { k: ( "%.2f" % v ) for k, v in percent.items() }
        return percent

    def get_most_common_tokens(self, num_top_tokens: int) -> List[int]:
        if self.token_counts is None:
            return []
        top_tokens = self.token_counts.argpartition(-num_top_tokens )[-num_top_tokens:]
        top_tokens = top_tokens[
            np.argsort(self.token_counts[top_tokens])
        ][::-1]
        return top_tokens

######################################################################################
# Data Store Classes
######################################################################################

# function for converting nested dicts to dicts with slash notation
def flatten_dict(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# define data store class
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

    def __init__(self, input_data: dict = None, datasets: List[str] = None):
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

        # Misc
        self.misc = {}

        if not input_data is None:
            self.update(input_data)

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
            'areas': self.areas,
            'misc': self.misc,
        }

    def summary_wandb(self, is_first_run: bool = False):
        data = self.summary()
        if is_first_run:
            data['init_accuracy'] = data['accuracy']
        return flatten_dict(data, sep="/")

    def flat_summary(self):
        dataset_loss = {}
        dataset_accuracy = {}
        dataset_misc = {}
        for dataset in self.loss_data.keys():
            for key, val in self.loss_data[dataset].items():
                dataset_loss[dataset+'_'+key] = val
        for dataset in self.accuracy.keys():
            for key, val in self.accuracy[dataset].items():
                dataset_accuracy[dataset+'_'+key] = val
        for dataset in self.misc.keys():
            for key, val in self.misc[dataset].items():
                if isinstance(val, dict):
                    continue
                dataset_misc[dataset+'_'+key] = val

        areas = {f'area_{key}': value for key, value in self.areas.items()}

        return { **dataset_loss, **dataset_accuracy, **dataset_misc, **self.deletions, **areas }

    def __str__(self):
        return str(self.summary())

######################################################################################
# Run Data History Class
######################################################################################

class RunDataHistory:
    def __init__(self, datasets: List[str] = None, use_wandb: bool = True):
        self.datasets = datasets if datasets is not None else ['pile', 'code']
        self.history : List[RunDataItem] = []
        self.df = pd.DataFrame()
        self.use_wandb = use_wandb

    def add(self, item: Union[RunDataItem, dict]):
        # Add RunDataItem to history
        if not isinstance(item, RunDataItem) and isinstance(item, dict):
            item = RunDataItem(datasets=self.datasets).update(item)
        self.history.append(item)

        # Calculate EXTRACT prediction areas
        try:
            self.calculate_areas()
        except:
            pass
            # print("adding areas didn't work")
        item = self.history[-1]

        # Log to wandb
        if self.use_wandb:
            is_first_run = (len(self.history) == 1)
            wandb.log(item.summary_wandb(is_first_run))

        # save to pandas DataFrame
        self.df_append(item)

    def df_append(self, item: RunDataItem):
        new_data = pd.DataFrame({ k:[v] for k,v in item.flat_summary().items() })
        self.df = pd.concat([ self.df, new_data ], ignore_index=True )
        return self.df

    def calculate_areas(self):
        keys = RunDataItem.keys_accuracy
        areas = {}
        dataset_1, dataset_2 = self.datasets[:2]
        for k in keys:
            pile_data = [ run.accuracy[dataset_1][k] for run in self.history ]
            code_data = [ run.accuracy[dataset_2][k] for run in self.history ]

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

@dataclass
class ActivationSummary:
    """ Class for summarizing activations for each neuron.

    Attributes:
        sqrt: mean of square root of activations
        mean: mean of activations
        std: standard deviation of activations
        pos_mass: mean of positive activations
        pos_var: variance of positive activations
        neg_mass: mean of negative activations
        neg_var: variance of negative activations
        pos_count: fraction of positive activations
    """
    sqrt: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    pos_mass: torch.Tensor
    pos_var: torch.Tensor
    neg_mass: torch.Tensor
    neg_var: torch.Tensor
    pos_count: torch.Tensor

@dataclass
class ActivationSummaryHolder:
    """ Holder for multiple ActivationSummary objects."""
    orig: ActivationSummary
    loss_normed: ActivationSummary = None
    log_loss_normed: ActivationSummary = None
    misc: dict = None

    def __getitem__(self, key):
        return getattr(self, key)

@dataclass
class ActivationOverview:
    """Output from activation collection on multiple possible parts"""
    texts_viewed: int
    ff: Optional[ActivationSummaryHolder] = None
    attn: Optional[ActivationSummaryHolder] = None
    raw: Optional[dict] = None
    misc_data: Optional[dict] = None

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
        self.sqrt: Welford = Welford(dtype=self.dtype).detach()
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

    def add(self, data_point: Tensor):
        # Add mean and variance of data_point to all_activation
        self.n_points += 1
        self.all.add(data_point)
        self.sqrt.add(data_point.abs().sqrt())

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

    def summary(self, dtype: torch.dtype = torch.float32, allow_nan=False):
        if self.n_points == 0:
            if allow_nan:
                return ActivationSummary(0,0,0,0,0,0,0,0)
            raise ValueError('No data points added to ActivationCollector')

        return ActivationSummary(
            sqrt = self.sqrt.mean.to(dtype=dtype),
            mean = self.all.mean.to(dtype=dtype),
            std = self.all.var_s.to(dtype=dtype),
            pos_mass = self.pos.mean.to(dtype=dtype),
            pos_var = self.pos.var_s.to(dtype=dtype),
            neg_mass = self.neg.mean.to(dtype=dtype),
            neg_var = self.neg.var_s.to(dtype=dtype),
            pos_count = self.pos_counter / self.n_points,
        )

######################################################################################
# Pruning Config Data Class
######################################################################################

@dataclass
class PruningConfig:
    model_repo: str
    token_limit: int = None
    ff_frac: float = 0.1
    ff_eps: float = 0.001
    attn_frac: float = 0.0
    attn_eps: float = 1e-4
    dtype: str = "fp16"
    use_accelerator: bool = True
    model_device: Optional[str] = None
    mask_fn: Optional[str] = "step"

    collection_sample_size: int = 1e5
    eval_sample_size: int = 1e5
    topk = 10

    scoring_normalization: str = "orig" # "original" or "loss_normed"
    ff_scoring: str = "abs"

    attn_scoring: str = "abs"
    attn_mode: str = "pre-out"
    svd_attn: bool = False
    attn_prune_heads: bool = False
    attn_prune_heads_mode: str = "mean"
    do_attn_mean_offset: bool = False
    svd_combine_biases: bool = False

    focus: str = "pile_codeless"
    cripple: str = "code"
    additional_datasets: Tuple[str] = tuple()
    run_pre_test: bool = True
    recalculate_activations: bool = True

    wandb_entity: str = "seperability"
    wandb_project: str = "separability-compare"
    n_steps: int = 1
    wandb_run_name: Optional[str] = None
    misc: dict = None

    @property
    def model_size(self): # legacy code
        return self.model_repo

    @property
    def datasets(self):
        _datasets = list(set([
            *list(sorted([self.focus, self.cripple])),
            *self.additional_datasets
        ]))
        return _datasets

    def to_dict(self):
        obj = {}
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if attr_name.startswith("_"):
                continue
            if callable(attr):
                continue
            obj[attr_name] = attr
        return obj

    def arg_keys(self, exclude=None):
        if exclude is None:
            exclude = []
        filters = set([*exclude, "model_size", "datasets"])
        keys = []
        for key, value in self.to_dict().items():
            if key in filters:
                continue
            keys.append(key)
        return keys

    def arg_items(self, exclude=None):
        keys = self.arg_keys(exclude)
        return [ (key, getattr(self, key)) for key in keys ]

    def __setitem__(self, key, value):
        assert hasattr(self, key)
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __str__(self):
        return str(self.to_dict())
