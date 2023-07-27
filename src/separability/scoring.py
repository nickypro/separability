from typing import Dict, Callable

import torch
from torch import Tensor

from . import Model
from .data_classes import ActivationSummary

#####################################################################################
# Scoring Functions for Activations
#####################################################################################

def score_indices_by_freq( opt: Model,
        focus_out: ActivationSummary,
        cripple_out: ActivationSummary,
        eps: float = 1e-3,
    ):
    cripple_count  = cripple_out.pos_count
    focus_count = focus_out.pos_count
    ratios = cripple_count / ( focus_count + eps )
    return ratios

def score_indices_by_sqrt( opt: Model,
        focus_out: ActivationSummary,
        cripple_out: ActivationSummary,
        eps: float = 1e-6,
    ):
    focus_stds   = focus_out.sqrt
    cripple_stds = cripple_out.sqrt
    ratios = cripple_stds / ( focus_stds + eps )
    return ratios

def score_indices_by_abs( opt: Model,
        focus_out: ActivationSummary,
        cripple_out: ActivationSummary,
        eps: float = 1e-6,
    ):
    focus_mean_abs   = focus_out.pos_mass + focus_out.neg_mass.abs()
    cripple_mean_abs = cripple_out.pos_mass + cripple_out.neg_mass.abs()
    ratios = cripple_mean_abs / ( focus_mean_abs + eps )

    return ratios

def score_indices_by_std( opt: Model,
        focus_out: ActivationSummary,
        cripple_out: ActivationSummary,
        eps: float = 1e-6,
    ):
    """
    Gets indices with highest ratio between Standard Deviations.

    Args:
        opt (Model): OPT model with my special sauce modifications
        focus_out (Dict[str, Tensor]): focus dataset neuron activations
        cripple_out (Dict[str, Tensor]): cripple dataset neuron activations
        top_frac (float): Fraction of neurons to return
        eps (float): Epsilon for numerical stability

    Returns:
        removal_indices (Tensor)
        threshold (float)
    """
    focus_stds   = focus_out.std
    cripple_stds = cripple_out.std
    ratios = cripple_stds / ( focus_stds + eps )

    return ratios

def score_indices_by_rms( opt: Model,
        focus_out: ActivationSummary,
        cripple_out: ActivationSummary,
        eps: float = 1e-6,
    ):
    focus_rms   = torch.sqrt( focus_out.std**2 + focus_out.mean**2 )
    cripple_rms = torch.sqrt( cripple_out.std**2 + cripple_out.mean**2 )
    ratios = cripple_rms / ( focus_rms + eps )
    return ratios

def score_indices_by_mean( opt: Model,
        focus_out: ActivationSummary,
        cripple_out: ActivationSummary,
        eps: float = 1e-6,
    ):
    focus_mean   = focus_out.mean
    cripple_mean = cripple_out.mean
    diff = torch.abs( cripple_mean - focus_mean )
    return diff

def score_indices_randomly( opt: Model,
        focus_out: ActivationSummary,
        cripple_out: ActivationSummary,
        eps: float = 1e-6,
    ):
    return torch.randn(focus_out.mean.shape, device=focus_out.mean.device)

# Combine into a single callable

def score_indices_by(key: str) -> Callable:
    """Get the scoring function we want to use.

    Args:
        key (str): The name of the scoring function to use. Options:
            'freq': Importance = number of times activation is positive
            'abs': Mean absolute activation from zero.
            'sqrt': Mean square root activation from zero.
            'std': Standard deviation of activation from mean.
            'rms': Root Mean Square activation from zero.
            'mean': difference between mean activations.
            'rand': Randomly generated scores.

    Returns:
        scoring_func (Callable): The scoring function we want to use.
    """
    scoring_map = {
        'freq': score_indices_by_freq,
        'abs':  score_indices_by_abs,
        'sqrt': score_indices_by_sqrt,
        'std':  score_indices_by_std,
        'rms':  score_indices_by_rms,
        'mean': score_indices_by_mean,
        'rand': score_indices_randomly,
    }
    return scoring_map[key]
