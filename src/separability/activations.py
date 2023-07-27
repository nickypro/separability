"""
Code for getting attention activations and evaluating model. Includes specific
references to functions from texts.py, so is not included in model.py Model.
"""

import os
import datetime
# Import types for typed python
from typing import Optional, Union, Dict, Tuple, List, Callable
from torch import Tensor

import torch
import numpy as np
import einops
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import from this project
from .model import Model
from .texts import prepare
from .data_classes import RunDataItem, ActivationCollector
from .eval import evaluate, evaluate_all

######################################################################################
# Code for counting both FF and Self-Attention activations
######################################################################################

def get_midlayer_activations( opt: Model,
        dataset_name: str,
        sample_size: int = 10000,
        attn_mode: str = "pre-out",
        check_accuracy: bool = False,
        k: int = 10,
        check_skips: bool = False,
        calculate_ff: bool = True,
        calculate_attn: bool = True,
        collect_ff: bool = False,
        collect_attn: bool = False,
        use_ff_activation_function: bool = True,
    ):
    """Gets the number of activations of the midlayer ('key' layer) of MLPs for
    each layer, as well as for the pre_out layer of attention for each layer.

    Args:
        opt (Model): my special sauce opt model
        dataset_name (str): 'code' or 'pile'
        sample_size (int, optional): number of tokens to sample. Defaults to 10000.
        num_samples (int, optional): number of times to run. Defaults to 1.
        check_accuracy (bool, optional): whether to only look at accurate outputs.
            Defaults to False.
        k (int, optional): top k to check when looking at accuracy. Defaults to 10.
        check_skips (bool, optional): whether to skip most frequent tokens.
            Defaults to False.
        calculate_ff (bool, optional): whether to calculate the number of activations
            of the midlayer of MLPs. Defaults to True.
        calculate_attn (bool, optional): whether to calculate self-attention
            activation means and masses. Defaults to True.
        collect_ff (bool, optional): whether to collect all ff activations.
            Defaults to False.
        collect_attn (bool, optional): whether to collect all attn pre out
            activations. Defaults to False.
        use_ff_activation_function (bool, optional): whether to use the activation
            function on the ff activations. Defaults to True.

    Returns:
        Dict:
            "ff" (dict: ActivationCollector.summary):
                tensor shapes (n_layers, d_ff)
            "attn" (dict: ActivationCollector.summary):
                tensor shapes: (n_layers, d_attn, d_ff):
            "raw":
                "ff": Tensor[n_tokens, n_layers, d_ff]]
                "attn": Tensor[n_tokens, n_layers, n_head, d_head]
                "criteria": Tensor[n_tokens]
            "texts_viewed" (int): number of texts viewed in the dataset

    ActivationCollector.summary:
        "mean": Tensor[shape]
        "std": Tensor[shape]
        "pos_mass": Tensor[shape]
        "pos_var": Tensor[shape]
        "neg_mass": Tensor[shape]
        "neg_var": Tensor[shape]
        "pos_count": Tensor[shape]
    """
    dataset, label, skip_eval = prepare( dataset_name )
    do_ff   = calculate_ff   or collect_ff
    do_attn = calculate_attn or collect_attn
    if attn_mode not in ["pre-out", "value"]:
        raise NotImplementedError("attn_mode must be 'pre-out' or 'value'")

    # ff activation collector
    if do_ff:
        ff_shape = (opt.cfg.n_layers, opt.cfg.d_mlp)
        ff_data = ActivationCollector( ff_shape, opt.output_device, collect_ff )

    # self-attention activation collector
    if do_attn:
        attn_shape = (opt.cfg.n_layers, opt.cfg.n_heads, opt.cfg.d_head)
        attn_data = ActivationCollector( attn_shape, opt.output_device, collect_attn )

    if collect_ff or collect_attn:
        criteria_raw = []

    if not (calculate_ff or calculate_attn or collect_ff or collect_attn):
        raise ValueError("Must calculate or collect either ff or attn."
                        + "Otherwise, use evaluate_all() instead")

    # Prepare skip ids if they are being used
    if check_skips:
        skip_ids = set()
        for skip_string in skip_eval:
            skip_id = int( opt.get_ids( skip_string ).squeeze()[-1] )
            skip_ids.add( skip_id )

    # Number of tokens viewed counter (that meet criteria)
    curr_count = 0
    texts_viewed = 0

    with tqdm(total=sample_size) as pbar:
        for data in dataset:
            texts_viewed += 1
            text = data[label]
            # Get all necessary activations
            with torch.no_grad():
                input_ids = opt.get_ids( text ).detach()
                ids = input_ids.squeeze().detach()

                # Skip if there is only 1 token
                if len(ids.shape) == 0:
                    continue

                text_activations = opt.get_text_activations( input_ids=input_ids )
                residual_stream = opt.get_residual_stream(
                    text_activations=text_activations ).detach()

                # Get activations of self attention
                if do_attn and attn_mode == "pre-out":
                    attn_activations = opt.get_attn_pre_out_activations(
                        text_activations=text_activations, reshape=True ).detach()
                    attn_activations = einops.rearrange(attn_activations,
                        'layer token head pos -> token layer head pos')

                if do_attn and attn_mode == "value":
                    attn_activations = opt.get_attn_value_activations(
                        text_activations=text_activations, reshape=True ).detach()
                    attn_activations = einops.rearrange(attn_activations,
                        'layer token (head pos) -> token layer head pos',
                        head=opt.cfg.n_heads, pos=opt.cfg.d_head)

                # Get activations of FF mid layer
                if do_ff:
                    ff_keys = opt.get_ff_key_activations(
                        residual_stream=residual_stream,
                        use_activation_function=use_ff_activation_function ).detach()
                    ff_keys = einops.rearrange( ff_keys,
                        'layer token pos -> token layer pos')

            # Initialize criteria for counting the token activation
            criteria = torch.ones_like( ids, dtype=torch.bool ).detach()

            # (Optional) Check if prediction is accurate enough to count
            if check_accuracy:
                logits = opt.unembed( residual_stream[-1] ).detach()
                top_k_tokens = opt.top_k_tokens( logits, k=k ).squeeze()

                for index in range(len(ids)-1):
                    criteria[index] *= (ids[index+1] in top_k_tokens[index])

            # (Optional) Choose a set of token ids to skip
            if check_skips:
                for index in range(len(ids)-1):
                    criteria[index] *= (ids[index+1] in skip_ids)

            # Count the number of activations in FF
            if do_ff:
                for token_index, ff_activation in enumerate(ff_keys):
                    if not criteria[token_index]:
                        continue
                    ff_data.add( ff_activation )

            # Count the number of activations in Self-Attention
            if do_attn:
                for token_index, attn_activation in enumerate(attn_activations):
                    if not criteria[token_index]:
                        continue
                    attn_data.add( attn_activation )

            if collect_ff or collect_attn:
                for criterion in criteria:
                    criteria_raw.append( criterion.cpu() )

            # Keep track of number of tokens looked at
            num_valid_tokens = criteria.sum()
            curr_count += num_valid_tokens
            pbar.update( int(num_valid_tokens) )

            if curr_count > sample_size:
                break

    output = {
        "texts_viewed": texts_viewed,
    }

    # Summary information about activations
    if calculate_ff:
        output["ff"]   = ff_data.summary(dtype=opt.dtype)
    if calculate_attn:
        output["attn"] = attn_data.summary(dtype=opt.dtype)


    # Raw activations of data
    if collect_ff or collect_attn:
        output["raw"] = { "criteria": torch.stack(criteria_raw) }
    if collect_ff:
        output["raw"]["ff"] = ff_data.get_raw()
    if collect_attn:
        output["raw"]["attn"] = attn_data.get_raw()

    return output

def get_top_frac( values_tensor: Tensor, top_frac: float ) -> Tuple[Tensor, float]:
    """
    Return top-k values and their fraction

    Args:
        values_tensor (Tensor): tensor of values
        top_frac (float): fraction of top-k values to return

    Returns:
        criteria (Tensor): tensor with 1s for top-k values, 0s otherwise
        threshold (float): minimum value to be in the top-k values
    """
    # Get the number of entries in the tensor, and the number of entries to get
    shape = values_tensor.shape
    n_entries = np.prod(shape)
    k = int( top_frac * n_entries )

    # Get the top k values
    topk_values = torch.topk( values_tensor.flatten(), k,
        dim=-1, largest=True, sorted=False )

    # Create a criteria tensor with value 1 for all values in topk_values
    criteria = torch.zeros( n_entries, dtype=torch.bool )
    criteria[ topk_values.indices ] = True
    criteria = criteria.reshape( shape )

    # Get the threshold value, the value above which all values are in topk_values
    threshold = float( topk_values.values.flatten().min() )

    return criteria, threshold

####################################################################################
# Code for getting attention activations
####################################################################################

def get_attn_activations( opt: Model,
        dataset_name: str,
        sample_size: int = 10000,
        check_accuracy: bool = True,
        k: int = 10,
        check_skips: bool = False,
    ):
    """gets the mean activations and the probability mass of positive and negative
    activations for each pre-out neuron in an attention layer.
    Shorthand for get_midlayer_activations for attention only.

    Args:
        opt (Model): OPT model with my special sauce modifications
        dataset_name (str): 'pile' or 'code'
        sample_size (int, optional): Number of tokens to sample. Defaults to 10000.
        check_accuracy (int, optional): Whether to filter activations to the
            cases where it is accurate. Defaults to False.
        k (int, optional): Top-k accuracy check if check_accuracy is True.
            Defaults to 10.
        check_skips (bool, optional): Whether to filter against the most common
            tokens. Defaults to False.

    Returns:
        Dict:
            means: Mean activation of each pre-out neuron
            stds: Standard deviation of each pre-out neuron
            pos: Probability mass of positive activations
            neg: Probability mass of negative activations
    """

    output = get_midlayer_activations( opt,
        dataset_name=dataset_name,
        sample_size=sample_size,
        check_accuracy=check_accuracy,
        k=k,
        check_skips=check_skips,
        calculate_ff=False,
        calculate_attn=True
    )

    return output['attn']

#####################################################################################
# "Choosing Functions"
#####################################################################################

def choose_attn_heads_by_mean( opt: Model,
        attn_scores: Tensor,
        top_frac: float,
    ):
    std_ratio_medians = torch.quantile(
        attn_scores.to(dtype=torch.float32), q=0.5, dim=-1)
    return get_top_frac(std_ratio_medians, top_frac)

def choose_attn_heads_by_median( opt: Model,
        attn_scores: Tensor,
        top_frac: float,
    ):
    std_ratio_means = attn_scores.mean(dim=-1)
    return get_top_frac(std_ratio_means, top_frac)

def choose_attn_heads_by(key: str):
    choosing_map = {
        'mean': choose_attn_heads_by_mean,
        'median': choose_attn_heads_by_median,
    }
    return choosing_map[key]

def save_timestamped_tensor_dict( opt: Model,
        data: Dict[str, Tensor],
        name: str ):
    now = datetime.datetime.now().strftime( "%Y-%m-%d_%H:%M:%S" )
    os.makedirs( f'tmp/{opt.model_size}', exist_ok=True )
    filename = f'tmp/{opt.model_size}/{opt.model_size}-{name}-{now}.pt'
    torch.save( data, filename )
    print( f'Saved {filename} to {opt.model_size}' )
    return filename

####################################################################################
# Look at FF Key activations
####################################################################################

def count_ff_key_activations( opt: Model,
        dataset_name: str,
        sample_size: int = 10000,
        check_accuracy: bool = False,
        k: int = 10,
        check_skips: bool = False
    ):
    """Gets the number of activations of the midlayer ('key' layer) of MLPs for
    each layer.
    Shorthand for get_midlayer_activations with the 'key' layer only.

    Args:
        opt (Model): my special sauce opt model
        dataset_name (str): 'code' or 'pile'
        sample_size (int, optional): number of tokens to sample. Defaults to 10000.
        num_samples (int, optional): number of times to run. Defaults to 1.
        check_accuracy (bool, optional): whether to only look at accurate outputs.
            Defaults to False.
        k (int, optional): top k to check when looking at accuracy. Defaults to 10.
        check_skips (bool, optional): whether to skip most frequent tokens.
            Defaults to False.

    Returns:
        counters (Tensor): Tensor containing activation frequency of every ff
            mid layer activation
    """
    output = get_midlayer_activations( opt,
        dataset_name=dataset_name,
        sample_size=sample_size,
        check_accuracy=check_accuracy,
        k=k,
        check_skips=check_skips,
        calculate_ff=True,
        calculate_attn=False
    )

    return output['ff']

def save_numpy_ff( opt: Model,
        freq_multiple: float,
        array: np.ndarray,
        name: str
    ):
    filename = f'tmp/{opt.model_size}/{opt.model_size}-ff-{freq_multiple}x-{name}.npy'
    os.makedirs( f'tmp/{opt.model_size}', exist_ok=True )
    with open(filename, 'wb') as f:
        np.save(f, np.array(array) )
    print("saved successfully")

# pylint: disable=too-many-arguments, too-many-locals
def delete_ff_and_evaluate(
        opt: Model,
        top_frac: float = 0.02,
        eps: float = 1e-2,
        counter_sample_size: int = 5e4,
        eval_sample_size: int = 1e5,
        pile_counters: Optional[Union[Tensor, str]] = None,
        code_counters: Optional[Union[Tensor, str]] = None,
        save_files: bool = True,
        ):

    # Get counts of activations of MLP middle layers
    save_pile = (True and save_files)
    save_code = (True and save_files)
    if isinstance( pile_counters, str ):
        pile_counters = torch.tensor( np.load( pile_counters ), dtype=torch.float32 )
        save_pile = False
    if isinstance( code_counters, str ):
        code_counters = torch.tensor( np.load( code_counters ), dtype=torch.float32 )
        save_code = False

    if pile_counters is None:
        pile_counters = count_ff_key_activations( opt, 'pile',
            sample_size=counter_sample_size, check_accuracy=True )
    if code_counters is None:
        code_counters = count_ff_key_activations( opt, 'code',
            sample_size=counter_sample_size, check_accuracy=True )

    pile_counters = pile_counters.squeeze().cpu()
    code_counters = code_counters.squeeze().cpu()

    # Get Relative Frequency of Activations
    rel_freq = ( code_counters / ( pile_counters + eps ) ).flatten()

    # Delete the top fraction of most frequent activations
    ff_criteria, ff_threshold = get_top_frac( rel_freq, top_frac )

    # Give summary of how many will be removed in each layer
    sums = [ x.sum() for x in ff_criteria.detach().numpy() ]
    num_removed = np.sum(sums)
    print( f"%5d - {sums}" % num_removed )

    # Finally, delete the keys
    opt.delete_ff_keys( ff_criteria )

    # Save the indices of the deleted keys, but if unsuccessful, don't crash
    try:
        # Save the indices that were deleted into the timestamped file
        print("saving files...")
        now = datetime.datetime.now().strftime( "%Y-%m-%d_%H:%M:%S" )
        if save_files:
            save_numpy_ff( opt, top_frac, ff_criteria.cpu(),     f'criteria_{now}' )
        if save_pile:
            save_numpy_ff( opt, top_frac, pile_counters.cpu(), f'counters-pile_{now}')
        if save_code:
            save_numpy_ff( opt, top_frac, code_counters.cpu(), f'counters-code_{now}')

    # pylint: disable=broad-except
    except Exception as err:
        print("# WARNING: Sadly, did not save ff activations :( ")
        print(err)

    # See the effect deletion has on performance
    data = RunDataItem()
    data.update( evaluate_all( opt, eval_sample_size ) )
    data.update({'deletions': {
        'ff_del': num_removed,
        'ff_threshold': ff_threshold,
    }})
    return data
