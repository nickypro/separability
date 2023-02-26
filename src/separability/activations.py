"""
Code for getting attention activations and evaluating model. Includes specific
references to functions from texts.py, so is not included in model.py Model.
"""

import os
import datetime
# Import types for typed python
from typing import Optional, Union, Dict, Tuple
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

####################################################################################
# Code for Evaluating Model
####################################################################################

def evaluate( opt: Model,
        dataset_name: str,
        sample_size: int = 1e5,
        topk: int = 10,
        verbose: bool = False,
        dataset_texts_to_skip: int = 0,
    ):
    dataset, label, skip_eval = prepare( dataset_name )
    dataset = dataset.skip( dataset_texts_to_skip )
    out = opt.evaluate_dataset( dataset, k=topk, start_index=1,
        sample_size=sample_size, skip_eval=skip_eval, dataset_text_label=label,
        count_tokens=False, loading_bar_desc="%6s"%dataset_name )

    percent  = out['percent']
    loss     = round(float(out['loss']), 4)
    log_loss = round(float(out['log_loss']), 4)
    out['loss_data'] = {
        'loss': loss,
        'log_loss': log_loss,
    }

    if verbose:
        start = f' - {dataset_name}'
        print( f'{start} loss:', out['loss'] )
        print( f'{start} log loss:', out['log_loss'] )
        print( f'{start} no skip top{topk}:', '%.2f' % percent['topk'], '%')
        print( f'{start} w/ skip top{topk}:', '%.2f' % percent['topk_skip'], '%')
        print( f'{start} no skip:', '%.2f' % percent['base'], '%')
        print( f'{start} w/ skip:', '%.2f' % percent['skip'], '%')
        print()

    return out

def evaluate_all( opt: Model,
        sample_size: int = 1e5,
        datasets = None,
        topk: int = 10,
        verbose: bool = False,
        texts_to_skip: int = 0,
    ):
    if datasets is None:
        datasets = ['pile', 'code']

    out = { 'loss_data': {}, 'accuracy': {} }
    for dataset in datasets:
        dataset_out = evaluate(opt, dataset, sample_size, topk, verbose, texts_to_skip)

        out['loss_data'].update({ dataset: dataset_out['loss_data'] })
        out['accuracy'].update({  dataset: dataset_out['percent'] })

    return out

######################################################################################
# Code for counting both FF and Self-Attention activations
######################################################################################

def get_midlayer_activations( opt: Model,
        dataset_name: str,
        sample_size: int = 10000,
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

    # ff activation collector
    if do_ff:
        ff_shape = (opt.n_layers, opt.d_ff)
        ff_data = ActivationCollector( ff_shape, opt.output_device, collect_ff )

    # self-attention activation collector
    if do_attn:
        attn_shape = (opt.n_layers, opt.n_heads, opt.d_head)
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
                text_activations = opt.get_text_activations( input_ids=input_ids )
                residual_stream = opt.get_residual_stream(
                    text_activations=text_activations ).detach()

                # Get activations of self attention pre_out layer
                if do_attn:
                    attn_pre_out = opt.get_attn_pre_out_activations(
                        text_activations=text_activations, reshape=True ).detach()
                    attn_pre_out = einops.rearrange(attn_pre_out,
                        'layer token head pos -> token layer head pos')

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
                for token_index, attn_activation in enumerate(attn_pre_out):
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

def prune_and_evaluate( opt: Model,
        ff_prune_frac: float,
        attn_prune_frac: float,
        ff_eps: float,
        sample_size: int = 1e5,
        eval_size: int = 1e5,
        save: bool = False,
        cripple: str = "code",
        focus: str = "pile",
        **kwargs
    ):
    """
    Prune and evaluate the model

    Args:
        opt (Model): model to prune and evaluate
        ff_prune_frac (float): fraction of FF to prune
        attn_prune_frac (float): fraction of Attention to prune
        ff_eps (float): epsilon for FF pruning (to avoid division by 0).
        sample_size (int): number of samples to use for evaluation
        eval_size (int): number of samples to use for evaluation
        save (bool): whether to save the results to a file
        cripple (str): Which dataset to cripple. ("code", "pile")
        focus (str): Which dataset to focus. ("pile", "code")

    Returns:
        output (dict): dictionary to be added to pandas DataFrame.
    """
    # Find out what we are doing
    do_ff   = ff_prune_frac > 0
    do_attn = attn_prune_frac > 0
    if not do_ff and not do_attn:
        raise ValueError("Must prune at least one of FF or Attention")

    # Get midlayer activations of FF and ATTN
    datasets = [focus, cripple]
    focus_out   = get_midlayer_activations( opt, focus, sample_size, **kwargs )
    cripple_out = get_midlayer_activations( opt, cripple, sample_size, **kwargs )

    # Get the top fraction FF activations and prune
    if do_ff > 0:
        cripple_ff_count = cripple_out["ff"]["pos_count"]
        focus_ff_count = focus_out["ff"]["pos_count"]
        ff_rel_freq = ( cripple_ff_count / ( focus_ff_count + ff_eps ) ).cpu()
        ff_criteria, ff_threshold = get_top_frac( ff_rel_freq, ff_prune_frac )
        opt.delete_ff_keys( ff_criteria )

    # Get the top fraction of Attention activations and prune
    if do_attn > 0:
        attn_criteria, attn_threshold = choose_attn_heads_by_std( opt,
                focus_out["attn"], cripple_out["attn"], attn_prune_frac )
        opt.delete_attn_pre_out_heads( attn_criteria, focus_out["attn"]["mean"] )

    # Save the removals to file
    if save:
        tensor_data = {
            "ff_rel_freq": ff_rel_freq if do_ff else None,
            # FIXME: doesn't return attn_std_mean
            "attn_std_mean": attn_criteria if do_attn else None,
            "ff_frac": torch.tensor( ff_prune_frac ) if do_ff else None,
            "attn_frac": torch.tensor( attn_prune_frac ) if do_attn else None,
        }
        save_timestamped_tensor_dict( opt, tensor_data, "activation_metrics" )

    # Initialize the output dictionary
    data = RunDataItem()

    # Evaluate the model
    texts_to_skip = max( focus_out["texts_viewed"], cripple_out["texts_viewed"] )
    data.update(
        evaluate_all( opt, eval_size, datasets, texts_to_skip=texts_to_skip )
    )

    data.update({'deletions': {
        "ff_threshold": ff_threshold if do_ff else 0,
        "attn_threshold": attn_threshold if do_attn else 0,
        "ff_del": float( torch.sum(ff_criteria) ) if do_ff else 0,
        "attn_del": float( torch.sum(attn_criteria) ) if do_attn else 0,
    }})

    data.update({'deletions_per_layer': {
        'ff': ff_criteria.sum(dim=-1).tolist() if do_ff else [],
        'attn': attn_criteria.sum(dim=-1).tolist() if do_attn else [],
    }})

    return data

def prune_random( opt: Model,
        ff_frac: float,
        attn_frac: float,
        ff_pruned: Optional[np.ndarray] = None,
        attn_pruned: Optional[np.ndarray] = None
        ):
    """Prune a random fraction of FF and Attention weights
    Args:
        opt (Model): model to prune and evaluate
        ff_frac (float): fraction of FF to prune
        attn_frac (float): fraction of Attention to prune

    """
    if ff_pruned is None:
        ff_pruned = np.zeros( (opt.n_layers, opt.d_model), dtype=np.bool_ )
    if attn_pruned is None:
        attn_pruned = np.zeros( (opt.n_layers, opt.n_heads ), dtype=np.bool_ )

    n_ff_to_prune   = int( ff_frac   * opt.d_model )
    n_attn_to_prune = int( attn_frac * opt.n_heads )

    # First prune the FF
    if not ff_frac == 0:
        for layer in range( opt.n_layers ):
            # choose new ff neurons to prune
            indices = np.where(ff_pruned == 0)[0]
            random_indices = np.random.choice(indices, n_ff_to_prune, replace=False)
            ff_pruned[layer][random_indices] = 1

        # Prune the model
        opt.delete_ff_keys( ff_pruned )

    if not attn_frac == 0:
        for layer in range( opt.n_layers ):
            # choose new attention heads to prune
            indices = np.where(attn_pruned == 0)[0]
            random_indices = np.random.choice(indices, n_attn_to_prune, replace=False)
            attn_pruned[layer][random_indices] = 1

        # Prune the model
        opt.delete_ff_keys( attn_pruned )

    data_out = {
        "ff_del": n_ff_to_prune*opt.n_layers,
        "attn_del": n_attn_to_prune*opt.n_layers
    }
    return ff_pruned, attn_pruned, data_out

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

def get_attn_crossover( opt: Model,
        pile_out: Dict[str, Tensor],
        code_out: Dict[str, Tensor],
        eps: float = 1e-6,
    ):
    """
    Calculates the attention crossover between the pile and code activations.

    Args:
        opt (Model): OPT model with my special sauce modifications
        pile_out (Dict[str, Tensor]): pile activations
        code_out (Dict[str, Tensor]): code activations

    Returns:
        Dict[str, Tensor]:
            pile_means: mean activations of each neuron in pre-out on the pile
            pile_pos: positive mass
            pile_neg: negative mass
            code_means: mean activations of each neuron in pre-out on the pile
            code_pos: positive mass
            code_neg: negative mass
            crossover: The of probability mass on crossover between positive and
                negative mass from code compared to baseline pile activation
    """

    pile_means, pile_pos, pile_neg = \
        pile_out["mean"], pile_out["pos_mass"], pile_out["neg_mass"]
    code_means, code_pos, code_neg = \
        code_out["mean"], code_out["pos_mass"], code_out["neg_mass"]

    crossover_multiple = torch.ones((opt.n_layers, opt.n_heads))
    pos_code_rel_freq = code_pos / ( pile_pos + eps )
    neg_code_rel_freq = code_neg / ( pile_neg - eps )

    for layer in range( opt.n_layers ):
        for head in range( opt.n_heads ):
            # Relative probability mass in positive and negative directions
            pos_rel, _pos_index = torch.sort( pos_code_rel_freq[layer][head] )
            neg_rel, _neg_index = \
                torch.sort( neg_code_rel_freq[layer][head], descending=True )

            #cross-over position
            i = ( neg_rel > pos_rel ).sum()

            crossover =  ( pos_rel[i-1] + pos_rel[i] + neg_rel[i-1] + neg_rel[i] )/4
            crossover_multiple[layer][head] = crossover

    # Save data in a dict
    data = {
        'pile_means' : pile_means,
        'pile_pos'   : pile_pos,
        'pile_neg'   : pile_neg,
        'code_means' : code_means,
        'code_pos'   : code_pos,
        'code_neg'   : code_neg,
        'crossover_multiple' : crossover_multiple
    }
    return data


def choose_attn_heads_by_std( opt: Model,
        focus_out: Dict[str, Tensor],
        cripple_out: Dict[str, Tensor],
        top_frac: float,
        eps: float = 1e-6,
    ):
    """
    Calculates the attention crossover between the pile and code activations.

    Args:
        opt (Model): OPT model with my special sauce modifications
        pile_out (Dict[str, Tensor]): pile activations
        code_out (Dict[str, Tensor]): code activations

    Returns:
        Dict[str, Tensor]:
    """
    focus_stds   = focus_out["std"]
    cripple_stds = cripple_out["std"]
    std_ratios = cripple_stds / ( focus_stds + eps )
    std_ratio_means = std_ratios.mean(dim=-1)

    return get_top_frac( std_ratio_means, top_frac )


def delete_attn_and_evaluate( opt: Model,
        frac_removed: float,
        sample_size: int = 1e5,
        eval_size: int = 1e5,
        eps: float = 1e-6,
        pile_out: Optional[Dict[str, Tensor]] = None,
        code_out: Optional[Dict[str, Tensor]] = None,
        make_plots: bool = False,
        **kwargs
        ):
    """Gets how much more probability mass the median activation of code has for an
    attention head neuron compared to activations in the pile.

    Args:
        opt (Model): The model to run
        frac_removed (float): The fraction of attention heads removed from the model
        sample_size (int, optional): token sample size to collect data for.
            Defaults to 1e5.
        eval_size (int, optional): token sample size to use for evaluating the model.
        eps (float, optional): epsilon for numerical stability. Defaults to 1e-6.
        pile_out (Dict[str, Tensor], optional): pile activations output from
            running get_attn_activations. Defaults to None (i.e: compute here).
        code_out (Dict[str, Tensor], optional): code activations output from
            running get_attn_activations. Defaults to None (i.e: compute here).

    Returns:
        data: Dict of data from the evaluation after removing attention heads.

    """
    if pile_out is None:
        pile_out = get_attn_activations(opt, 'pile', sample_size, **kwargs)
    if code_out is None:
        code_out = get_attn_activations(opt, 'code', sample_size, **kwargs)

    # extract data on attention crossover from activations
    attn_data = get_attn_crossover(opt, pile_out, code_out, eps=eps)

    # Choose and delete attention heads
    removals, threshold = get_top_frac(attn_data["crossover_multiple"], frac_removed)
    opt.delete_attn_pre_out_heads( removals, attn_data["pile_means"] )

    if make_plots:
        # Choose Attn Heads to Remove
        print( "min: %.2f" % float(attn_data['crossover_multiple'].min()) )
        print( "max: %.2f" % float(attn_data['crossover_multiple'].max()) )
        log_crossover = ( torch.log2(attn_data['crossover_multiple']) )

        # Plot Attn Heads
        _fig, ax = plt.subplots(1, 2)
        ax[0].imshow( removals )
        ax[1].imshow( log_crossover )
        plt.show()

    # Evaluate
    data = RunDataItem()
    data.update( evaluate_all(opt, eval_size) )
    data.update({'deletions': {
        'attn_del': int( removals.sum().item() ),
        'attn_threshold': threshold,
    }})

    return data

attn_data_keys = ["crossover_multiple", "pile_means"]

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
