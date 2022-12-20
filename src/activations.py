"""
Code for getting attention activations and evaluating model. Includes specific
references to functions from texts.py, so is not included in model.py Model.
"""

import os
import datetime
# Import types for typed python
from typing import Optional, Union, Dict
from torch import Tensor

import torch
import numpy as np
from welford_torch import Welford
import einops
from tqdm import tqdm

# Import from this project
from model import Model
from texts import prepare

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
        count_tokens=False )

    percent = out['percent']
    out['loss'] = round(float(out['loss']), 4)
    out['log_loss'] = round(float(out['log_loss']), 4)

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
        topk: int = 10,
        verbose: bool = False
    ):
    pile_out = evaluate( opt, 'pile', sample_size, topk, verbose )
    code_out = evaluate( opt, 'code', sample_size, topk, verbose )

    percentages = {
        "pile_loss": pile_out['loss'],
        "pile_log_loss": pile_out['log_loss'],
        "code_loss": code_out['loss'],
        "code_log_loss": code_out['log_loss'],
    }
    percentages.update({ ('pile_'+k): v for (k,v) in pile_out['percent'].items() })
    percentages.update({ ('code_'+k): v for (k,v) in code_out['percent'].items() })
    return percentages

####################################################################################
# Code for getting attention activations
####################################################################################

def get_attn_activations( opt: Model,
        dataset_name: str,
        sample_size: int = 10000,
        token_limit: Optional[int] = None,
        check_accuracy: bool = True,
        k: int = 10,
        check_skips: bool = False,
    ):
    """gets the mean activations and the probability mass of positive and negative
    activations for each pre-out neuron in an attention layer.

    Args:
        opt (Model): OPT model with my special sauce modifications
        dataset_name (str): 'pile' or 'code'
        sample_size (int, optional): Number of tokens to sample. Defaults to 10000.
        token_limit (Optional[int], optional): Maximum text size limit,
            mainly for small models. Defaults to None.
        check_accuracy (int, optional): Whether to filter activations to the
            cases where it is accurate. Defaults to False.
        k (int, optional): Top-k accuracy check if check_accuracy is True.
            Defaults to 10.
        check_skips (bool, optional): Whether to filter against the most common
            tokens. Defaults to False.

    Returns:
        means: Mean activation of each pre-out neuron
        pos_mass: Probability mass of positive activations
        neg_mass: Probability mass of negative activations
    """
    dataset, label, skip_eval = prepare( dataset_name )

    # See: Welford's Online Algorithm
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    all_activations = Welford().detach() # to get mean & variance of all activations
    neg = Welford().detach() # to get mean ("negative mass") of negative activations
    pos = Welford().detach() # to get mean ("positive mass") of positive activations

    curr_count = 0

    # Get the ids of the tokens to skip
    if check_skips:
        skip_ids = set()
        for skip_string in skip_eval:
            skip_id = int( opt.get_ids( skip_string ).squeeze()[-1] )
            skip_ids.add( skip_id )

    with tqdm(total=sample_size) as pbar:
        for data in dataset:
            # Get the text and its ids, and run it into the model
            text = data[label]
            with torch.no_grad():
                input_ids = opt.get_ids( text, limit=token_limit ).detach()
                text_activations = opt.get_text_activations( input_ids=input_ids )
                attn_pre_out = opt.get_attn_pre_out_activations(
                    text_activations=text_activations, reshape=True ).detach()
                # Keep track of criteria for counting the token activation
                ids = input_ids.squeeze()
                criteria = torch.ones_like( ids, dtype=torch.bool ).detach()

            # check if prediction is accurate enough to count (top-k accuracy)
            if check_accuracy:
                residual_stream = opt.get_residual_stream(
                    text_activations=text_activations )
                logits = opt.unembed( residual_stream[-1] )
                top_k_tokens = opt.top_k_tokens( logits, k=k ).squeeze()

                for index in range(len(ids)-1):
                    criteria[index] *= (ids[index+1] in top_k_tokens[index])

            # check for tokens is in skip list
            if check_skips:
                for index in range(len(ids)-1):
                    criteria[index] *= (ids[index+1] in skip_ids)

            # Rearrange so we can add all activations for each token
            # to the Welford accumulator in form [layer head pos]
            attn_pre_out = einops.rearrange(attn_pre_out,
                'layer token head pos -> token layer head pos')

            for token_index, activation in enumerate(attn_pre_out):
                if not criteria[token_index]:
                    continue
                all_activations.add( activation )
                pos.add( activation * ( activation > 0 ) )
                neg.add( activation * ( activation < 0 ) )

            # Keep track of number of tokens looked at
            num_valid_tokens = criteria.sum().cpu()
            pbar.update( int(num_valid_tokens) )
            curr_count += num_valid_tokens

            if curr_count > sample_size:
                break

    means = all_activations.mean.cpu()
    pos_mass = pos.mean.cpu()
    neg_mass = neg.mean.cpu()

    return means, pos_mass, neg_mass

def calculate_attn_crossover( opt: Model,
        sample_size: int = 1e5,
        token_limit: Optional[int] = None,
        eps: float = 1e-6,
        **kwargs
        ):
    """Gets how much more probability mass the median activation of code has for an
    attention head neuron compared to activations in the pile.

    Args:
        opt (Model): The model to run
        sample_size (int, optional): token sample size to collect data for.
            Defaults to 1e5.
        token_limit (Optional[int], optional): limit to number of tokens in a text,
            mainly for smaller models. Defaults to None.

    Returns:
        data: Dictionary containing information about activations
            pile_means: mean activations of each neuron in pre-out on the pile
            pile_pos: positive mass
            pile_neg: negative mass
            code_means: mean activations of each neuron in pre-out on the pile
            code_pos: positive mass
            code_neg: negative mass
            crossover: The of probability mass on crossover between positive and
                negative mass from code compared to baseline pile activation
    """
    pile_out = get_attn_activations(opt, 'pile', sample_size, token_limit, **kwargs)
    code_out = get_attn_activations(opt, 'code', sample_size, token_limit, **kwargs)
    pile_means, pile_pos, pile_neg = pile_out
    code_means, code_pos, code_neg = code_out

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

attn_data_keys = ["crossover_multiple", "pile_means"]

def save_torch_attn( opt: Model,
        attn_data: Dict[str, Tensor],
        name: Optional[str] = None
    ):
    if name is None:
        name = datetime.datetime.now().strftime( "%Y-%m-%d_%H:%M:%S" )
    os.makedirs( f'tmp/{opt.model_size}', exist_ok=True )
    filename = f'tmp/{opt.model_size}/{opt.model_size}-attn_data-{name}.pt'
    torch.save( attn_data, filename )
    print( f'Saved {filename} to {opt.model_size}' )
    return filename

####################################################################################
# Look at FF Key activations
####################################################################################

def count_ff_key_activations( opt: Model,
        dataset_name: str,
        sample_size: int = 10000,
        token_limit: int = None,
        check_accuracy: bool = False,
        k: int = 10,
        check_skips: bool = False
    ):
    """Gets the number of activations of the midlayer ('key' layer) of MLPs for
    each layer.

    Args:
        opt (Model): my special sauce opt model
        dataset_name (str): 'code' or 'pile'
        sample_size (int, optional): number of tokens to sample. Defaults to 10000.
        token_limit (int, optional): limit to text token length. Defaults to None.
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
    dataset, label, skip_eval = prepare( dataset_name )
    counter = torch.zeros( (opt.n_layers, opt.d_ff) ).to( opt.device )
    curr_count = 0

    if check_skips:
        skip_ids = set()
        for skip_string in skip_eval:
            skip_id = int( opt.get_ids( skip_string ).squeeze()[-1] )
            skip_ids.add( skip_id )

    with tqdm(total=sample_size) as pbar:
        for data in dataset:
            text = data[label]
            with torch.no_grad():
                input_ids = opt.get_ids( text, limit=token_limit ).detach()
                residual_stream = opt.get_residual_stream( input_ids=input_ids )
                ids = input_ids.squeeze().detach()
                ff_keys = opt.get_ff_key_activations(
                    residual_stream=residual_stream ).detach()
                # Criteria for counting the token activation
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

            # Rearrange ff_keys to be [layer, token, activation]
            ff_keys = einops.rearrange( ff_keys,
                'layer token pos -> token layer pos')

            # Count the number of activations
            for token_index, ff_activation in enumerate(ff_keys):
                if not criteria[token_index]:
                    continue
                counter += ( ff_activation != 0 )

            # Keep track of number of tokens looked at
            num_valid_tokens = criteria.sum()
            curr_count += num_valid_tokens
            pbar.update( int(num_valid_tokens) )

            if curr_count > sample_size:
                counter = counter / curr_count
                break

    return counter.detach()

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

    # Get Relative Frequenct of Activations
    rel_freq = ( code_counters / ( pile_counters + eps ) ).flatten()

    # Delete the top fraction of most frequent activations
    k = int( top_frac * opt.n_layers * opt.d_ff )
    rel_topk = torch.topk( rel_freq, k, dim=-1, largest=True, sorted=False )
    ff_criterion = torch.zeros( (opt.n_layers * opt.d_ff) )
    ff_criterion[ rel_topk.indices ] = 1
    ff_criterion = ff_criterion.reshape( (opt.n_layers, opt.d_ff) )

    # Give summary of how many will be removed in each layer
    sums = [ x.sum() for x in ff_criterion.detach().numpy() ]
    num_removed = np.sum(sums)
    print( f"%5d - {sums}" % num_removed )

    # Finally, delete the keys
    opt.delete_ff_keys( ff_criterion )

    # Save the indices of the deleted keys, but if unsuccessful, don't crash
    try:
        # Save the indices that were deleted into the timestamped file
        print("saving files...")
        now = datetime.datetime.now().strftime( "%Y-%m-%d_%H:%M:%S" )
        if save_files:
            save_numpy_ff( opt, top_frac, ff_criterion.cpu(),     f'criterion_{now}' )
        if save_pile:
            save_numpy_ff( opt, top_frac, pile_counters.cpu(), f'counters-pile_{now}')
        if save_code:
            save_numpy_ff( opt, top_frac, code_counters.cpu(), f'counters-code_{now}')

    # pylint: disable=broad-except
    except Exception as err:
        print("# WARNING: Sadly, did not save ff activations :( ")
        print(err)

    # See the effect deletion has on performance
    data = evaluate_all( opt, eval_sample_size )
    data['removed'] = num_removed
    return data
