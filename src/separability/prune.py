from typing import Optional, List

import numpy as np
import torch
import wandb
import copy

from .model import Model
from .data_classes import PruningConfig, RunDataHistory, \
                          RunDataItem, ActivationOverview
from .eval import evaluate_all
from .scoring import score_indices_by, score_indices
from .activations import get_midlayer_activations, get_top_frac, \
    choose_attn_heads_by, save_timestamped_tensor_dict
from .texts import prepare

def prune_and_evaluate(
        opt: Model,
        pruning_config: PruningConfig,
        focus_out: Optional[dict] = None,
        cripple_out: Optional[dict] = None,
        iteration: Optional[int] = None,
    ):
    """
    Prune and evaluate the model

    Args:
        opt (Model): model to prune and evaluate
        pruning_config (PruningConfig): config for pruning
        focus_out (dict): output of get_midlayer_activations for focus dataset
        cripple_out (dict): output of get_midlayer_activations for cripple dataset
        iteration (int): iteration number for when activations are not recalculated

    Returns:
        output (RunDataItem): Eval data to add to RunDataHistory.
    """
    c = copy.deepcopy(pruning_config)

    # Find out what we are doing
    do_ff   = pruning_config.ff_frac > 0
    do_attn = pruning_config.attn_frac > 0
    if not do_ff and not do_attn:
        raise ValueError("Must prune at least one of FF or Attention")
    if do_attn and pruning_config.attn_mode not in ["pre-out", "value"]:
        raise NotImplementedError("attn_mode must be 'pre-out' or 'value'")

    # Get midlayer activations of FF and ATTN
    if pruning_config.recalculate_activations:
        focus_out   = get_midlayer_activations( opt, pruning_config.focus,
            pruning_config.collection_sample_size, pruning_config.attn_mode )
        cripple_out = get_midlayer_activations( opt, pruning_config.cripple,
            pruning_config.collection_sample_size, pruning_config.attn_mode )

    # Otherwise, import activation data, and adjust the "pruning fraction"
    else:
        c["ff_frac"]   = min( 1.0, c["ff_frac"]*(iteration+1) )
        c["attn_frac"] = min( 1.0, c["attn_frac"]*(iteration+1) )
        assert not (focus_out is None or cripple_out is None or iteration is None), \
            "Must provide focus_out and cripple_out if not recalculate_activations"

    # Prune the model using the activation data
    data = score_and_prune(opt, focus_out, cripple_out, c)

    # Evaluate the model
    with torch.no_grad():
        eval_out = evaluate_all(opt, c.eval_sample_size, c.datasets,
                                dataset_tokens_to_skip=c.collection_sample_size)
        data.update(eval_out)

    return data

def score_and_prune( opt: Model,
            focus_activations_data: ActivationOverview,
            cripple_activations_data: ActivationOverview,
            pruning_config: PruningConfig,
            save=False,
        ):
    # Get the top fraction FF activations and prune
    ff_frac, ff_eps     = pruning_config.ff_frac,   pruning_config.ff_eps
    attn_frac, attn_eps = pruning_config.attn_frac, pruning_config.attn_eps
    do_ff   = ff_frac > 0
    do_attn = attn_frac > 0

    act_subset = pruning_config.scoring_normalization
    if do_ff > 0:
        ff_focus_data   = focus_activations_data.ff[act_subset]
        ff_cripple_data = cripple_activations_data.ff[act_subset]
        ff_scoring_fn = score_indices_by(pruning_config.ff_scoring)

        ff_scores = ff_scoring_fn(opt, ff_focus_data, ff_cripple_data, ff_eps)
        ff_criteria, ff_threshold = get_top_frac(ff_scores, ff_frac)
        opt.delete_ff_keys(ff_criteria)

    # Get the top fraction of Attention activations and prune
    if do_attn > 0:
        attn_focus_data   = focus_activations_data.attn[act_subset]
        attn_cripple_data = cripple_activations_data.attn[act_subset]
        # scoring for attention
        attn_scoring_fn = score_indices_by(pruning_config.attn_scoring)
        attn_scores = attn_scoring_fn(opt, attn_focus_data, attn_cripple_data, attn_eps)

        # offset by means if desired (probably bad?)
        means = None
        if pruning_config.do_attn_mean_offset:
            means = attn_focus_data["mean"]

        # get criteria for "neurons", or for "heads" if using full heads
        if pruning_config.attn_prune_heads:
            attn_head_scoring_fn = \
                choose_attn_heads_by(pruning_config.attn_prune_heads_mode)
            attn_criteria, attn_threshold = \
                attn_head_scoring_fn(opt, attn_scores, attn_frac)
            attn_criteria = opt.expand_remove_heads_to_remove_indices(attn_criteria)
        else:
            attn_criteria, attn_threshold = get_top_frac(attn_scores, attn_frac)
            _shape = (opt.cfg.n_layers, opt.cfg.n_heads, opt.cfg.d_head)
            attn_criteria = attn_criteria.reshape(_shape)

        # get criteria and prune if using only attention neurons
        if pruning_config.attn_mode == "pre-out":
            opt.delete_attn_pre_out( attn_criteria, means )
        elif pruning_config.attn_mode == "value":
            opt.delete_attn_values( attn_criteria, means )
        else:
            raise NotImplementedError("attn_mode must be 'pre-out' or 'value'")

    # Save the removals to file
    tensor_data = {
        "ff_scores": ff_scores if do_ff else None,
        # FIXME: doesn't return attn_std_mean
        "attn_scores": attn_scores if do_attn else None,
        "ff_criteria": ff_criteria if do_ff else None,
        "attn_criteria": attn_criteria if do_attn else None,
    }
    if save:
        save_timestamped_tensor_dict( opt, tensor_data, "activation_metrics" )

    # Initialize the output dictionary
    data = RunDataItem()

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

    # Save removals and scores to history
    _numpify = lambda x: x.cpu().numpy() if x is not None else None
    data.update({'raw': {
        k: _numpify(v) for k,v in tensor_data.items()
    }})

    return data

def prune_random( opt: Model,
        ff_frac: float,
        attn_frac: float,
        ff_pruned: Optional[np.ndarray] = None,
        attn_pruned: Optional[np.ndarray] = None,
        ):
    """Prune a random fraction of FF and Attention weights
    Args:
        opt (Model): model to prune and evaluate
        ff_frac (float): fraction of FF to prune
        attn_frac (float): fraction of Attention to prune

    """
    if ff_pruned is None:
        ff_pruned = np.zeros( (opt.cfg.n_layers, opt.cfg.d_mlp), dtype=np.bool_ )
    if attn_pruned is None:
        attn_pruned = np.zeros( (opt.cfg.n_layers, opt.cfg.d_model ), dtype=np.bool_ )

    n_ff_to_prune   = int( ff_frac   * opt.cfg.d_mlp )
    n_attn_to_prune = int( attn_frac * opt.cfg.d_model )

    # First prune the FF
    if not ff_frac == 0:
        for layer in range( opt.cfg.n_layers ):
            # choose new ff neurons to prune
            indices = np.where(ff_pruned[layer] == 0)[0]
            random_indices = np.random.choice(indices, n_ff_to_prune, replace=False)
            ff_pruned[layer][random_indices] = 1

        # Prune the model
        opt.delete_ff_keys( ff_pruned )

    if not attn_frac == 0:
        for layer in range( opt.cfg.n_layers ):
            # choose new attention heads to prune
            indices = np.where(attn_pruned[layer] == 0)[0]
            random_indices = np.random.choice(indices, n_attn_to_prune, replace=False)
            attn_pruned[layer][random_indices] = 1

        # Prune the model
        opt.delete_attn_pre_out( attn_pruned )

    data_out = {
        "ff_del": n_ff_to_prune*opt.cfg.n_layers,
        "attn_del": n_attn_to_prune*opt.cfg.n_layers
    }
    return ff_pruned, attn_pruned, data_out

def prune_random_and_evaluate( opt: Model,
        c: PruningConfig,
        ff_pruned: Optional[np.ndarray] = None,
        attn_pruned: Optional[np.ndarray] = None,
        ):
    """
    To use, run once with ff_pruned=None and attn_pruned=None, then run again
    with the parameters given as output passed back in.

    Args:
        opt (Model): The model to prune and evaluate
        c (PruningConfig): The pruning configuration
        ff_pruned (Optional[np.ndarray]): Bool list of FF neurons, default None.
        attn_pruned (Optional[np.ndarray], optional: Bool list of ATTN neurons, default None.

    Returns:
        ff_pruned (Optional[np.ndarray]):
        attn_pruned (Optional[np.ndarray]):
        data (RunDataItem):
    """


    # Prune the model randomly
    ff_pruned, attn_pruned, data_out = \
        prune_random( opt, c.ff_frac, c.attn_frac, ff_pruned, attn_pruned )

    # Initialize the output dictionary
    data = RunDataItem()

    # Evaluate the model
    data.update(
        evaluate_all( opt, c.eval_sample_size, c.datasets,
                      dataset_tokens_to_skip=c.collection_sample_size )
    )

    data.update({'deletions': data_out })

    data.update({'deletions_per_layer': {
        'ff': ff_pruned.sum(axis=-1).tolist() if (not ff_pruned is None) else 0,
        'attn': attn_pruned.sum(axis=-1).tolist() if (not attn_pruned is None) else 0,
    }})

    return ff_pruned, attn_pruned, data

######################################################################################
# Run Whole Pruning Procedure from Config
######################################################################################

def run_pruning(c: PruningConfig):
    # Initilaise Model and show details about model
    opt = Model(
        c.model_size,
        limit=c.token_limit,
        dtype=c.dtype,
        svd_attn=c.svd_attn,
        use_accelerator=c.use_accelerator,
        model_device=c.model_device,
        mask_fn=c.mask_fn,
        )

    # Prepare data logging
    history = RunDataHistory(c.datasets)
    wandb.init(
        project=c.wandb_project,
        entity=c.wandb_entity,
        name=c.wandb_run_name,
        )
    wandb.config.update(c.to_dict())

    # Evaluate model before removal of any neurons
    if c.run_pre_test:
        data = evaluate_all(opt, c.eval_sample_size,
            c.datasets, c.collection_sample_size)
        history.add(data)
        print(history.df.T)

    # If pruning randomly, no need to get activations
    if c.ff_scoring == "random" and c.attn_scoring == "random":
        ff_pruned, attn_pruned = None, None
        for i in range(c.n_steps):
            ff_pruned, attn_pruned, data = \
                prune_random_and_evaluate(opt, c, ff_pruned, attn_pruned)
            history.add(data)

    # Iteratively prune neurons and evaluate
    elif c.recalculate_activations:
        for _ in range(c.n_steps):
            data = prune_and_evaluate(opt, c)
            history.add(data)

    # Non-iteratively get activations, then iteratively prune and evaluate
    else:
        focus_out   = get_midlayer_activations(opt, c.focus,
                        c.collection_sample_size, c.attn_mode)
        cripple_out = get_midlayer_activations(opt, c.cripple,
                        c.collection_sample_size, c.attn_mode)
        for i in range(c.n_steps):
            data = prune_and_evaluate(opt, c, focus_out, cripple_out, i)
            history.add(data)

    # Format history to print
    print(history.history[-1])
    print(history.df.T)
    print(history.df.T.to_csv())

    return opt, history

######################################################################################
# "Forsaken"-style pruning
######################################################################################

def forsaken_pruning(c: PruningConfig,
        num_texts: int = 1,
        lr: float = 0.1,
        sigmoid_offset: float = 2.0,
        l1_norm_coeff: float = 1.0,
        ):
    # Initilaise Model and show details about model
    c.mask_fn = "sigmoid"
    c.misc = {
        "num_texts": num_texts,
        "lr": lr,
        "sigmoid_offset": sigmoid_offset,
        "l1_norm_coeff": l1_norm_coeff,
    }

    opt = Model(
        c.model_size,
        limit=c.token_limit,
        dtype=c.dtype,
        svd_attn=c.svd_attn,
        use_accelerator=c.use_accelerator,
        model_device=c.model_device,
        mask_fn=c.mask_fn,
        )

    # Prepare data logging
    history = RunDataHistory(c.datasets)
    wandb.init(
        project=c.wandb_project,
        entity=c.wandb_entity,
        name=c.wandb_run_name,
        )
    wandb.config.update(c.to_dict())

    # Evaluate model before removal of any neurons
    if c.run_pre_test:
        data = evaluate_all(opt, c.eval_sample_size,
            c.datasets, c.collection_sample_size)
        history.add(data)
        print(history.df.T)

    # Get activations
    focus_out   = get_midlayer_activations(opt, c.focus,
                    c.collection_sample_size, c.attn_mode)
    cripple_out = get_midlayer_activations(opt, c.cripple,
                    c.collection_sample_size, c.attn_mode)

    def normalize_scores(scores):
        normed_scores = []
        for score in scores:
            normed_scores.append(
                (score - score.mean()) / score.std()
            )
        return torch.stack(normed_scores)

    # Set masks for feed-forward layers
    ff_scores   = score_indices(c.ff_scoring,
        opt, focus_out.ff.orig,   cripple_out.ff.orig)
    ff_masks    = sigmoid_offset - normalize_scores(ff_scores)
    for layer_index in range(opt.cfg.n_layers):
        mask = opt.masks["mlp_pre_out"][layer_index]
        mask.set_mask(ff_masks[layer_index])

    # Set masks for attention heads
    attn_scores = score_indices(c.attn_scoring,
        opt, focus_out.attn.orig, cripple_out.attn.orig)
    attn_masks  = sigmoid_offset - normalize_scores(attn_scores)
    for layer_index in range(opt.cfg.n_layers):
        mask = opt.masks["attn_pre_out"][layer_index]
        mask.set_mask(attn_masks[layer_index])

    # Evaluate again now that we have adjusted the masks
    if True:
        data = evaluate_all(opt, c.eval_sample_size,
            c.datasets, c.collection_sample_size)
        history.add(data)

    # Get parameters for back propagation
    mask_params = [
        *[p for mask in opt.masks["mlp_pre_out"] for p in mask.parameters()],
        *[p for mask in opt.masks["attn_pre_out"] for p in mask.parameters()],
    ]
    mask_l1_norm = torch.stack([
        *[(1-mask.get_mask()).mean() for mask in opt.masks["mlp_pre_out"]],
        *[(1-mask.get_mask()).mean() for mask in opt.masks["attn_pre_out"]],
    ]).mean()

    # Generate Inputs
    n_iter = 4
    optim = torch.optim.LBFGS(mask_params, lr, max_iter=n_iter)
    kl_loss_fn = torch.nn.KLDivLoss()
    #ce_loss_fn = torch.nn.CrossEntropyLoss()

    # Load datasets
    def gen_texts(num_texts=1):
        _cripple_texts, _focus_texts = [], []

        cripple_dataset, cripple_label, _skip50 = prepare(c.cripple)
        i = 0
        for data in cripple_dataset:
            i += 1
            if i > num_texts:
                break
            _cripple_texts.append(data[cripple_label])

        focus_dataset, focus_label, _skip50     = prepare(c.focus)
        i = 0
        for data in focus_dataset:
            i += 1
            if i > num_texts:
                break
            _focus_texts.append(data[focus_label])

        return _cripple_texts, _focus_texts

    # Begin calculating loss for with LBGFS
    def get_new_ids(n_batches = None):
        batches = []
        cripple_texts, focus_texts = gen_texts()
        bad_ids, junk_ids, good_ids = [], [], []
        with torch.no_grad():
            for text in cripple_texts:
                bad_ids.append( opt.get_ids(text) )
                junk_ids.append(
                    torch.randint_like(bad_ids[-1], 5, opt.tokenizer.vocab_size)
                )
            for text in focus_texts:
                good_ids.append( opt.get_ids(text) )
        return bad_ids, junk_ids, good_ids

    for j in range(c.n_steps//n_iter):
        bad_ids, junk_ids, good_ids = get_new_ids()

        # Begin LBGFS
        def closure():
            loss = 0
            optim.zero_grad()

            # Generate loss
            loss += mask_l1_norm * l1_norm_coeff

            for i in range(num_texts):
                # Get junk loss L_kl(gamma,P)
                with torch.no_grad():
                    junk_logits = opt.get_all_logits(junk_ids[i])[..., :-1, :]
                bad_logits = opt.get_all_logits(bad_ids[i])[..., :-1, :]
                loss += kl_loss_fn(bad_logits, junk_logits).mean()

                # Get good loss L_kl(gamma,Q)
                with torch.no_grad():
                    opt.masking_enabled = False
                    orig_logits = opt.get_all_logits(good_ids[i])[..., :-1, :]
                    opt.masking_enabled = True
                new_logits = opt.get_all_logits(good_ids[i])[..., :-1, :]
                loss += kl_loss_fn(new_logits, orig_logits).mean()

            # Backpropagate
            loss.backward(retain_graph=True)
            return loss

        # loss step
        for i in range(n_iter):
            optim.step(closure)
            data = evaluate_all(opt, c.eval_sample_size,
                c.datasets, c.collection_sample_size)
            history.add(data)

    # Format history to print
    print(history.history[-1])
    print(history.df.T)
    print(history.df.T.to_csv())

    return opt, history

