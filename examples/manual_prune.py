import torch
from separability import Model
from separability.data_classes import RunDataItem
from separability.activations import get_midlayer_activations, evaluate_all, \
    get_top_frac, get_attn_crossover

def manual_prune_and_evaluate(model_name):
    sample_size, eval_size = 1e4, 1e4
    focus, cripple = "pile", "code"

    do_ff = True
    ff_prune_frac = 0.01
    ff_eps = 1e-3

    do_attn = True
    attn_prune_frac = 0.01

    opt = Model(model_name, limit=1000, use_accelerator=False)
    example_text = " example test input string for large language model"

    act = opt.get_text_activations(example_text)
    print(act)

    # Get midlayer activations of FF and ATTN
    focus_out   = get_midlayer_activations( opt, focus, sample_size )
    cripple_out = get_midlayer_activations( opt, cripple, sample_size )

    # Get the top fraction FF activations and prune
    if do_ff > 0:
        cripple_ff_count = cripple_out["ff"]["pos_count"]
        focus_ff_count = focus_out["ff"]["pos_count"]
        ff_rel_freq = ( cripple_ff_count / ( focus_ff_count + ff_eps ) ).cpu()
        ff_criteria, ff_threshold = get_top_frac( ff_rel_freq, ff_prune_frac )
        opt.delete_ff_keys( ff_criteria )

    # Get the top fraction of Attention activations and prune
    if do_attn > 0:
        attn_data = get_attn_crossover( opt, focus_out["attn"], cripple_out["attn"] )
        attn_criteria, attn_threshold = \
            get_top_frac( attn_data["crossover_multiple"], attn_prune_frac )
        opt.delete_attn_pre_out_heads( attn_criteria, attn_data["pile_means"] )

    # Initialize the output dictionary
    data = RunDataItem()

    # Evaluate the model
    texts_to_skip = max( focus_out["texts_viewed"], cripple_out["texts_viewed"] )
    data.update( evaluate_all( opt, eval_size, texts_to_skip=texts_to_skip ) )

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

    act = opt.get_text_activations(example_text)
    print(act)

    print(opt.model.decoder.layers[0])

    return data

data = manual_prune_and_evaluate("facebook/galactica-125m")
print(data.summary())
