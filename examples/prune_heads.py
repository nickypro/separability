import torch
from separability import Model
from separability.data_classes import RunDataItem, RunDataHistory
from separability.activations import evaluate_all
import wandb

global_state = {'init': False}

# init wandb and config stuff
wandb.init(entity="seperability", project="gal-125m-heads")
c = wandb.config
c.update({
    "model_size"  : "facebook/opt-125m",
    "token_limit" : 1000,
    "run_pre_test": True,
    "ff_frac"  : 0,
    "ff_eps"   : 0,
    "attn_frac": 0,
    "attn_prune_type": "pre_out",
    "svd_attn": False,
    "do_attn_mean_offset": False,
    "attn_prune_heads": True,
})
eval_size = 1e5
datasets = ["pile", "code", "python"]
history = RunDataHistory(datasets)

def prune_head_and_evaluate(head_index):

    # load model
    opt = Model(c.model_size, limit=c.token_limit, dtype=torch.float16,
        svd_attn=c.svd_attn)

    # get which head to prune:
    if head_index >= (opt.n_heads * opt.n_layers):
        raise ValueError(f"Head index {head_index} is out of bounds")
    attn_criteria = torch.zeros((opt.n_layers * opt.n_heads))
    attn_criteria[head_index] = 1
    attn_criteria = attn_criteria.reshape((opt.n_layers, opt.n_heads))

    # Init data history
    if global_state['init'] == False:
        init_data = RunDataItem()
        init_data.update(
            evaluate_all(opt, eval_size, datasets=datasets)
        )
        history.add(init_data)
        global_state['init'] = init_data
        print()

    #else:
    #    history.add(global_state['init'])

    opt.delete_attn_pre_out_heads(attn_criteria)

    # Evaluate the model
    data = RunDataItem()

    data.update( evaluate_all( opt, eval_size, datasets=datasets ) )

    data.update({'deletions': {
        "attn_head_index": head_index,
        "attn_head_layer": head_index // opt.n_heads,
        "attn_head_subindex": head_index % opt.n_heads,
        "attn_del": 1,
    }})

    history.add( data )

    return data

for i in range(12*12):
    _data = prune_head_and_evaluate(i)
    print(_data.summary())
