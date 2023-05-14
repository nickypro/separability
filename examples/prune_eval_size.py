import torch
from separability import Model
from separability.data_classes import RunDataItem, RunDataHistory
from separability.activations import prune_and_evaluate
from separability.eval import evaluate_all
import wandb

global_state = {'init': False}

# init wandb and config stuff
wandb.init(entity="seperability", project="hyperparameters")
c = wandb.config
c.update({
    "model_size"  : "facebook/opt-1.3b",
    "token_limit" : 1000,
    "run_pre_test": True,
    "ff_frac"  : 0.0,
    "ff_eps"   : 1e-3,
    "attn_frac": 0.05,
    "attn_prune_type": "pre_out",
    "svd_attn": False,
    "do_attn_mean_offset": False,
    "attn_prune_heads": False,
    "attn_scoring": "abs",
    "cripple": "code",
    "focus": "pile",
})
datasets = ["pile", "code", "python"]
history = RunDataHistory(datasets)

for sample_size in [1e3, 2e3, 4e3, 1e4, 2e4, 4e4, 1e5, 2e5, 4e5, 1e6, 2e6, 4e6, 1e7]:
    eval_size = 1e6
    opt = Model(c.model_size, limit=c.token_limit, dtype=torch.float16,
        svd_attn=c.svd_attn)

    data = prune_and_evaluate(opt, c.ff_frac, c.attn_frac, c.ff_eps,
        cripple=c.cripple, focus=c.focus,
        sample_size=sample_size, eval_size=eval_size,
        do_attn_mean_offset=c.do_attn_mean_offset,
        attn_scoring=c.attn_scoring, attn_prune_heads=c.attn_prune_heads)

    data.update({'misc': {'sample_size': sample_size}})

    history.add(data)
    print(data.summary())
