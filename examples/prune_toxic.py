import argparse
import torch
import numpy as np
import pandas as pd
import einops
import matplotlib.pyplot as plt
import wandb

from separability import Model
from separability.data_classes import RunDataHistory
from separability.activations import prune_and_evaluate, evaluate, evaluate_all
from separability.eval import evaluate_toxicity

# Configure initial model and tests
model_size, token_limit  = "facebook/opt-1.3b", 1000
run_pre_test             = True
pre_removals = []

# Removals parameters
ff_frac,   ff_eps   = 0.01, 0.001
attn_frac           = 0.005
focus, cripple      = "civil", "toxic"
project             = "civil-toxic"
datasets            = [*list(sorted([focus, cripple])), "pile"]

parser = argparse.ArgumentParser()

# Add an argument
parser.add_argument('repo', type=str)
parser.add_argument('-r', '--reverse', action='store_true')
parser.add_argument('--svd', action='store_true')
parser.add_argument('-s', '--attn_scoring', type=str, default="abs")
parser.add_argument('--prune_heads', type=str, default=False) # mean, median
parser.add_argument('--project', type=str, default=project)
parser.add_argument('--svd_combine_biases', action='store_true')

# Parse the argument
args = parser.parse_args()
model_size = args.repo
svd_attn = args.svd
if args.reverse:
    focus, cripple = cripple, focus

# Prepare data logging
wandb.init(project=args.project, entity="seperability")
c = wandb.config
c.update({
    "model_size"  : model_size,
    "token_limit" : token_limit,
    "run_pre_test": run_pre_test,
    "ff_frac"  : ff_frac,
    "ff_eps"   : ff_eps,
    "attn_frac": attn_frac,
    "cripple": cripple,
    "focus"  : focus,
    "attn_prune_type": "pre_out",
    "svd_attn": svd_attn,
    "do_attn_mean_offset": False,
    "attn_scoring": args.attn_scoring,
    "attn_prune_heads": args.prune_heads,
    "svd_combine_biases": args.svd_combine_biases,
    "delete_residual_biases": False,
})

# Load model and show details about model
history = RunDataHistory(datasets)
opt = Model(c.model_size, limit=c.token_limit, dtype=torch.float16,
    svd_attn=c.svd_attn)

# Pre-pruning of model
opt.delete_ff_keys_from_files(pre_removals)

opt.delete_residual_biases()

# Evaluate model before removal of any neurons
if c.run_pre_test:
    data = evaluate_all(opt, 1e5, datasets)

    # Get toxicity evaluations
    ratio_toxic, toxicity_mean = evaluate_toxicity(opt, 1000)
    data.update({"misc": {
        'ratio_toxic': ratio_toxic,
        'toxicity_mean': toxicity_mean,
    }})
    print(data["misc"])

    history.add(data)
    print(history.df.T)

# First do some pruning of the feed forward layers
for i in range(20):
    data = prune_and_evaluate(opt, c.ff_frac, c.attn_frac, c.ff_eps,
        cripple=c.cripple, focus=c.focus, do_attn_mean_offset=c.do_attn_mean_offset,
        attn_scoring=c.attn_scoring, attn_prune_heads=c.attn_prune_heads)

    # Get toxicity evaluations
    ratio_toxic, toxicity_mean = evaluate_toxicity(opt, 1000)
    data.update({"misc": {
        'ratio_toxic': ratio_toxic,
        'toxicity_mean': toxicity_mean,
    }})
    print(data.misc)

    # Get pile evaluation
    pile_data = evaluate(opt, "pile")
    data.update({
        "accuracy": {
            "pile": pile_data['percent']
        },
        "loss_data": {
            "pile": pile_data['loss_data']
        }
    })

    history.add(data)

print(history.history[-1])
print(history.df.T)
print(history.df.T.to_csv())
