import argparse
import torch
import numpy as np
import pandas as pd
import einops
import matplotlib.pyplot as plt
import wandb

from separability import Model
from separability.data_classes import RunDataHistory, PruningConfig
from separability.activations import prune_and_evaluate, \
    get_midlayer_activations, score_and_prune
from separability.eval import evaluate_all

# Wandb config
project   = "method-compare"

# Configure initial model and tests
c = PruningConfig(
    model_repo   = "facebook/opt-1.3b",
    token_limit  = 1000,
    run_pre_test = True,
    # Removals parameters
    ff_frac   = 0.00,
    ff_eps    = 0.001,
    attn_frac = 0.05,
    attn_eps  = 1e-4,
    focus     = "pile_codeless",
    cripple   = "code",
    additional_datasets=tuple(),
    collection_sample_size=1e5,
    recalculate_activations=False,
)
pre_removals = []


# Build a CLI parser
parser = argparse.ArgumentParser()

parser.add_argument('model_repo', type=str)
parser.add_argument('--project', type=str, default=project)
parser.add_argument('-n', "--name", type=str, default=None, help="wandb run name")
parser.add_argument('-r', '--reverse', action='store_true', help="cripple <--> focus")
parser.add_argument('--n_steps', type=int, default=None)

args_exclude = ["model_repo", "n_steps"]
for key, val in c.arg_items(args_exclude):
    parser.add_argument(f'--{key}', type=type(val), default=val)

# Parse the argument
args = parser.parse_args()

c.model_repo = args.model_repo
for key in c.arg_keys(args_exclude):
    c[key] = getattr(args, key)
if args.reverse:
    c.focus, c.cripple = c.cripple, c.focus
# First do some pruning of the feed forward layers
n_steps = args.n_steps
if n_steps is None:
    n_steps = int( 1 / max(c.ff_frac, c.attn_frac) )


# Prepare data logging
wandb.init(project=args.project, entity="seperability", name=args.name)
wandb.config.update(c.to_dict())

# Load model and show details about model
history = RunDataHistory(c.datasets)
opt = Model(c.model_size, limit=c.token_limit, dtype=c._dtype, svd_attn=c.svd_attn)

# Pre-pruning of model
opt.delete_ff_keys_from_files(pre_removals)

# Evaluate model before removal of any neurons
if c.run_pre_test:
    history.add(evaluate_all(opt, 1e5, c.datasets))
    print(history.df.T)

# Get midlayer activations of FF and ATTN
focus_out   = get_midlayer_activations(opt, c.focus, c.collection_sample_size, c.attn_mode)
cripple_out = get_midlayer_activations(opt, c.cripple, c.collection_sample_size, c.attn_mode)
texts_to_skip = max(focus_out["texts_viewed"], cripple_out["texts_viewed"] )

# Prune more of the model each time
ff_frac_original, attn_frac_original = c.ff_frac, c.attn_frac
for i in range(1, n_steps+1):
    c.ff_frac   = i * ff_frac_original
    c.attn_frac = i * attn_frac_original
    # Prune the model using the activation data
    data = score_and_prune(opt, focus_out, cripple_out, c)

    # Evaluate the model
    data.update(
        evaluate_all(opt, c.eval_sample_size, c.datasets, texts_to_skip=texts_to_skip)
    )
    history.add(data)

print(history.history[-1])
print(history.df.T)
print(history.df.T.to_csv())
