import argparse
import torch
import numpy as np
import pandas as pd
import einops
import matplotlib.pyplot as plt
import wandb

from separability import Model
from separability.data_classes import RunDataHistory, PruningConfig
from separability.activations import prune_and_evaluate, evaluate_all

# Wandb config
project   = "pile-code-attn"

# Configure initial model and tests
c = PruningConfig(
    model_repo   = "facebook/opt-1.3b",
    token_limit  = 1000,
    run_pre_test = True,
    # Removals parameters
    ff_frac   = 0.02,
    ff_eps    = 0.001,
    attn_frac = 0.00,
    attn_eps  = 1e-4,
    focus     = "pile_codeless",
    cripple   = "code",
    additional_datasets=tuple(),
)
pre_removals = []


# Build a CLI parser
parser = argparse.ArgumentParser()

parser.add_argument('model_repo', type=str)
parser.add_argument('--project', type=str, default=project)
parser.add_argument('-n', "--name", type=str, default=None, help="wandb run name")
parser.add_argument('-r', '--reverse', action='store_true', help="cripple <--> focus")
parser.add_argument('--n_steps', type=int, default=None)
parser.add_argument('--model_device', type=str, default=None)

args_exclude = ["model_repo", "n_steps", "model_device"]
for key, val in c.arg_items(args_exclude):
    parser.add_argument(f'--{key}', type=type(val), default=val)

# Parse the argument
args = parser.parse_args()

c.model_repo = args.model_repo
c.model_device = args.model_device
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
opt = Model(c.model_size, limit=c.token_limit, dtype=c._dtype, svd_attn=c.svd_attn,
            use_accelerator=c.use_accelerator, model_device=c.model_device)

# Pre-pruning of model
opt.delete_ff_keys_from_files(pre_removals)

# Evaluate model before removal of any neurons
if c.run_pre_test:
    history.add(evaluate_all(opt, 1e5, c.datasets))
    print(history.df.T)

for i in range(n_steps):
    data = prune_and_evaluate(opt, c)
    history.add(data)

print(history.history[-1])
print(history.df.T)
print(history.df.T.to_csv())
