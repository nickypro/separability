import argparse
import torch
import numpy as np
import pandas as pd
import einops
import matplotlib.pyplot as plt
import wandb

from seperability import Model
from seperability.data_classes import RunDataHistory
from seperability.activations import prune_and_evaluate, evaluate_all

# Configure initial model and tests
model_size, token_limit  = "facebook/opt-13b", 1000
run_pre_test             = True
pre_removals = []

# Removals parameters
ff_frac,   ff_eps   = 0.04, 0.001
attn_frac           = 0.01
focus, cripple      = "pile", "code"
project             = "seperability-pile-code"
datasets            = list(sorted([focus, cripple]))

parser = argparse.ArgumentParser()

# Add an argument
parser.add_argument('repo', type=str)

# Parse the argument
args = parser.parse_args()
model_size = args.repo

# Prepare data logging
wandb.init(project=project, entity="seperability")
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
})

# Load model and show details about model
history = RunDataHistory(datasets)
opt = Model( c.model_size, limit=c.token_limit, dtype=torch.float16 )

# Pre-pruning of model
opt.delete_ff_keys_from_files(pre_removals)

# Evaluate model before removal of any neurons
if c.run_pre_test:
    history.add( evaluate_all( opt, 1e5, datasets ) )
    print( history.df.T )

# First do some pruning of the feed forward layers
for i in range(30):
    data = prune_and_evaluate( opt, c.ff_frac, c.attn_frac, c.ff_eps, cripple=c.cripple, focus=c.focus )
    history.add( data )

print(history.history[-1])
print(history.df.T)
print(history.df.T.to_csv())
