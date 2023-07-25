import torch
import numpy as np
import pandas as pd
import einops
import matplotlib.pyplot as plt
import wandb

from separability import Model
from separability.data_classes import RunDataHistory, PruningConfig
from separability.activations import prune_and_evaluate
from separability.eval import evaluate_all
from separability.parser import cli_parser

# Configure initial model and tests
c = PruningConfig(
    wandb_project = "testing",
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
c, args = cli_parser(c)

# Prepare data logging
wandb.init(project=c.wandb_project, entity="seperability", name=c.wandb_run_name)
wandb.config.update(c.to_dict())

# Load model and show details about model
history = RunDataHistory(c.datasets)
opt = Model(c.model_size, limit=c.token_limit, dtype=c.dtype, svd_attn=c.svd_attn,
            use_accelerator=c.use_accelerator, model_device=c.model_device)

# Pre-pruning of model
opt.delete_ff_keys_from_files(pre_removals)

# Evaluate model before removal of any neurons
if c.run_pre_test:
    history.add(
        evaluate_all(opt, c.eval_sample_size, c.datasets, c.collection_sample_size)
    )
    print(history.df.T)

for i in range(c.n_steps):
    data = prune_and_evaluate(opt, c)
    history.add(data)

print(history.history[-1])
print(history.df.T)
print(history.df.T.to_csv())
