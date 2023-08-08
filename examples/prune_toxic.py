from separability.data_classes import PruningConfig
from separability.parser import cli_parser
from separability.prune import run_pruning
import torch

# Configure initial model and tests
c = PruningConfig(
    wandb_project = "civil-toxic",
    model_repo   = "facebook/galactica-6.7b",
    token_limit  = 1000,
    run_pre_test = True,
    # Removals parameters
    ff_frac   = 0.002,
    ff_eps    = 0.001,
    attn_frac = 0.002,
    attn_eps  = 1e-4,
    focus     = "civil",
    cripple   = "toxic",
    additional_datasets=["wiki", "toxicity", "mmlu:all"],
)

# Parse CLI for arguments
c, args = cli_parser(c)

# Run the iterated pruning
with torch.no_grad():
    model, history = run_pruning(c)
