import torch
from seperability import Model
from seperability.activations import evaluate_all, prune_and_evaluate

opt = Model("facebook/galactica-125m", limit=1000, use_accelerator=True, dtype=torch.float16, output_device="cuda:1")

evaluate_all(opt, 1e4)

prune_and_evaluate(opt, 0.05, 0.05, 0.001, 1e4, 1e4)