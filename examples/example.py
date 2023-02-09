import torch
from seperability import Model
from seperability.data_classes import RunDataItem
from seperability.activations import evaluate_all, prune_and_evaluate

opt = Model("facebook/opt-125m", limit=1000, use_accelerator=True, dtype=torch.float32)


data = evaluate_all(opt, 1e4)
print( RunDataItem().update(data).summary() )

data = prune_and_evaluate(opt, 0.2, 0.1, 0.001, 1e4, 1e4)
print( data.summary())
