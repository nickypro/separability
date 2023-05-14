import torch
from separability import Model
from separability.data_classes import RunDataItem
from separability.activations import prune_and_evaluate
from separability.eval import evaluate_all

opt = Model("facebook/opt-125m", limit=1000, use_accelerator=True, dtype=torch.float32)

data_dict : dict = evaluate_all(opt, 1e4)
print( RunDataItem(data_dict) )

data : RunDataItem = prune_and_evaluate(opt, 0.2, 0.05, 0.001, 1e4, 1e4)
print( data )
