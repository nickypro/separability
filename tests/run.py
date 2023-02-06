from model import Model
from activations import evaluate_all, prune_and_evaluate

opt = Model("2.7b", limit=1000, use_accelerator=True)

evaluate_all(opt, 1e4)

prune_and_evaluate(opt, 0.05, 0.05, 0.001, 1e4, 1e4)
