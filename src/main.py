import torch
import numpy as np
import pandas as pd
import einops
import matplotlib.pyplot as plt

from model import Model
from texts import prepare
from activations import calculate_attn_crossover, \
    delete_ff_and_evaluate, evaluate_all


opt = Model("125m", limit=1000)
print( "layers:", opt.n_layers, "embedding dimension:", opt.d_model )

evaluate_all( opt, 1e4 )