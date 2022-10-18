from datasets import load_dataset
from evaluate import evaluator
import torch
import numpy as np
from model import Model, most_common_code_tokens

Text2TextEvaluator = evaluator("text2text-generation")

def load_code():
    dataset = load_dataset("codeparrot/codeparrot-clean-valid")
    return dataset['train']

#skip_eval = set([' ', '\n', '.', '_', ',', '#', '\n\n', '\t' ])
#skip_eval = set([' ', '\n', '.', '_', ',', '#', ' =', '(', ' import', 'from', ' the', '\t', ':', ')', " '", '\n\n', '-', '/', 'import', '):', "'", "',", ' self', 'self', ' "', ' of', '__', '=', ' (', '</s>', ' in', ' is', 's', ' License', '\r', ' for', '0', ' def', "('", ';', '1', '()', ' -', ' #', 'XXXX', '2', ' and', '",', ' to', ' as'])

if __name__ == "__main__":
    dataset = load_code()

    opt = Model('125m')

    skip_eval = most_common_code_tokens
    opt.evaluate_dataset( dataset, token_limit=100, k=5, skip_eval=skip_eval )