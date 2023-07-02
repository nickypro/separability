"""
Some commands used for loading datasets used in my research.
That is, the 'codeparrot-clean' and 'the pile' datasets.
"""

import argparse
from datasets import load_dataset

from .model import Model

# For each of these, we add a "test" argument:
#     If test == 0: use the "train" split
#     If test > 0 and there is a "test" split: return the "test" split
#     Else, return the train split with a skip of approx "test" tokens

def load_code(test=0, _name=None):
    repo = "codeparrot/github-code-clean"
    name = 'all-all'

    # allow custom configurations
    if (_name is not None) and (1 <= len(_name)):
        name = _name

    _dataset = load_dataset(repo, name, streaming=True)
    if test:
        skip_n = int(test//100)
        print( "Warning: 'code' has no 'test' split.",
              f"Using 'train' split and skipping {skip_n} texts instead.")
        return _dataset['train'].skip(skip_n) # Conservative skip limit
    return _dataset['train']

def load_pile(test=0):
    # repo = "EleutherAI/pile_deduplicated" # has no testing split
    repo = "EleutherAI/pile"
    _dataset = load_dataset(repo, streaming=True)

    if test:
        return _dataset['test']
    return _dataset['train']

def load_pile_codeless(test=0):
    repo = "EleutherAI/pile" # deduplicated does not have meta tags
    _dataset = load_dataset(repo, "all", streaming=True)
    def filter_out_code(example):
        return example['meta']['pile_set_name'] != 'Github'
    _dataset = _dataset.filter(filter_out_code)
    if test:
        return _dataset['test']
    return _dataset['train']

def load_civil(test=0):
    _dataset = load_dataset("civil_comments", streaming=True)
    # Filter the dataset for toxicity > 0.8
    def filter_toxicity(example):
        return example["toxicity"] <= 0.2
    low_toxicity_dataset = _dataset.filter(filter_toxicity)
    if test:
        return low_toxicity_dataset['test']
    return low_toxicity_dataset['train']

def load_toxic(test=0):
    _dataset = load_dataset("civil_comments", streaming=True)
    # Filter the dataset for toxicity > 0.8
    def filter_toxicity(example):
        return example["toxicity"] >= 0.8
    high_toxicity_dataset = _dataset.filter(filter_toxicity)
    if test:
        return high_toxicity_dataset['test']
    return high_toxicity_dataset['train']

def load_wiki(test=0):
    _dataset = load_dataset("wikitext", "wikitext-103-v1", streaming=True)
    if test:
        return _dataset['test']
    return _dataset['train']

# Hard load the most common tokens from the datasets from previous runs.
# pylint: disable=line-too-long
most_common_code_tokens = [' ', '\n', '.', '_', ',', '#', '(', ' =', ' import', 'from', ' the', ':', ')', '\n\n', 'import', " '", '/', '-', '):', '\t', "',", ' "', ' self', '=', ' of', "'", '__', ' (', 'self', ' in', ' License', '</s>', ' is', '0', ' for', ' to', 's', '1', '2', ' a', ' as', '\r', ' -', ' and', ' def', ' #', 'x', '()', "('", '\\']
most_common_pile_tokens = ['\n', '.', ',', ' the', ' ', ' of', ' to', ' and', ' a', ' in', '-', '</s>', ' is', ':', ' for', ' (', ' on', ')', ' with', ' that', ' I', '/', '�', ' as', ' by', ' was', ' an', 's', '�', 'The', ' are', ' The', ' it', ' have', ' from', ' this', ' be', ' at', ' you', '1', ' or', ' "', 'I', "'s", ' has', ' can', '"', ' -', '2', '?']

def prepare( dataset_name, test:int = 0 ):
    if dataset_name == 'pile_codeless':
        return load_pile_codeless(test), 'text', most_common_pile_tokens

    if dataset_name == 'pile':
        return load_pile(test), 'text', most_common_pile_tokens

    if dataset_name == 'python':
        return load_code(test, 'Python-all'), 'code', most_common_code_tokens

    if dataset_name[:4] == 'code':
        name = dataset_name[5:]
        return load_code(test, name), 'code', most_common_code_tokens

    if dataset_name[:5] == 'civil':
        return load_civil(test), 'text', most_common_pile_tokens

    if dataset_name[:5] == 'toxic':
        return load_toxic(test), 'text', most_common_pile_tokens

    if dataset_name[:4] == 'wiki':
        return load_wiki(test), 'text', most_common_pile_tokens

    raise ValueError( f"Unknown dataset: {dataset_name}" )
