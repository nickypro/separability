"""
Some commands used for loading datasets used in my research.
That is, the 'codeparrot-clean' and 'the pile' datasets.
"""

import argparse
from datasets import load_dataset

from .model import Model

def load_code(_name=None):
    repo = "codeparrot/github-code-clean"
    name = 'all-all'

    # allow custom configurations
    if (_name is not None) and (1 <= len(_name)):
        name = _name

    _dataset = load_dataset(repo, name, streaming=True)
    return _dataset['train']

def load_pile():
    repo = "EleutherAI/the_pile_deduplicated"
    _dataset = load_dataset(repo, streaming=True )
    return _dataset['train']

def load_pile_codeless():
    repo = "EleutherAI/pile" # deduplicated does not have meta tags
    _dataset = load_dataset(repo, "all", streaming=True)
    def filter_out_code(example):
        return example['meta']['pile_set_name'] != 'Github'
    _dataset = _dataset.filter(filter_out_code)
    return _dataset['train']

# Hard load the most common tokens from the datasets from previous runs.
# pylint: disable=line-too-long
most_common_code_tokens = [' ', '\n', '.', '_', ',', '#', '(', ' =', ' import', 'from', ' the', ':', ')', '\n\n', 'import', " '", '/', '-', '):', '\t', "',", ' "', ' self', '=', ' of', "'", '__', ' (', 'self', ' in', ' License', '</s>', ' is', '0', ' for', ' to', 's', '1', '2', ' a', ' as', '\r', ' -', ' and', ' def', ' #', 'x', '()', "('", '\\']
most_common_pile_tokens = ['\n', '.', ',', ' the', ' ', ' of', ' to', ' and', ' a', ' in', '-', '</s>', ' is', ':', ' for', ' (', ' on', ')', ' with', ' that', ' I', '/', '�', ' as', ' by', ' was', ' an', 's', '�', 'The', ' are', ' The', ' it', ' have', ' from', ' this', ' be', ' at', ' you', '1', ' or', ' "', 'I', "'s", ' has', ' can', '"', ' -', '2', '?']

def prepare_code(name = None):
    return load_code(name), 'code', most_common_code_tokens

def prepare_pile():
    return load_pile(), 'text', most_common_pile_tokens

def prepare_pile_codeless():
    return load_pile_codeless(), 'text', most_common_pile_tokens

def prepare_civil():
    _dataset = load_dataset("civil_comments", streaming=True)
    # Filter the dataset for toxicity > 0.8
    def filter_toxicity(example):
        return example["toxicity"] <= 0.2
    low_toxicity_dataset = _dataset.filter(filter_toxicity)
    return low_toxicity_dataset['train'], 'text', most_common_pile_tokens

def prepare_toxic():
    _dataset = load_dataset("civil_comments", streaming=True)
    # Filter the dataset for toxicity > 0.8
    def filter_toxicity(example):
        return example["toxicity"] >= 0.8
    high_toxicity_dataset = _dataset.filter(filter_toxicity)
    return high_toxicity_dataset['train'], 'text', most_common_pile_tokens

def prepare_wiki():
    _dataset = load_dataset("wikitext", "wikitext-103-v1", streaming=True)
    return _dataset["train"], "text", most_common_pile_tokens

def prepare( dataset_name ):
    if dataset_name == 'pile_codeless':
        return prepare_pile_codeless()

    if dataset_name == 'pile':
        return prepare_pile()

    if dataset_name == 'python':
        return prepare_code('Python-all')

    if dataset_name[:4] == 'code':
        name = dataset_name[5:]
        return prepare_code(name)

    if dataset_name[:5] == 'civil':
        return prepare_civil()

    if dataset_name[:5] == 'toxic':
        return prepare_toxic()

    if dataset_name[:4] == 'wiki':
        return prepare_wiki()

    raise ValueError( f"Unknown dataset: {dataset_name}" )
