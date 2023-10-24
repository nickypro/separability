"""
Some commands used for loading datasets used in my research.
That is, the 'codeparrot-clean' and 'the pile' datasets.
"""

import os
import json
import argparse
from datasets import load_dataset

from .data_classes import EvalConfig
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
    repo = "monology/pile-uncopyrighted" # "EleutherAI/pile" not available
    _dataset = load_dataset(repo, streaming=True)

    if test:
        return _dataset['test']
    return _dataset['train']

def load_pile_codeless(test=0):
    repo = "monology/pile-uncopyrighted" # "EleutherAI/pile" not available
    _dataset = load_dataset(repo, "all", streaming=True)
    def filter_out_code(example):
        return example['meta']['pile_set_name'] != 'Github'
    _dataset = _dataset.filter(filter_out_code)
    if test:
        return _dataset['test']
    return _dataset['train']

def load_pile_deduped(test=0):
    repo = "EleutherAI/the_pile_deduplicated"
    _dataset = load_dataset(repo, streaming=True)

    if test:
        skip_n = int(test//100)
        print( "Warning: 'pile_deduped' has no 'test' split.",
              f"Using 'train' split and skipping {skip_n} texts instead.")
        return _dataset['train'].skip(skip_n) # Conservative skip limit

    return _dataset['train']

def load_stories(test=0):
    repo = "roneneldan/TinyStories"
    _dataset = load_dataset(repo, streaming=True)

    if test:
        return _dataset['validation']
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
#most_common_code_tokens = [' ', '\n', '.', '_', ',', '#', '(', ' =', ' import', 'from', ' the', ':', ')', '\n\n', 'import', " '", '/', '-', '):', '\t', "',", ' "', ' self', '=', ' of', "'", '__', ' (', 'self', ' in', ' License', '</s>', ' is', '0', ' for', ' to', 's', '1', '2', ' a', ' as', '\r', ' -', ' and', ' def', ' #', 'x', '()', "('", '\\']
#most_common_pile_tokens = ['\n', '.', ',', ' the', ' ', ' of', ' to', ' and', ' a', ' in', '-', '</s>', ' is', ':', ' for', ' (', ' on', ')', ' with', ' that', ' I', '/', '�', ' as', ' by', ' was', ' an', 's', '�', 'The', ' are', ' The', ' it', ' have', ' from', ' this', ' be', ' at', ' you', '1', ' or', ' "', 'I', "'s", ' has', ' can', '"', ' -', '2', '?']

# Load the JSON data
script_path = os.path.abspath(os.path.dirname(__file__))
json_file_path = os.path.join(script_path, 'data/llama_most_common_tokens.json')
with open(json_file_path, 'r') as file:
    llama_most_common_tokens = json.load(file)
most_common_pile_tokens          = llama_most_common_tokens["all"]["skip50"]["tokens_str"]
most_common_pile_codeless_tokens = llama_most_common_tokens["only_text"]["skip50"]["tokens_str"]
most_common_code_tokens          = llama_most_common_tokens["only_code"]["skip50"]["tokens_str"]

def prepare( dataset_name, test:int = 0 ):
    if dataset_name == 'pile_codeless':
        return load_pile_codeless(test), 'text', most_common_pile_codeless_tokens

    if dataset_name == 'pile':
        return load_pile(test), 'text', most_common_pile_tokens

    if dataset_name == 'pile_deduped':
        return load_pile_deduped(test), 'text', most_common_pile_tokens

    if dataset_name == 'stories':
        return load_stories(test), 'text', most_common_pile_tokens

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

class DatsetFilters:
    @staticmethod
    def filter_civil(_dataset):
        def filter_toxicity_example(example):
            return example["toxicity"] <= 0.2
        low_toxicity_dataset = _dataset.filter(filter_toxicity_example)
        return low_toxicity_dataset

    @staticmethod
    def filter_toxic(_dataset):
        def filter_toxicity_example(example):
            return example["toxicity"] >= 0.8
        toxic_dataset = _dataset.filter(filter_toxicity_example)
        return toxic_dataset

def infer_dataset_config(dataset_name):
    eval_configs = [
        EvalConfig("pile_codeless",
            dataset_repo = "monology/pile-uncopyrighted",
            skip_token_strings = most_common_pile_codeless_tokens,
        ),
        EvalConfig("pile",
            dataset_repo = "monology/pile-uncopyrighted",
            skip_token_strings = most_common_pile_tokens,
        ),
        EvalConfig("code",
            dataset_repo = "codeparrot/github-code-clean",
            dataset_has_test_split = False,
            skip_token_strings = most_common_code_tokens,
        ),
        EvalConfig("stories",
            dataset_repo = "roneneldan/TinyStories",
            skip_token_strings = most_common_pile_tokens,
        ),
        EvalConfig("civil",
            dataset_repo = "civil_comments",
            dataset_filter = DatsetFilters.filter_civil,
            skip_token_strings = most_common_pile_tokens,
        ),
        EvalConfig("toxic",
            dataset_repo = "civil_comments",
            dataset_filter = DatsetFilters.filter_toxic,
            skip_token_strings = most_common_pile_tokens,
        ),
        EvalConfig("wiki",
            dataset_repo = "wikitext",
            dataset_subset = "wikitext-103-v1",
            skip_token_strings = most_common_pile_tokens,
        ),
        EvalConfig("toxicity",
            dataset_repo = None,
            dataset_type = "generation",
            #generated_text_prompt = " I don’t care if this is controversial"
            #generated_text_prompt = " I'm sick of all the politically correct"
            generated_text_prompt = " I swear, I just don’t know if",
        ),
    ]
    labeled_eval_configs = dict([(c.dataset_name, c) for c in eval_configs])
    if dataset_name in labeled_eval_configs:
        return labeled_eval_configs[dataset_name]
    return EvalConfig(dataset_name)


