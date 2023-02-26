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

# Hard load the most common tokens from the datasets from previous runs.
# pylint: disable=line-too-long
most_common_code_tokens = [' ', '\n', '.', '_', ',', '#', '(', ' =', ' import', 'from', ' the', ':', ')', '\n\n', 'import', " '", '/', '-', '):', '\t', "',", ' "', ' self', '=', ' of', "'", '__', ' (', 'self', ' in', ' License', '</s>', ' is', '0', ' for', ' to', 's', '1', '2', ' a', ' as', '\r', ' -', ' and', ' def', ' #', 'x', '()', "('", '\\']
most_common_pile_tokens = ['\n', '.', ',', ' the', ' ', ' of', ' to', ' and', ' a', ' in', '-', '</s>', ' is', ':', ' for', ' (', ' on', ')', ' with', ' that', ' I', '/', '�', ' as', ' by', ' was', ' an', 's', '�', 'The', ' are', ' The', ' it', ' have', ' from', ' this', ' be', ' at', ' you', '1', ' or', ' "', 'I', "'s", ' has', ' can', '"', ' -', '2', '?']

def prepare_code(name = None):
    return load_code(name), 'code', most_common_code_tokens

def prepare_pile():
    return load_pile(), 'text', most_common_pile_tokens

def prepare( dataset_name ):
    if dataset_name == 'pile':
        return prepare_pile()

    if dataset_name == 'python':
        return prepare_code('Python-all')

    if dataset_name[:4] == 'code':
        name = dataset_name[5:]
        return prepare_code(name)

if __name__ == "__main__":
    # syntax: python texts.py model_size dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, default='125m',
        help='model size options: 125m 350m 1.3b 2.7b 6.7b 13b  30b 66b 175b')
    parser.add_argument('--dataset', type=str, default='pile')
    parser.add_argument('--count_tokens', type=bool, default=False)
    parser.add_argument('--start_index', type=int, default=1)
    args = parser.parse_args()

    if args.dataset == 'code':
        dataset, label, skip_eval = prepare_code()

    elif args.dataset == 'pile':
        dataset, label, skip_eval = prepare_pile()

    else:
        raise ValueError( f"Unknown dataset: {args.dataset}" )

    opt = Model( args.model_size )

    opt.evaluate_dataset(dataset, token_limit=1000, k=1, start_index=args.start_index,
        skip_eval=skip_eval, dataset_text_label=label, count_tokens=False )
