from datasets import load_dataset
from evaluate import evaluator
import torch
import numpy as np
from model import Model, most_common_code_tokens
import argparse

Text2TextEvaluator = evaluator("text2text-generation")

def load_code():
    dataset = load_dataset("codeparrot/codeparrot-clean-valid")
    return dataset['train']

def load_pile():
    dataset = load_dataset("the_pile", split="validation", streaming=True )
    return dataset

#skip_eval = set([' ', '\n', '.', '_', ',', '#', '\n\n', '\t' ])
#skip_eval = set([' ', '\n', '.', '_', ',', '#', ' =', '(', ' import', 'from', ' the', '\t', ':', ')', " '", '\n\n', '-', '/', 'import', '):', "'", "',", ' self', 'self', ' "', ' of', '__', '=', ' (', '</s>', ' in', ' is', 's', ' License', '\r', ' for', '0', ' def', "('", ';', '1', '()', ' -', ' #', 'XXXX', '2', ' and', '",', ' to', ' as'])
most_common_pile_tokens = ['\n', '.', ',', ' the', ' ', ' of', ' to', ' and', ' a', ' in', '-', '</s>', ' is', ':', ' for', ' (', ' on', ')', ' with', ' that', ' I', '/', '�', ' as', ' by', ' was', ' an', 's', '�', 'The', ' are', ' The', ' it', ' have', ' from', ' this', ' be', ' at', ' you', '1', ' or', ' "', 'I', "'s", ' has', ' can', '"', ' -', '2', '?']

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
        code = load_code()
        dataset, label, skip_eval = code, 'content', most_common_code_tokens

    elif args.dataset == 'pile':
        pile = load_pile()
        dataset, label, skip_eval = pile, 'text',    most_common_pile_tokens
    
    else:
            raise ValueError('Unknown dataset: {}'.format(args.dataset))

    opt = Model( args.model_size )

    opt.evaluate_dataset( dataset, token_limit=1000, k=1, start_index=args.start_index,
        skip_eval=skip_eval, dataset_text_label=label, count_tokens=False )