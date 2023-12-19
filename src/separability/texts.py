"""
Some commands used for loading datasets used in my research.
That is, the 'codeparrot-clean' and 'the pile' datasets.
"""

import os
import json
import argparse
from datasets import load_dataset, concatenate_datasets

from .data_classes import EvalConfig
from .model import Model

# For each of these, we add a "test" argument:
#     If test == 0: use the "train" split
#     If test > 0 and there is a "test" split: return the "test" split
#     Else, return the train split with a skip of approx "test" tokens

# Hard load the most common tokens from the datasets from previous runs.
# pylint: disable=line-too-long
opt_most_common_code_tokens = [' ', '\n', '.', '_', ',', '#', '(', ' =', ' import', 'from', ' the', ':', ')', '\n\n', 'import', " '", '/', '-', '):', '\t', "',", ' "', ' self', '=', ' of', "'", '__', ' (', 'self', ' in', ' License', '</s>', ' is', '0', ' for', ' to', 's', '1', '2', ' a', ' as', '\r', ' -', ' and', ' def', ' #', 'x', '()', "('", '\\']
opt_most_common_pile_tokens = ['\n', '.', ',', ' the', ' ', ' of', ' to', ' and', ' a', ' in', '-', '</s>', ' is', ':', ' for', ' (', ' on', ')', ' with', ' that', ' I', '/', '�', ' as', ' by', ' was', ' an', 's', '�', 'The', ' are', ' The', ' it', ' have', ' from', ' this', ' be', ' at', ' you', '1', ' or', ' "', 'I', "'s", ' has', ' can', '"', ' -', '2', '?']

# Load the JSON data
def script_path(filename):
    __script_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(__script_path, filename)


json_file_path = script_path('data/llama_most_common_tokens.json')
with open(json_file_path, 'r') as file:
    llama_most_common_tokens = json.load(file)
most_common_pile_tokens          = llama_most_common_tokens["all"]["skip50"]["tokens_str"]
most_common_pile_codeless_tokens = llama_most_common_tokens["only_text"]["skip50"]["tokens_str"]
most_common_code_tokens          = llama_most_common_tokens["only_code"]["skip50"]["tokens_str"]

class DatasetFilters:
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

    @staticmethod
    def filter_birds(_dataset):
        with open(script_path("data/imagenet_birds.json"), "r") as file:
            bird_json = json.load(file)
        bird_ids = set(bird_json["id2label"].keys())
        def filter_birds_example(example):
            return str(example["label"]) in bird_ids
        bird_dataset = _dataset.filter(filter_birds_example)
        return bird_dataset

    @staticmethod
    def filter_birdless(_dataset):
        with open(script_path("data/imagenet_birds.json"), "r") as file:
            bird_json = json.load(file)
        bird_ids = set(bird_json["id2label"].keys())
        def filter_birds_out_example(example):
            return str(example["label"]) not in bird_ids
        bird_dataset = _dataset.filter(filter_birds_out_example)
        return bird_dataset

    @staticmethod
    def filter_mushroom(_dataset):
        mushroom_ids = set([ "52" ])
        def filter_mushroom_example(example):
            return str(example["fine_label"]) in mushroom_ids
        mushroom_dataset = _dataset.filter(filter_mushroom_example)
        return mushroom_dataset

    @staticmethod
    def filter_mushroomless(_dataset):
        mushroom_ids = set([ "52" ])
        def filter_mushroom_out_example(example):
            return str(example["fine_label"]) not in mushroom_ids
        mushroomless_dataset = _dataset.filter(filter_mushroom_out_example)
        return mushroomless_dataset

    @staticmethod
    def filter_rocket(_dataset):
        rocket_ids = set([ "69" ])
        def filter_rocket_example(example):
            return str(example["fine_label"]) in rocket_ids
        rocket_dataset = _dataset.filter(filter_rocket_example)
        return rocket_dataset

    @staticmethod
    def filter_rocketless(_dataset):
        rocket_ids = set([ "69" ])
        def filter_rocket_out_example(example):
            return str(example["fine_label"]) not in rocket_ids
        rocketless_dataset = _dataset.filter(filter_rocket_out_example)
        return rocketless_dataset

    @staticmethod
    def filter_veh2(_dataset):
        rocket_ids = set([ "19" ])
        def filter_rocket_example(example):
            return str(example["coarse_label"]) in rocket_ids
        rocket_dataset = _dataset.filter(filter_rocket_example)
        return rocket_dataset

def infer_dataset_config(dataset_name:str, dataset_subset:str=None):
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
            dataset_repo           = "codeparrot/github-code-clean",
            dataset_subset         = "all-all",
            dataset_text_key       = "code",
            dataset_has_test_split = False,
            skip_token_strings = most_common_code_tokens,
        ),
        EvalConfig("python",
            dataset_repo           = "codeparrot/github-code-clean",
            dataset_subset         = "Python-all",
            dataset_text_key       = "code",
            dataset_has_test_split = False,
            skip_token_strings = most_common_code_tokens,
        ),
        EvalConfig("stories",
            dataset_repo = "roneneldan/TinyStories",
            skip_token_strings = most_common_pile_tokens,
        ),
        EvalConfig("civil",
            dataset_repo = "civil_comments",
            dataset_filter = DatasetFilters.filter_civil,
            skip_token_strings = most_common_pile_tokens,
        ),
        EvalConfig("toxic",
            dataset_repo = "civil_comments",
            dataset_filter = DatasetFilters.filter_toxic,
            skip_token_strings = most_common_pile_tokens,
        ),
        EvalConfig("wiki",
            dataset_repo = "wikitext",
            dataset_subset = "wikitext-103-v1",
            sample_size = int(1e6),
            skip_token_strings = opt_most_common_pile_tokens,
        ),
        EvalConfig("toxicity",
            dataset_repo = None,
            dataset_type = "generation",
            generated_text_prompt = "I don’t care if this is controversial",
            #generated_text_prompt = " I swear, I just don’t know if",
            generated_text_length = 200,
            generated_text_include_prompt = True,
            generated_text_num_samples = 1000,
            generated_text_temperature = 1.0,
            skip_token_strings = most_common_pile_tokens,
        ),
        EvalConfig("mmlu",
            dataset_repo = "tasksource/mmlu",
            dataset_type = "mmlu",
            dataset_subset = "all", # Overwritten if use "mmlu:subject_name"
            skip_token_strings = most_common_pile_tokens,
        ),
        EvalConfig("imagenet-1k",
            dataset_split = "validation",
            dataset_repo = "imagenet-1k",
            dataset_type = "image-classification",
        ),
        EvalConfig("imagenet-1k-birds",
            dataset_split = "validation",
            dataset_repo = "imagenet-1k",
            dataset_type = "image-classification",
            dataset_filter=DatasetFilters.filter_birds,
        ),
        EvalConfig("imagenet-1k-birdless",
            dataset_split = "validation",
            dataset_repo = "imagenet-1k",
            dataset_type = "image-classification",
            dataset_filter=DatasetFilters.filter_birdless,
        ),
        EvalConfig("cifar100",
            dataset_repo = "cifar100",
            dataset_type = "image-classification",
            streaming = False,
            dataset_image_key = "img",
            num_texts_to_skip = 1,
            dataset_image_label_key = "fine_label",
        ),
        EvalConfig("cifar100-mushroom",
            dataset_repo = "cifar100",
            dataset_type = "image-classification",
            streaming = False,
            dataset_split = ["train", "test"],
            is_train_mode = True,
            num_texts_to_skip = 1,
            dataset_image_key = "img",
            dataset_image_label_key = "fine_label",
            dataset_filter=DatasetFilters.filter_mushroom,
        ),
        EvalConfig("cifar100-mushroomless",
            dataset_repo = "cifar100",
            dataset_type = "image-classification",
            streaming = False,
            dataset_split = "test",
            is_train_mode = False,
            num_texts_to_skip = 1,
            dataset_image_key = "img",
            dataset_image_label_key = "fine_label",
            dataset_filter=DatasetFilters.filter_mushroomless,
        ),
        EvalConfig("cifar100-mushroom-mia",
            dataset_repo = "cifar100",
            dataset_type = "image-membership-inference-attack",
            is_train_mode = True,
            dataset_image_key = "img",
            dataset_image_label_key = "fine_label",
            mia_retain = "cifar100-mushroomless",
            mia_retain_split = "train",
            mia_forget = "cifar100-mushroom",
            mia_forget_split = "train",
            mia_test = "cifar100",
            mia_test_split = "test",
            dataset_filter=DatasetFilters.filter_mushroom,
        ),
        EvalConfig("cifar100-rocket",
            dataset_repo = "cifar100",
            dataset_type = "image-classification",
            dataset_split = ["train", "test"],
            streaming = False,
            is_train_mode = True,
            num_texts_to_skip = 1,
            dataset_image_key = "img",
            dataset_image_label_key = "fine_label",
            dataset_filter=DatasetFilters.filter_rocket,
        ),
        EvalConfig("cifar100-rocketless",
            dataset_repo = "cifar100",
            dataset_type = "image-classification",
            dataset_split = "test",
            is_train_mode = False,
            num_texts_to_skip = 1,
            dataset_image_key = "img",
            dataset_image_label_key = "fine_label",
            dataset_filter=DatasetFilters.filter_rocketless,
        ),
        EvalConfig("cifar100-rocket-mia",
            dataset_repo = "cifar100",
            dataset_type = "image-membership-inference-attack",
            is_train_mode = True,
            dataset_image_key = "img",
            dataset_image_label_key = "fine_label",
            mia_retain = "cifar100-rocketless",
            mia_retain_split = "train",
            mia_forget = "cifar100-rocket",
            mia_forget_split = "train",
            mia_test = "cifar100",
            mia_test_split = "test",
            dataset_filter=DatasetFilters.filter_rocket,
        ),
        EvalConfig("cifar20",
            dataset_repo = "cifar100",
            dataset_type = "image-classification",
            dataset_image_key = "img",
            dataset_image_label_key = "coarse_label",
        ),
        EvalConfig("cifar20-veh2",
            dataset_repo = "cifar100",
            dataset_type = "image-classification",
            dataset_split = ["train", "test"],
            is_train_mode = True,
            dataset_image_key = "img",
            dataset_image_label_key = "coarse_label",
            dataset_filter=DatasetFilters.filter_veh2,
        ),
    ]

    # Convert into searchable dict
    labeled_eval_configs = dict([(c.dataset_name, c) for c in eval_configs])

    # Search the dict for config
    if dataset_name in labeled_eval_configs:
        eval_config = labeled_eval_configs[dataset_name]
    else:
        eval_config = EvalConfig(dataset_name)

    # Add subset data
    if dataset_subset is not None:
        eval_config.dataset_subset = dataset_subset

    # Add loading bar label if there is none
    if eval_config.loading_bar_desc is None or eval_config.loading_bar_desc == "":
        eval_config.loading_bar_desc = "%6s" % eval_config.dataset_name

    return eval_config

def prepare_dataset(eval_config: EvalConfig):
    """ Returns iterable dataset object. """

    # check if it has test split, or only a train split
    split = eval_config.dataset_split
    if split is None:
        split = "test" if eval_config.dataset_has_test_split else "train"

    # Load the dataset
    _dataset = load_dataset(
        eval_config.dataset_repo,
        eval_config.dataset_subset,
        streaming=eval_config.streaming,
    )


    # Post-split processing
    if isinstance(split, list) or isinstance(split, tuple):
        __d = [_dataset[s] for s in split]
        _dataset = concatenate_datasets(__d)

    else:
        _dataset = _dataset[split]

    # Apply filter if relevant
    if eval_config.dataset_filter is not None:
        _dataset = eval_config.dataset_filter(_dataset)

    # Skip n texts if relevant
    if eval_config.num_texts_to_skip >= 1:
        print(f"skipping {eval_config.num_texts_to_skip} texts in {eval_config.dataset_name}")

        # Skip only works for DatasetIterable. Kinda annoying ngl
        if hasattr(_dataset, "skip"):
            _dataset = _dataset.skip(eval_config.num_texts_to_skip) # Conservative skip limit
        else:
            _dataset = _dataset[eval_config.num_texts_to_skip:]

    # Skip tokens is no split
    if split == "train" and not eval_config.is_train_mode:
        skip_n = int(eval_config.num_tokens_to_skip//100)
        print( "Warning: 'pile_deduped' has no 'test' split.",
              f"Using 'train' split and skipping {skip_n} texts instead.")
        _dataset = _dataset.skip(skip_n) # Conservative skip limit

    return _dataset

def prepare(dataset_name):
    eval_config = infer_dataset_config(dataset_name)
    eval_config.dataset_split = "train"
    _dataset = prepare_dataset(eval_config)
    return _dataset, eval_config.dataset_text_key, eval_config.skip_token_strings
