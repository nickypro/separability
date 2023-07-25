import os
import random
from pathlib import Path
from typing import List, Union

import json
import numpy as np
import torch
import wandb

from torch import nn
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizer, PreTrainedTokenizerFast,
                          Trainer, TrainingArguments)
from transformers.utils import logging

from separability import Model

def map_to_ids(data, tokenizer):
    start_text = data["input"]
    full_text  = data["input"] + data["output"]
    start_ids  = tokenizer(start_text)["input_ids"]
    input_ids  = tokenizer(full_text)["output_ids"]
    return {
        "input_ids": input_ids,
        "start_index": len(start_ids.flatten())
    }


def create_dataset(tokenizer: AutoTokenizer) -> Dataset:
    """Create the dataset.

    This is a collection of full game prompts (tokenized).

    Args:
        tokenizer: Tokenizer
        number_games: Number of games

    Returns:
        Dataset: Full game prompts dataset
    """

    # load outputs.json
    with open('outputs_open_llama_3b_v2.json') as json_file:
        generations = json.load(json_file)

    # format as input and output dict
    input_list, output_list = [], []
    for gen in generations:
        input_list.append(gen["input"])
        output_list.append(gen["output"])

    # Create the dataset from a list of game strings
    dataset = Dataset.from_dict({"input": input_list, "output": output_list})

    # Tokenize the text prompts (creates "input_ids" property for each dataset
    # item)
    dataset = dataset.map(
        lambda examples: map_to_ids(examples, tokenizer),
        batched=True,
        batch_size=50,
    )

    return dataset

fine_tuned_checkpoint = Path(
    __file__).parent / "checkpoints" / "fine_tuned_gpt2"


def fine_tune(
    model: Model,
    log_weights_and_biases: bool = False,
) -> AutoModelForCausalLM:

    tokenizer   : Union[PreTrainedTokenizer, PreTrainedTokenizerFast] \
        = model.tokenizer
    transformer : PreTrainedModel = model.map["model"]

    # Create tokenized datasets (train and eval)
    train_dataset = create_dataset(tokenizer, 5000)  # type: ignore
    eval_dataset  = create_dataset(tokenizer, 50)  # type: ignore

    # Build custom loss for trainer
    class CustomTrainer(Trainer):
        def compute_loss(self, model: Model, inputs, return_outputs=False):
            labels = inputs.get("input_ids")
            start_indices = inputs.get("start_index")

            # forward passa
            outputs = model.predictor(**inputs)
            logits = outputs.get("logits")

            # get the logits and labels from the start index onwards
            logits = logits[:, start_indices:]
            labels = labels[:, start_indices:]

            # We calculate the loss only for the predicted tokens
            loss = model.evaluate_ce_loss(input_ids=labels, logits=logits)

            return (loss, outputs) if return_outputs else loss

    # Initialise Weights & Biases
    if log_weights_and_biases:
        wandb.login()
        wandb.init(entity="seperability", project="prediction-v0")

    training_args = TrainingArguments(
        output_dir=".checkpoints",
        evaluation_strategy="epoch",
        num_train_epochs=1,
        seed=0,
        data_seed=0
    )

    # Fine tune
    trainer = CustomTrainer(
        model=transformer,
        args=training_args,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore

    )
    trainer.train()

    # print model output
    out = model.generate(max_length=1000, do_sample=True)  # type: ignore
    print(tokenizer.decode(out[0], skip_special_tokens=True))

    return model
