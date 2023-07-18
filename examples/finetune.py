import os
import random
from pathlib import Path
from typing import List, Union

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

def create_dataset(tokenizer: AutoTokenizer,
                   number_games: int = 10) -> Dataset:
    """Create the dataset

    This is a collection of full game prompts (tokenized).

    Args:
        tokenizer: Tokenizer
        number_games: Number of games

    Returns:
        Dataset: Full game prompts dataset
    """
    # Create the dataset from a list of game strings
    list_of_game_strings = generate_dataset(number_games)
    dataset = Dataset.from_dict({"text": list_of_game_strings})

    # Tokenize the text prompts (creates "input_ids" property for each dataset
    # item)
    dataset = dataset.map(
        lambda examples: tokenizer(examples["text"]),  # type: ignore
        batched=True
    )

    # Set the labels to be the same as the input IDs
    dataset = dataset.map(
        lambda examples: {
            "labels": examples["input_ids"]},
        batched=True)

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
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")

            # compute custom loss (suppose one has 3 labels with different weights)
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
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