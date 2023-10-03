import torch
import numpy as np
import wandb

from separability import Model
from separability.data_classes import RunDataHistory, PruningConfig
from separability.activations import prune_and_evaluate
from separability.eval import evaluate_all
from separability.parser import cli_parser

import random
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import ray
import torch
import trlx
import wandb
from game import TicTacToeGame
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune import CLIReporter
from ray.tune.search.bayesopt import BayesOptSearch
from transformers import AutoModelForCausalLM, AutoTokenizer

from separability import Model
from separability.data_classes import PruningConfig


wandb_project_name = "seperability"

def soft_opt_experiment(params: Dict[str, float]) -> None:
    """Soft optimization experiment

    Args:
        params: Parameters from Ray
    """

    # do initial tests
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained(model_path).to('cuda')

    samples = infer_game(model, tokenizer, num_samples=200)
    proxy_rewards: List[float] = []
    human_rewards: List[float] = []
    valid = []
    g_proxy = TicTacToeGame(check_valid_move=False, check_valid_state=False)
    g_human = TicTacToeGame()
    for s in samples:
        proxy_rewards.append(g_proxy.evaluate_game_string(s))
        human_rewards.append(g_human.evaluate_game_string(s))
        valid.append(g_human.validate_game_string(s)[0])
    proxy_rewards_arr = np.array(proxy_rewards)
    human_rewards_arr = np.array(human_rewards)
    print("########################################")
    print(f"Proxy reward of base model: {np.mean(proxy_rewards_arr)}")
    print(f"True reward of base model: {np.mean(human_rewards_arr)}")
    print(f"Validity rate of base model: {np.mean(valid)}")
    print("########################################")

    # Cutoff
    cutoff = get_cutoff()

    def reward_fn(samples, prompts=None, outputs=None):
        rewards = []
        g = TicTacToeGame(check_valid_move=False, check_valid_state=False)
        for s in samples:
            rewards.append(g.evaluate_game_string(s))
        rewards_arr = np.array(rewards)
        return loglikelihood_approx(rewards_arr, cutoff)

    # Set params from Ray
    config = default_config_override(params)

    trainer = trlx.train(
        str(valid_games_fine_tuned_checkpoint),
        reward_fn=reward_fn,
        config=config,
        prompts=["Let's play Tic Tac Toe:"] * config.train.batch_size,
        eval_prompts=["Let's play Tic Tac Toe:"] * config.train.batch_size,
        metric_fn=metrics,
    )

    # Save checkpoints
    fine_tuned_model_path = Path(__file__).parent / \
        ".checkpoints" / "soft_opt_model"
    trainer.save(fine_tuned_model_path)


def tune_function(
    train_function: Callable, param_space: Dict[str, Any], resources: Dict[str, float]
) -> None:
    """Tune a training function with Ray

    Args:
        train_function: Function to train - will receive param_space as a single parameter
        param_space: Parameter space
        resources: Resources per experiment
    """
    tune_config = tune.TuneConfig(
        mode="max",
        # Metric to optimize (can be e.g. "returns/mean" or "metrics/is_valid")
        metric="metrics/true_reward",
        # Use Bayes Search if params are being tuned
        # https://docs.ray.io/en/latest/tune/faq.html#which-search-algorithm-scheduler-should-i-choose
        search_alg=BayesOptSearch() if len(param_space) >= 1 else None,
        # scheduler=ASHAScheduler(metric="objective", mode="max"))
        num_samples=1,  # Keep sampling forever
        max_concurrent_trials=8
    )

    # Set the metrics to report to the CLI
    reporter = CLIReporter(
        max_progress_rows=10,
        metric_columns=[
            "metrics/true_reward",
            "returns/mean",
            "metrics/is_valid"]
    )

    tuner = tune.Tuner(
        tune.with_resources(train_function, resources=resources),
        param_space=param_space,
        tune_config=tune_config,
        run_config=ray.air.RunConfig(
            local_dir="ray_results",  # Needed for wandb
            callbacks=[
                WandbLoggerCallback(project=wandb_project_name)
            ],
            # log_to_file=True, # Needed
            progress_reporter=reporter,
        ),
    )

    tuner.fit()


def ray_experiment():
    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Ray: Resources per hyper parameter experiment (i.e. if you want 8
    # runs, you need 8x this number of resources)
    resources: Dict[str, float] = {
        "cpu": 1,
        "gpu": 1,
    }

    # Ray: Param config
    # Good choices from https://arxiv.org/pdf/2006.05990.pdf (in comments
    # below). Must be set using deep dictionary notation.
    param_space: Dict = {
        "method.init_kl_coef": tune.loguniform(0.1, 1),
        "optimizer.kwargs.lr": tune.loguniform(1e-5, 1e-7),
        # "method.gamma": tune.loguniform(0.95, 1.0),
        # # Float to work with search (rounded later)
        # "train.batch_size": tune.loguniform(8, 256),
        # "method.ppo_epochs": tune.loguniform(2, 16)
    }

    # Weights & Biases
    wandb.login()

    # Ray: Tune
    tune.register_trainable(wandb_project_name, soft_opt_experiment)
    tune_function(soft_opt_experiment, param_space, resources)


def single_experiment():
    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    param_dict = {
        "method.init_kl_coef": 1.0,
        "method.cliprange_reward": 4,  # Default 10
        "method.cliprange": 0.07,  # Default was 0.2
        "method.cliprange_value": 0.07,  # Default was 0.2
        "method.ppo_epochs": 4,  # default was 4
        "method.num_rollouts": 64,  # default was 64
        "method.lam": 0.95,  # was 0.95
        "method.scale_reward": True,  # type: ignore
        "optimizer.kwargs": {
            "lr": 1.0e-5,
            "betas": [0.9, 0.95],
            "eps": 1.0e-8,
            "weight_decay": 1.0e-6,
        },
        "train.batch_size": 128,
        "train.epochs": 6000,
        "train.eval_interval": 20,

    }
    soft_opt_experiment(param_dict)


# Configure initial model and tests
c = PruningConfig(
    wandb_project = "testing",
    model_repo   = "facebook/opt-1.3b",
    token_limit  = 1000,
    run_pre_test = True,
    # Removals parameters
    ff_frac   = 0.02,
    ff_eps    = 0.001,
    attn_frac = 0.00,
    attn_eps  = 1e-4,
    focus     = "pile_codeless",
    cripple   = "code",
    additional_datasets=tuple(),
)
pre_removals = []



if __name__ == "__main__":
    single_experiment()