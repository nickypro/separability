from typing import List, Callable
import argparse
from .data_classes import PruningConfig

def cli_parser(
        c: PruningConfig,
        add_args_fn: Callable = None,
        add_args_exclude: List[str] = None,
    ):
    """
    Args:
        c (PruningConfig): default pruning config data class
        add_args_fn (Callable[parser] -> None): function to add additional
            arguments to the parser )
        add_arguments_exclude (List[str]): argument strings to manually exclude

    Returns:
        c (PruningConfig): updated pruning config data class
        args (argparse.Namespace): parsed arguments
    """
    # Build a CLI parser
    parser = argparse.ArgumentParser()

    if add_args_exclude is None:
        add_args_exclude = []

    if add_args_fn is not None:
        add_args_fn(parser)

    parser.add_argument('model_repo', type=str)
    parser.add_argument('-n', "--name", type=str, default=None, help="wandb run name")
    parser.add_argument('-r', '--reverse', action='store_true', help="cripple <--> focus")
    parser.add_argument('--n_steps', type=int, default=None)
    parser.add_argument('--model_device', type=str, default=None)

    args_exclude = [
        "model_repo",
        "name",
        "wandb_run_name",
        "reverse",
        "n_steps",
        "model_device",
        "additional_datasets",
        *add_args_exclude,
    ]
    for key, val in c.arg_items(exclude=args_exclude):
        parser.add_argument(f'--{key}', type=type(val), default=val)

    # Parse the argument
    args = parser.parse_args()

    c.model_repo = args.model_repo
    c.model_device = args.model_device
    c.wandb_run_name = args.name
    for key in c.arg_keys(args_exclude):
        c[key] = getattr(args, key)
    if args.reverse:
        c.focus, c.cripple = c.cripple, c.focus
    # First do some pruning of the feed forward layers
    n_steps = args.n_steps
    if n_steps is None:
        n_steps = int( 1 / max(c.ff_frac, c.attn_frac) )
    c.n_steps = n_steps

    return c, args
