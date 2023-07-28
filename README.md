# separability

My basic library for studying LLMs, with support for Multi-GPU inference and
editing, as well as quantilised inference (not editing yet).
This includes functions for analysing the activations of the models for
different inputs, and for pruning different parts of the model based on those
activations.

The currently tested list of models is:
- GPT2
- EleutherAI's Pythia
- Meta Opt
- Meta Galactica

## Pruning based on Capabilities

For a full example, see `src/examples/prune_30.py`.

The simple example is:
```
from separability.data_classes import PruningConfig
from separability.parser import cli_parser
from separability.prune import run_pruning

# Configure initial model and tests
c = PruningConfig(
    wandb_project = "testing",
    model_repo   = "facebook/opt-125m",
    token_limit  = 1000,
    run_pre_test = True,

    # Removals parameters
    ff_scoring = "abs"
    ff_frac   = 0.02,
    ff_eps    = 0.001,
    attn_scoring = "abs",
    attn_frac = 0.00,
    attn_eps  = 1e-4,

    # Eval
    focus     = "pile_codeless",
    cripple   = "code",
    additional_datasets=tuple(),
)

# optionally, use parser to get CLI arguments.
# c, args = cli_parser(c)

# Run the iterated pruning
model, history = run_pruning(c)

```

## model.py
This defines a wrapper function that encapsulates the HuggingFace implementation of Meta OPT.
To get the model, simply run:

```
from separability import Model

m = Model("facebook/opt-125m", limit=1000)
```

Where you can provide any of the model sizes that are pre-trained for OPT, and the token limit must be smaller than the max token length that the model is able to handle.

Next, you can run the model to do 2 tokens of predictions, by, for example, running:
```
text = 'Hello, my name is'
inpt, output = opt.predict( text, num=2 )
```

We can look at the residual stream of how the output changes over time.
```
residual_stream = opt.get_residual_stream( text )
```
This will return a tensor of size `2 + 2*n_layers`.
i.e:
- the input (w/ positional encoding)
- n attention layer outputs
- n feed forward layer outputs
- the final output

If we want just the output of the attention / feed forward layers, we can instead look at the activations:
```
inpt, attn_out, ff_out, output = opt.get_text_activations( text )
```
or alternatively:
```
inpt, attn_out, ff_out, output = opt.get_text_activations( residual_stream=residual_stream )
```

To get the activations for the input text at all of the MLP mid layers, we can look at:
`opt.get_ff_key_activations( text )` or `opt.get_ff_key_activations( residual_stream=residual_stream )`.

## texts.py
Has some basic tools for loading the two text datasets I am using:
- 'pile', ( EleutherAI's 'The Pile' dataset)
- 'code' (CodeParrot's 'github-code' dataset)

## activations.py
Has code specific to the two datasets I am using to analyze and attempt to remove capabilities from the models.

