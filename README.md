# opt-tools
My basic library for studying the Meta OPT models.
This includes functions for analysing the activations of the models for different inputs, and for pruning different parts of the model based on those activations.

## model.py
This defines a wrapper function that encapsulates the HuggingFace implementation of Meta OPT. 
To get the model, simply run:

```
from model import Model

opt = Model('125m', limit=1000)
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
- 'the_pile' ( validation set of The Pile )
- 'codeparrot-clean-valid' ( validation set of codeparrot )

## activations.py
Has code specific to the two datasets I am using to analyze and attempt to remove capabilities from the OPT.
