import datasets
from transformers import GPT2Tokenizer, OPTForCausalLM
from transformers.models.opt.modeling_opt import OPTAttention
import torch
import numpy as np
import copy
from tqdm.notebook import tqdm
from welford import Welford

# import types for typed python
from typing import Optional, List, Tuple
from torch import Tensor

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

# Return with the output tensors detached from gpu
def detached( output ):
    if type(output) is tuple:
        return ( detached(out) for out in output )
    if type(output) is Tensor:
        return output.detach()
    return None

def pad_zeros( d, n=2 ):
    s = str(d)
    k = n - len(s)
    k = k if k > 0 else 0
    return "0"*k + s

class InverseLinear(torch.nn.Module):
    """_summary_
    Produces a torch layer which undoes what the "original" linear layer does.

    This was built to get the output right before out_proj, and I thought
    it would be easier and/or faster than using the values matrix 
    (it probably is not actually easier or faster)

    Args:
        original: The original Linear layer we are getting the inverse of
    """
    def __init__(self, original: torch.nn.Linear):
        super(InverseLinear, self).__init__()
        weights, bias = original.parameters()

        # Check that the original transformation is square
        original_size = weights.size()
        if original_size[0] != original_size[1]:
            raise ValueError("Original Linear Layer must be square")

        # Define the Inverese Bias vector
        self.inverse_bias = -bias

        # Define the Inverse Linear Layer
        inverse_weights = weights.inverse()
        size = inverse_weights.size()
        self.fc = torch.nn.Linear( size[0], size[1], bias=False )
        self.fc.load_state_dict({'weight': inverse_weights})

    def forward(self, x):
        y = x + self.inverse_bias
        y = self.fc( y )
        return y

model_sizes = [ "125m", "350m", "1.3b", "2.7b", "6.7b", "13b", "30b", "66b", "175b" ]

class Model():
    def __init__( self, model_size : str  = "125m", limit: int = None ):
        """
        OPT Model with functions for extracting activations.
        model_size : 125m, 350m, 1.3b, 2.7b, 6.7b, 13b, 30b, 66b, 175b
        """

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.init_model( model_size )
        self.limit = limit
        self.to( self.device )

        # Indices of outputs for reference
        self.layer_index     = -3
        self.token_index     = -2
        self.dimension_index = -1
    
    def set_repo( self, model_size: str ):
        if model_size not in model_sizes:
            raise ValueError( "model_size must be one of the following: " +
                              str(model_sizes) )
        self.model_size = model_size
        repo = f"facebook/opt-{model_size}"
        self.repo = repo

    def init_model( self, model_size: Optional[str] = None ):
        if not model_size is None:
            self.set_repo( model_size )
        self.tokenizer = GPT2Tokenizer.from_pretrained( self.repo )
        self.predictor = OPTForCausalLM.from_pretrained( self.repo )
        self.model = self.predictor.model
        print(f'- Loaded OPT-{self.model_size}')
        self.activations = {}

        attn0 = self.model.decoder.layers[0].self_attn
        self.d_model = attn0.embed_dim
        self.d_head  = attn0.head_dim
        self.n_heads = attn0.num_heads
        self.n_layers = len(self.model.decoder.layers)

        self.register_activations()
        self.register_inverse_out_proj()

    def to( self, device ):
        self.device = device
        self.predictor.to( device )
        self.model.to( device )

    def get_activation_of( self, name : str ):
        # Define hook function which adds output to self.activations
        def hook(model, input, output):
            if not type( output ) is tuple:
                return
            self.activations[name] = detached( output )
        return hook

    def register_activations( self ):
        # register the forward hook
        decoder_index   = 0
        attention_index = 0
        for module in self.model.decoder.layers.modules(): 
            if type(module) is OPTAttention:
                name = pad_zeros( attention_index ) + "-attention" 
                # print( f"registering : ({name}), OPTAttention layer" )
                module.register_forward_hook( self.get_activation_of( name ) )
                attention_index += 1
                continue
        print( f" - Registered {attention_index} OPT Attention Layers" )

    def register_inverse_out_proj( self ):
        # Make it possible to get the output right before out_proj
        for layer in self.model.decoder.layers:
            layer.self_attn.inv_out_proj = InverseLinear( layer.self_attn.out_proj )
            layer.self_attn.inv_out_proj.to(self.device)

    def get_ids( self, text:str, limit:Optional[int]=None ):
        limit = self.limit if (limit is None) else limit
        input_ids = self.tokenizer( text, return_tensors='pt').input_ids
        if not limit is None:
            input_ids = torch.stack([ input_ids[0][:limit] ])
        return input_ids.to( self.device )
    
    def get_inputs_embeds( self,
                text: Optional[str] = None,
                input_ids: Optional[Tensor] = None,
                verbose: bool = False,
                limit: Optional[int] = None
            ):
        if input_ids is None:
            input_ids = self.get_ids( text, limit )

        inputs_embeds = self.model.decoder.embed_tokens( input_ids )

        return inputs_embeds

    def get_recent_activations( self ) -> List[ Tuple[str, Tensor, Tensor, Tensor] ]:
        """
        Returns a list of output tuples \
        ( "##-attention", output, key_values, attention_output ) \
        from each attention block
        """
        layers = []
        for key, value in self.activations.items():
            layer = []
            layer.append( key )
            for out in value:
                if type(out) is Tensor:
                    layer.append( out.detach().cpu() )
                    continue
        
            if out is None:
                continue
        
            for o in out:
                layer.append( o )

            layers.append(layer)

        return layers

    def get_text_activations( self,
                text: Optional[str] = None,
                input_ids: Optional[Tensor] = None,
                inputs_embeds: Optional[Tensor] = None,
                verbose: bool = False,
                limit: Optional[int] = None,
                **kwargs
            ):
        """_summary_
        Gives the output of each major component of the transformer before being added to
        the residual_stream. i.e: ( input, attention_out, ff_out, output )

        Args:
            text (Optional[str], optional): _description_. Defaults to None.
            input_ids (Optional[Tensor], optional): _description_. Defaults to None.
            inputs_embeds (Optional[Tensor], optional): _description_. Defaults to None.
            verbose (bool, optional): _description_. Defaults to False.
            limit (Optional[int], optional): _description_. Defaults to None.

        Returns:
            List: Tensor
                input: The input tensor with positional encodings.
                attention_out: Intermedate attention output activations.
                ff_out: The intermedate ff output activations.
                output: The final output tensor.
        """
        if inputs_embeds is None and input_ids is None and text is None:
            raise ValueError( "must provide data: inputs_embeds | input_ids | text" )

        if text or (not input_ids is None):
            inputs_embeds = self.get_inputs_embeds( text, input_ids, verbose, limit )

        # run the model
        outputs = self.model( inputs_embeds=inputs_embeds,
                              output_hidden_states=True, **kwargs )

        # get the hidden states
        hidden_states = torch.stack( outputs.hidden_states ).squeeze().detach().cpu()
        inpt = hidden_states[0].detach()

        # get attention outputs
        attention_out = torch.stack([ out[1] for out in self.get_recent_activations() ])
        attention_out = attention_out.squeeze().detach()

        # get ff outputs
        ff_out =  [] 
        for i in range(len(attention_out)):
            ff_out.append( hidden_states[i+1] - attention_out[i] - hidden_states[i] )
        ff_out = torch.stack( ff_out ).squeeze().detach().detach()

        # get final output
        output: Tensor = outputs.last_hidden_state[0].detach()

        return inpt, attention_out, ff_out, output

    def get_residual_stream( self,
                text: Optional[str] = None,
                input_ids: Optional[Tensor] = None,
                inputs_embeds: Optional[Tensor] = None,
                text_activations: Optional[List[Tensor]] = None,
                verbose: bool = False,
                limit: Optional[int] = None,
                **kwargs
            ) -> Tensor:

        if text_activations is None:
            text_activations = self.get_text_activations( text,
                input_ids, inputs_embeds, verbose, limit, **kwargs )
        inpt, attention_out, ff_out, output = text_activations

        assert len(attention_out) == self.n_layers
        assert len(ff_out) == self.n_layers

        adjustments = [0]*(2*self.n_layers)
        adjustments[0::2] = attention_out
        adjustments[1::2] = ff_out


        residual_stream = []
        residual_stream.append( inpt )

        for delta in adjustments:
            residual_stream.append( residual_stream[-1] + delta )
        
        return torch.stack( residual_stream )

    def get_ff_key_activations( self,
                text: Optional[str] = None,
                input_ids: Optional[Tensor] = None,
                inputs_embeds: Optional[Tensor] = None,
                text_activations: Optional[List[Tensor]] = None,
                residual_stream: Optional[Tensor] = None,
                verbose: bool = False,
                limit: Optional[int] = None,
                **kwargs
            ) -> Tensor:
        if residual_stream is None:
            residual_stream = self.get_residual_stream( text, input_ids,
                inputs_embeds, text_activations, verbose, limit, **kwargs )
        
        ff_inputs = residual_stream[1:-2:2]
        ff_keys = self.calculate_ff_keys( ff_inputs.to(self.device) )

        return ff_keys
    
    def get_attn_pre_out_activations( self,
                text: Optional[str] = None,
                input_ids: Optional[Tensor] = None,
                inputs_embeds: Optional[Tensor] = None,
                text_activations: Optional[List[Tensor]] = None,
                verbose: bool = False,
                limit: Optional[int] = None,
                reshape: bool = True,
                transpose: bool = False,
                **kwargs
            ) -> Tensor:
        if text_activations is None:
            text_activations = self.get_text_activations( text,
                inputs_embeds, input_ids, verbose, limit, **kwargs )
        
        [ inpt, attn_out, ff_out, output ] = text_activations
        pre_outs = self.calculate_attn_pre_out(
            attn_out.to(self.device), reshape, transpose )

        return pre_outs
    
    # Functions for calculating attention
    # Brief description of attention mechanism with OPTAttention reference:
    # input: x_i
    # then: x_i -> k_i, q_i, v_i 
    # then: k_i, q_j            -> attention a_ij  "attn_weights"
    # then: sum_i( a_ij * v_j ) ->                 "attn_pre_out"
    # then: W_o * pre_out       -> output          "attn_out"
    # output: attn_out, attn_weights, (k_i, v_i)

    def prepare_attention_mask( self, input: Tensor ):
        decoder = self.model.decoder
        input_shape = input.size()[:-1]

        # embed positions
        attention_mask = torch.ones( input_shape, dtype=torch.bool, device=input.device )
        attention_mask = decoder._prepare_decoder_attention_mask(
            attention_mask, input_shape, input, past_key_values_length=0
        )
        return attention_mask

    def calculate_attn_out_layer( self,
                input: Tensor,
                layer: int,
                attention_mask: Tensor
            ):
        u = self.model.decoder.layers[ layer ]
        x = u.self_attn_layer_norm( input )
        x = u.self_attn( x, attention_mask=attention_mask )[0]
        return x

    def calculate_attn_out( self, attn_in: Tensor, add_residual: bool = False ):
        attention_mask = self.prepare_attention_mask( input )

        outs = []
        for layer, input in enumerate(attn_in):
            attn_out = self.calculate_attn_out_layer(input, layer, attention_mask)
            if add_residual:
                attn_out += input
            outs.append( attn_out )
        return torch.stack( outs )

    def calculate_attn_pre_out_layer( self,
            attn_out: Tensor,
            layer: int,
            reshape: bool,
            transpose: bool
            ):
        # Returns attention activations in the layer right before output
        # ie: turns attn_out into attn_pre_out
        self_attn = self.model.decoder.layers[layer].self_attn

        # Calculate the layer before
        pre_out = self_attn.inv_out_proj( attn_out )

        # reshape into the shape it was before W_out
        if reshape:
            [ tgt_len, embed_dim ] = attn_out.size() # see OPTAttention
            pre_out = pre_out.view(tgt_len, self.n_heads, self.d_head)

        # whether to transpose the output to what it originally looked like
        if reshape and transpose:
            pre_out = pre_out.transpose( 0, 1 )

        return pre_out

    def calculate_attn_pre_out( self,
            attn_out: Tensor,
            reshape: bool = True,
            transpose: bool = False
            ):
        out = []
        for layer in range(len(attn_out)):
            pre_out = self.calculate_attn_pre_out_layer(
                attn_out[layer], layer, reshape, transpose )
            out.append( pre_out)
        return torch.stack( out )

    def delete_attn_pre_out_layer( self,
            layer_index: int,
            remove_indices: Tensor,
            mean_values: Optional[Tensor] = None
            ):
        """
        A function that deletes the impact that the pre_out layer has on the model

        Args:
            layer_index (int): Layer of attention in which out_proj is being pruned.
            indices (Tensor): a tensor of size (d_model) or (n_heads, d_head) which
                has value True at each index which will be pruned.
            mean_values (Optional[Tensor], optional): The value to offset the output 
                by to compensate for the fact it is no longer in service.
                Defaults to None.
        """
        if type(remove_indices) is np.ndarray:
            remove_indices = torch.tensor(remove_indices, dtype=torch.float32)
        if type(mean_values) is np.ndarray:
            mean_values = torch.tensor(mean_values, dtype=torch.float32)

        with torch.no_grad():
            # check tensor sizes are correct
            size = remove_indices.size()
            if size[-1] == self.d_head:
                remove_indices = remove_indices.reshape(size[:-2], -1)
                size = remove_indices.size()
            assert remove_indices.size() == torch.Size([self.d_model])

            # get the attention that we are changing
            out_proj = self.model.decoder.layers[layer_index].self_attn.out_proj
            params   = out_proj.state_dict()
            weights : Tensor = params['weight']
            biases  : Tensor = params['bias']

            # adjust bias of out_proj by mean activations
            if not mean_values is None:
                assert mean_values.size() == torch.Size([self.d_model])
                mean_values  = copy.deepcopy(mean_values)
                mean_values *= remove_indices
                bias_adjustement = torch.matmul( weights, mean_values )
                biases += bias_adjustement

            # change weights of out_proj
            # We change it from "True => Remove" to "True => keep" so we can multiply
            keep_indices = torch.logical_not( remove_indices ).to( self.device )

            for row_index in range(len(weights)):
                weights[row_index] = weights[row_index] * keep_indices

            params.update({'weight': weights, 'bias': biases})
            out_proj.load_state_dict(params)
            return

    def delete_attn_pre_out( self,
            remove_indices: Tensor,
            mean_values: Tensor = None,
        ):
        """Delete effect of attn_pre_out for neurons at indices {remove_indices}.
        Optionally offset the output my some mean activation {mean_values}.

        Args:
            remove_indices (Tensor): Tensor of type [n_layer, n_heads, d_head] or
                [n_layer, d_model] with value True for nodes of attn_pre_out to
                prune / make inactive.
            mean_values (Tensor, optional): Mean activation to adjust the bias to compensate
                for the deletion of the attn_pre_out interactions. Defaults to None.

        Returns:
            self (Model)
        """
        use_means = not (mean_values is None)
        if use_means:
            assert mean_values.size() == remove_indices.size()
        
        for layer_index in range(len(remove_indices)):
            mean_values_layer = mean_values[layer_index] if use_means else None
            self.delete_attn_pre_out_layer( layer_index,
                remove_indices[layer_index], mean_values_layer )
        
        return self

    def delete_attn_pre_out_heads( self,
            remove_heads: Tensor,
            means: Tensor = None,
        ):
        """remove specific attention heads from model, and optionally offset
        activation by some mean activation

        Args:
            remove_heads (Tensor): tensor of model heads to remove of size 
                [n_layers, n_heads], with value True if you want to remove it
            means (Tensor, optional): tensor of means to offset activations by.
                Defaults to None.
        """
        # Check that the size for remove_heads is correct
        if remove_heads.size() != torch.Size([ self.n_layers, self.n_heads ]):
            raise ValueError( "Removals must have dimension [n_layers, n_heads]" )

        # if using means, check sizes are correct
        means_i = None

        # delete heads in each layer
        for layer in range(self.n_layers):
            # if using means, get means for current layer
            if not means is None:
                means_i = means[layer]
                means_i = torch.tensor( means_i.flatten(), dtype=torch.float32 )
                assert means_i.size() == torch.Size([ self.d_model ])

            # Convert 'heads' tensor into 'individual neurons' tensor 
            remove_indices = np.ones([ self.n_layers, self.n_heads, self.d_head])
            for head_index, val in enumerate(remove_heads[layer]):
                remove_indices[head_index] *= ( val )
            remove_indices = torch.tensor( remove_indices.flatten(), dtype=torch.float32 )

            self.delete_attn_pre_out_layer( layer, remove_indices, means_i )

    # Functions for calculating feed-forward fully connected layer activations
    def calculate_ff_keys_layer( self, ff_in: Tensor, layer: int ):
        u = self.model.decoder.layers[ layer ]
        x = u.final_layer_norm( ff_in )
        x = u.fc1( x )
        x = u.activation_fn( x )
        return x

    def calculate_ff_keys( self, ff_in: Tensor ):
        out = []
        for layer, input in enumerate(ff_in):
            out.append( self.calculate_ff_keys_layer( input, layer ) )
        return torch.stack( out )

    def calculate_ff_out_layer( self, ff_in: Tensor, layer: int):
        u = self.model.decoder.layers[ layer ]
        x = u.final_layer_norm( ff_in )
        x = u.fc1( x )
        x = u.activation_fn( x )
        x = u.fc2( x )
        return x

    def calculate_ff_out( self, ff_in: Tensor, add_residual: bool = False ):
        out = []
        for layer in range(len(ff_in)):
            ff_out = self.calculate_ff_out_layer( ff_in[layer], layer )
            if add_residual:
                ff_out += ff_in[layer]
            out.append( ff_out )
        return torch.stack( out )
    
    # functions for 'deleting' neurons from the MLP mid layers
    def delete_ff_keys( self, layer_key_map: Tensor ):
        for layer, key_map in enumerate(layer_key_map):
            # Get state dict
            ff_1  = self.model.decoder.layers[ layer ].fc1
            state_dict = ff_1.state_dict()

            # set biases to -inf to 'delete' the keys (due to ReLU)
            for index, delete in enumerate(key_map):
                if delete:
                    state_dict['bias'][index] = - torch.inf
            ff_1.load_state_dict( state_dict )

    def delete_ff_keys_from_files( self, files: List[str] ):
        """Delete ff mid layer neurons from list of numpy files
        pointing to which neurons to delete.

        Args:
            files (List[str]): List of ".npy" file paths
        """
        if len( files ) == 0:
            return

        criteria = None
        for filename in files:
            ff_criterion = np.load(filename)
            if criteria is None:
                criteria = np.zeros_like( ff_criterion )
            criteria += ff_criterion

            sums = [ x.sum() for x in ff_criterion ]
            print( "%5d -"%np.sum(sums), sums )

        self.delete_ff_keys( criteria )

    # Next token prediction, show tokens
    def predict( self, text : str, num: int = 10, limit: Optional[int] = None ):
        """ Predict the next {num} tokens from an input {text}."""

        inputs = self.tokenizer( text, return_tensors="pt" )
        input_ids = inputs.input_ids

        if limit:
            input_ids = input_ids[0][:limit].reshape(1, -1)

        new_len = len(input_ids[0])+num
        generate_ids = self.predictor.generate( input_ids, max_length=new_len )
        
        before = self.tokenizer.batch_decode( input_ids,
            skip_special_tokens=True, clean_up_tokenization_spaces=False )[0]
        after  = self.tokenizer.batch_decode( generate_ids,
            skip_special_tokens=True, clean_up_tokenization_spaces=False )[0]
        after = after[ len(before): ]
        return before, after

    def get_nth_tokens( self, output: Tensor, n: int = 16 ):
        L = output.size()[self.token_index]
        indices = torch.tensor( list(range( n-1, L, n )) )

        return torch.index_select( output, self.token_index, indices )

    def unembed( self, embedded_outputs: Tensor ):
        lm_head = self.predictor.get_output_embeddings()
        return lm_head( embedded_outputs.to(self.device) )
 
    def get_all_logits( self, input_ids ):
        """Get output logits from input token ids"""

        outputs = self.model( input_ids, output_hidden_states=False )
        logits = self.unembed( outputs.last_hidden_state )

        return logits

    def top_k_tokens( self, logits: Tensor, k: int = 10 ):
        topk = torch.topk( logits, k, dim=-1, largest=True, sorted=True )
        return topk.indices.squeeze()

    def predict_top_k_tokens( self, text: str, k: int = 10 ):
        logits = self.get_all_logits( text )
        return self.top_k_tokens( logits, k=k )

    def evaluate_top_k_performance( self,
            k: int,
            text: Optional[str] = None,
            input_ids: Optional[Tensor] = None,
            logits: Optional[Tensor] = None,
            start_index: int = 1,
            skip_strings: List[str] = [],
        ):
        """Evaluates performance with top-1 and top-k token predictions.

        Args:
            text (str, optional): The text to evaluat.
            input_ids (Tensor, optional): The input IDs from text to evaluate.
            logits (Tensor, optional): The pre-computed logits from text to evaluate.
            k (int): the number of tokens to consider.
            start_index (int, optional): The starting index from which to count.
                Defaults to 1.
            skip_strings (List[str], optional): Tokens to skip when evaluating.
                Defaults to [].

        Returns:
            output: dict of output information
        """
        if text is None and input_ids is None:
            raise ValueError( "Must provide either text or input_ids" ) 

        # Generate input token ids and output top k token ids
        with torch.no_grad():
            input_ids = self.get_ids( text ) if input_ids is None else input_ids
            logits = self.get_all_logits( input_ids ) if logits is None else logits
            token_dictionary_size = logits.size()[-1]
            top_tokens  = self.top_k_tokens( logits, 1 )
            topk_tokens = self.top_k_tokens( logits, k ) if k!=1 else top_tokens
        
        # Get the set of token ids to skip when evaluating performance
        skip_ids = set()
        for skip_string in skip_strings:
            skip_id = int( self.get_ids( skip_string ).squeeze()[-1] )
            skip_ids.add( skip_id )

        # Count top-k prediction accuracy 
        input_ids = input_ids.squeeze()
        num_predictions = 0
        num_accurate = 0
        num_topk_accurate = 0
        num_skip_predictions = 0
        num_skip_accurate = 0
        num_topk_skip_accurate = 0
        for i in range( start_index, len(input_ids) ):
            # Check if accurate
            is_accurate      = (input_ids[i] in top_tokens[i-1])
            is_topk_accurate = (input_ids[i] in topk_tokens[i-1])

            # Count normal prediction ( ignoring skip )
            num_predictions   += 1
            num_accurate      += is_accurate
            num_topk_accurate += is_topk_accurate

            # Now count only if the token is not supposed to be skipped
            if int(input_ids[i]) in skip_ids:
                continue
            num_skip_predictions   += 1
            num_skip_accurate      += is_accurate
            num_topk_skip_accurate += is_topk_accurate

        # Keep track of most used tokens
        token_counts = np.zeros( token_dictionary_size )
        for id in input_ids:
            token_counts[id] += 1

        output = {
            'num_predictions'  : num_predictions,
            'num_accurate'     : num_accurate,
            'num_topk_accurate': num_topk_accurate,
            'num_skip_predictions'  : num_skip_predictions,
            'num_skip_accurate'     : num_skip_accurate,
            'num_topk_skip_accurate': num_topk_skip_accurate,
            'token_counts': token_counts,
            'input_ids'   : input_ids,
            'top_tokens'  : top_tokens,
            'top_k_tokens': topk_tokens,
        }

        return output
    
    def evaluate_ce_loss( self, 
            text: Optional[str] = None,
            input_ids: Optional[Tensor] = None,
            logits: Optional[Tensor] = None
        ):
        """Cross entropy loss for predicting the next token

        Args:
            text (str, optional): The text to evaluat.
            input_ids (Tensor, optional): The input IDs from text to evaluate.
            logits (Tensor, optional): The pre-computed logits from text to evaluate.
        
        Returns:
            loss: Mean Cross-Entropy loss over tokens
        """
        if text is None and input_ids is None:
            raise ValueError( "Must provide either text or input_ids" ) 

        # Generate input token ids and output top k token ids
        with torch.no_grad():
            if input_ids is None:
                input_ids = self.get_ids( text )
            if logits is None:
                logits = self.get_all_logits( input_ids )
        
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        predicted_log_probs = log_probs[..., :-1, :].gather(
            dim=-1, index=input_ids[..., 1:, None]
        )[..., 0]
        return -predicted_log_probs.mean()

    def batch_decode( self, input_ids ):
        output_str = self.tokenizer.batch_decode( input_ids )
        return output_str

    def calculate_evaluation_percentages( self, out: dict, string: bool = False ):
        # Print top1 prediction accuracy
        pred      = out["num_predictions"]
        skip_pred = out["num_skip_predictions"]
        percent = {
            "base": (100 * out["num_accurate"] / pred),
            "topk": (100 * out["num_topk_accurate"] / pred),
            "skip": (100 * out["num_skip_accurate"] / pred),
            "topk_skip": (100 * out["num_topk_skip_accurate"] / skip_pred ),
        }
        if string:
            percent = { k: ('%.2f'%v) for k, v in percent.items() }
        return percent

    def evaluate_dataset( self,
            dataset: datasets.Dataset,
            dataset_text_label: str = 'content',
            k: int = 10,
            start_index: int = 1,
            skip_eval: list = [],
            sample_size: int = 1e5,
            token_limit: Optional[int] = None,
            count_tokens: bool = False,
            num_top_tokens: int = 50,
            ):
        """An evaluation of next-token prediction accuracy for an iterable Dataset,
        which includes options for topk evaluation as well as skipping the most
        commonly occuring tokens.

        Args:
            dataset (datasets.Dataset): the Datset object to iterate over.
            dataset_text_label (str, optional): The label for the dataset text.
                eg: 'text'. 'content'. Defaults to 'content'.
            k (int, optional): Number of topk tokens to look at when asessing
                the accuracy of the model (in addition to top1). Defaults to 10.
            start_index (int, optional): The index of the first token to evaluate.
                Defaults to 1.
            skip_eval (list, optional): A list of token IDs to skip when evaluating.
                Defaults to [].
            sample_size (int, optional): The number of tokens to evaluate.
                Defaults to 1e5.
            token_limit (Optional[int], optional): The maximum length of tokens
                to allow before trimming each text input. Defaults to None.
            count_tokens (bool, optional): . Defaults to False.
            num_top_tokens (int, optional): If count_tokens is true, this number
                determines the number of top most-common accurate token predictions
                to output when counting tokens. Defaults to 50.

        Returns:
            dict: A dictionary which contains counts of #predictions and #accurate
                accross various the various possible combinations of outputs,
                as well as a sub dict 'percent' which contains percentage accuracy.
        """
        
        # Initialize variables
        limit = self.limit if token_limit is None else token_limit
        token_counts = None
        out = {
            "num_predictions": 0,
            "num_accurate": 0,
            "num_skip_predictions": 0,
            "num_skip_accurate": 0,
            "num_topk_accurate": 0,
            "num_topk_skip_accurate": 0,
            "token_counts": None,
        }
        loss_tracker = Welford()

        # Set up approx stopping index
        with tqdm(total=sample_size) as pbar:

            # Loop over the dataset
            for data in dataset:
                # predict next token from text
                text = data[ dataset_text_label ]
                with torch.no_grad():
                    input_ids = self.get_ids( text, token_limit )
                    logits = self.get_all_logits( input_ids )
                
                # perform evaluations
                topk =  self.evaluate_top_k_performance( k, input_ids=input_ids,
                    logits=logits, start_index=start_index, skip_strings=skip_eval )
                loss = self.evaluate_ce_loss( input_ids=input_ids, logits=logits )

                # Record performance
                loss_tracker.add( loss.detach().cpu().numpy() )
                out["num_predictions"]    += topk['num_predictions']
                out["num_accurate"]       += topk['num_accurate']
                out["num_topk_accurate"]  += topk['num_topk_accurate']
                out["num_skip_predictions"]   += topk['num_skip_predictions']
                out["num_skip_accurate"]      += topk['num_skip_accurate']
                out["num_topk_skip_accurate"] += topk['num_topk_skip_accurate']
                out["loss"] = loss_tracker.mean
                pbar.update( topk["num_skip_predictions"] )

                # Record token counts
                if count_tokens:
                    # Get current token counts
                    if token_counts is None:
                        token_counts = np.zeros_like( topk['token_counts'] )
                    token_counts += topk['token_counts']

                    # Save token counts
                    out["token_counts"] = token_counts

                # Print output string showing current accuracy
                percent  = self.calculate_evaluation_percentages( out, string=True )
                out_str  =   f"Acc: {percent['base']}|{percent['topk']} "
                out_str += f"(Skip: {percent['skip']}|{percent['topk_skip']})"
                pbar.set_description( out_str )
                
                # Stop if limit is reached
                if out["num_skip_predictions"] > sample_size:
                    break

            if count_tokens:
                topk = token_counts.argpartition(-num_top_tokens )[-num_top_tokens:]
                topk = topk[np.argsort(token_counts[topk])][::-1]
                print( self.batch_decode( topk ) )

            out['percent'] = self.calculate_evaluation_percentages( out ) 
            return out