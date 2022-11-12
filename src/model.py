from typing_extensions import Self
import datasets
from matplotlib.cbook import print_cycles
from transformers import GPT2Tokenizer, OPTModel, OPTConfig, OPTForCausalLM
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import copy
from tqdm import tqdm

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

model_sizes = [ "125m", "350m", "1.3b", "2.7b", "6.7b", "13b", "30b", "66b", "175b" ]

class Model():
    def __init__( self, model_size : str  = "125m" ):
        """
        OPT Model with functions for extracting activations.
        model_size : 125m, 350m, 1.3b, 2.7b, 6.7b, 13b, 30b, 66b, 175b
        """

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.set_repo( model_size )
        self.init_model()
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
        print(f'- Loaded OPT-{model_size}')
        self.activations = {}
        self.register_activations()

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

    def get_ids( self, text:str, limit:Optional[int]=None ):
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
                    layer.append( out )
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
        if inputs_embeds is None and input_ids is None and text is None:
            raise ValueError( "must provide data: inputs_embeds | input_ids | text" )

        if text or (not input_ids is None):
            inputs_embeds = self.get_inputs_embeds( text, input_ids, verbose, limit )

        # run the model
        outputs = self.model( inputs_embeds=inputs_embeds,
                              output_hidden_states=True, **kwargs )

        # get the hidden states
        hidden_states = torch.stack( outputs.hidden_states ).squeeze().detach()
        input = hidden_states[0]

        # get attention outputs
        attention_out = torch.stack([ out[1] for out in self.get_recent_activations() ])
        attention_out = attention_out.squeeze().detach()

        # get ff outputs
        ff_out =  [] 
        for i in range(len(attention_out)):
            ff_out.append( hidden_states[i+1] - attention_out[i] - hidden_states[i] )
        ff_out = torch.stack( ff_out ).squeeze().detach()

        # get final output
        output: Tensor = outputs.last_hidden_state[0].detach()

        return input, attention_out, ff_out, output

    def get_residual_stream( self,
                text: Optional[str] = None,
                inputs_embeds: Optional[Tensor] = None,
                input_ids: Optional[Tensor] = None,
                verbose: bool = False,
                limit: Optional[int] = None,
                **kwargs
            ) -> Tensor:

        input, attention_out, ff_out, output = self.get_text_activations( text,
            input_ids, inputs_embeds, verbose, limit, **kwargs )
        
        assert len(attention_out) == len(ff_out)
        L = len(attention_out)

        adjustments = [0]*(2*L)
        adjustments[0::2] = attention_out
        adjustments[1::2] = ff_out


        residual_stream = []
        residual_stream.append( input )

        for delta in adjustments:
            residual_stream.append( residual_stream[-1] + delta )
        
        return torch.stack( residual_stream )

    def get_ff_key_activations( self,
                text: Optional[str] = None,
                input_ids: Optional[Tensor] = None,
                inputs_embeds: Optional[Tensor] = None,
                residual_stream: Optional[Tensor] = None,
                verbose: bool = False,
                limit: Optional[int] = None,
                **kwargs
            ) -> Tensor:
        if residual_stream is None:
            residual_stream = self.get_residual_stream( text,
                inputs_embeds, input_ids, verbose, limit, **kwargs )
        
        ff_inputs = residual_stream[1:-2:2]
        ff_keys = self.calculate_ff_keys( ff_inputs )

        return ff_keys
    
    # Functions for calculating attention
    def prepare_attention_mask( self, input: Tensor ):
        decoder = self.model.decoder
        input_shape = input.size()[:-1]

        # embed positions
        attention_mask = torch.ones( input_shape, dtype=torch.bool, device=input.device )
        pos_embeds = decoder.embed_positions(attention_mask, past_key_values_length=0)
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
            ff_out = self.calculate_ff_out_layer( input, layer )
            if add_residual:
                ff_out += input
            out.append( ff_out )
        return torch.stack( out )

    # Next token prediction, show tokens
    def predict( self,
                text : str,
                num : int = 10,
                limit : Optional[int] = None
            ):

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
        return lm_head( embedded_outputs )
 
    def get_all_logits( self, input_ids ):
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
                text : str,
                k: int,
                start_index: int = 1,
                skip_strings: List[str] = [],
                limit: Optional[int] = None
            ):
        
        # Generate input token ids and output top k token ids
        with torch.no_grad():
            input_ids = self.get_ids( text, limit=limit )
            logits = self.get_all_logits( input_ids )
            token_dictionary_size = logits.size()[-1]
            top_k_tokens = self.top_k_tokens( logits, k )
        
        # Get the set of token ids to skip when evaluating performance
        skip_ids = set()
        for skip_string in skip_strings:
            skip_id = int( self.get_ids( skip_string ).squeeze()[-1] )
            skip_ids.add( skip_id )

        # Count top-k prediction accuracy 
        input_ids = input_ids.squeeze()
        num_predictions = 0
        num_accurate = 0
        num_skip_predictions = 0
        num_skip_accurate = 0
        for i in range( start_index, len(input_ids) ):
            # Check if accurate
            is_accurate = (input_ids[i] in top_k_tokens[i-1])

            # Count normal prediction ( ignoring skip )
            num_predictions += 1
            num_accurate    += is_accurate

            # Now count only if the token is not supposed to be skipped
            if int(input_ids[i]) in skip_ids:
                continue
            num_skip_predictions += 1
            num_skip_accurate    += is_accurate

        # Keep track of most used tokens
        token_counts = np.zeros( token_dictionary_size )
        for id in input_ids:
            token_counts[id] += 1

        output = {
            'num_predictions': num_predictions,
            'num_accurate': num_accurate,
            'num_skip_predictions': num_skip_predictions,
            'num_skip_accurate': num_skip_accurate,
            'token_counts': token_counts,
            'input_ids': input_ids,
            'top_k_tokens': top_k_tokens,
        }

        return output

    def batch_decode( self, input_ids ):
        output_str = self.tokenizer.batch_decode( input_ids )
        return output_str

    def evaluate_dataset( self,
            dataset: datasets.Dataset,
            dataset_text_label: str = 'content',
            token_limit: Optional[int] = None,
            k: int = 1,
            count_tokens: bool = False,
            num_top_tokens: int = 50,
            start_index: int = 1,
            skip_eval: list = [],
            stopping_index: int = 1e6,
            ):
        """
        dataset: the Datset object to iterate over
        dataset_text_label: the label for the dataset text. eg: 'text'. 'content'
        token_limit: the maximum number of tokens to evaluate
        k: the number of top-k tokens predictions to evaluate
        count_tokens: whether to also count the most common tokens
        num_top_tokens: the number of most common tokens to evaluate
        start_index: the index of the first token to evaluate
        skip_eval: a list of token ids to skip when evaluating

        """

        # Initialize variables
        token_counts = None
        out = {
            "num_predictions": 0,
            "num_accurate": 0,
            "num_skip_predictions": 0,
            "num_skip_accurate": 0,
            "token_counts": None,
        }

        # Set up approx stopping index
        with tqdm(total=stopping_index) as pbar:

            # Loop over the dataset
            for data in dataset:
                # predict next token from text
                text = data[ dataset_text_label ]
                curr =  self.evaluate_top_k_performance( text, k=k,
                    start_index=start_index, skip_strings=skip_eval, limit=token_limit )

                # Record performance
                out["num_predictions"] += curr['num_predictions']
                out["num_accurate"]    += curr['num_accurate']
                out["num_skip_predictions"] += curr['num_skip_predictions']
                out["num_skip_accurate"]    += curr['num_skip_accurate']
                pbar.update( curr["num_skip_predictions"] )

                # Record token counts
                if count_tokens:
                    # Get current token counts
                    if token_counts is None:
                        token_counts = np.zeros_like( curr['token_counts'] )
                    token_counts += curr['token_counts']

                    # Save token counts
                    out["token_counts"] = token_counts

                # Print overall average prediction accuracy
                pred, acc = out["num_skip_predictions"], out["num_skip_accurate"]
                percentage = round( 100 * acc / pred, 1 )
                out_str = f'accuracy {percentage}% '

                # Print current prediction accuracy
                pred, acc = curr["num_skip_predictions"], curr["num_skip_accurate"]
                percentage = round( 100 * acc / (pred+1), 1 )
                out_str += f'(curr {percentage}%)'

                pbar.set_description( out_str )
                
                # Stop if limit is reached
                if out["num_skip_predictions"] > stopping_index:
                    break

            if count_tokens:
                topk = token_counts.argpartition(
                    -num_top_tokens )[-num_top_tokens:]
                topk = topk[np.argsort(token_counts[topk])][::-1]
                print( self.batch_decode( topk ) )
            
            return out
    
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