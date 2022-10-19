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

most_common_code_tokens = [' ', '\n', '.', '_', ',', '#', '(', ' =', ' import', 'from', ' the', ':', ')', '\n\n', 'import', " '", '/', '-', '):', '\t', "',", ' "', ' self', '=', ' of', "'", '__', ' (', 'self', ' in', ' License', '</s>', ' is', '0', ' for', ' to', 's', '1', '2', ' a', ' as', '\r', ' -', ' and', ' def', ' #', 'x', '()', "('", '\\']

class Model():
    def __init__( self, model_size : str  = "125m" ):
        """
        OPT Model with functions for extracting activations.
        model_size : 125m, 350m, 1.3b, 2.7b, 6.7b, 13b, 30b, 66b, 175b
        """
        if model_size not in model_sizes:
            raise ValueError( "model_size must be one of the following: " + str(model_sizes) )

        repo = f"facebook/opt-{model_size}"

        """
        configuration = OPTConfig()
        model = OPTModel(configuration)
        configuration = model.config
        """

        self.tokenizer = GPT2Tokenizer.from_pretrained( repo )
        self.predictor = OPTForCausalLM.from_pretrained( repo )
        self.model = self.predictor.model

        self.activations = {}

        self.register_activations()

        # Indices of outputs for reference
        self.layer_index     = -3
        self.token_index     = -2
        self.dimension_index = -1

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

    def get_inputs_embeds( self,
                text : str,
                verbose : bool = False,
                limit : Optional[int] = None
            ):
        inputs = self.tokenizer(text, return_tensors="pt")

        input_ids = inputs.input_ids
        
        if verbose >= 2:
            print("inputs:")
            print( inputs.input_ids.size() )
        
        if limit:
            prev_size = input_ids.size()
            input_ids = input_ids[0][:limit].reshape(1, -1)
            new_size = input_ids.size()

            if verbose == 1:
                print("trimmed from", list(prev_size), "to", list(new_size) )

            if verbose >= 2:
                print("trimmed inputs:")
                print( new_size )

        inputs_embeds = self.model.decoder.embed_tokens( input_ids )
        
        if verbose >= 2:
            print( inputs_embeds.size() )

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
                text : Optional[str] = None,
                inputs_embeds : Tensor = None,
                verbose : bool = False,
                limit : Optional[int] = None,
                **kwargs
            ):
        if inputs_embeds is None and text is None:
            raise ValueError( "must provide either inputs_embeds or text" )

        if text:
            inputs_embeds = self.get_inputs_embeds( text, verbose, limit )

        # run the model
        outputs = self.model( inputs_embeds=inputs_embeds, output_hidden_states=True, **kwargs )

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
        output = outputs.last_hidden_state[0].detach()

        return input, attention_out, ff_out, output

    def get_residual_stream( self,
                text : Optional[str] = None,
                inputs_embeds : Tensor = None,
                verbose : bool = False,
                limit : Optional[int] = None,
                **kwargs
            ) -> Tensor:

        input, attention_out, ff_out, output = \
            self.get_text_activations( text, inputs_embeds, verbose, limit, **kwargs )
        
        assert len(attention_out) == len(ff_out)
        L = len(attention_out)

        adjustments = [0]*(2*L)
        adjustments[0::2] = attention_out
        adjustments[1::2] = ff_out


        residual_stream = []
        residual_stream.append( input )

        for delta in adjustments:
            residual_stream.append( residual_stream[-1] + delta )
        
        residual_stream.append( output )

        return torch.stack( residual_stream )

    def predict( self,
                text : str,
                num : int = 10,
                limit : Optional[int] = None
            ):

        inputs = self.tokenizer( text, return_tensors="pt" )
        input_ids = inputs.input_ids

        if limit:
            input_ids = input_ids[0][:limit].reshape(1, -1)
        
        generate_ids = self.predictor.generate( input_ids, max_length=len(input_ids[0])+num )
        
        before = self.tokenizer.batch_decode( input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        after  = self.tokenizer.batch_decode( generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        after = after[len(before):]
        return before, after

    def get_nth_tokens( self, output: Tensor, n: int = 16 ):

        L = output.size()[self.token_index]
        indices = torch.tensor( list(range( n-1, L, n )) )

        return torch.index_select( output, self.token_index, indices )

    def get_ids( self, text:str, limit:Optional[int]=None ):
        input_ids = self.tokenizer( text, return_tensors='pt').input_ids
        if not limit is None:
            input_ids = torch.stack([ input_ids[0][:limit] ])
        return input_ids

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
                min_index: int = 1,
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
        for i in range( min_index, len(input_ids) ):
            if int(input_ids[i]) in skip_ids:
                continue
            num_predictions += 1
            if input_ids[i] in top_k_tokens[i-1]:
                num_accurate += 1

        # Keep track of most used tokens
        token_counts = np.zeros( token_dictionary_size )
        for id in input_ids:
            token_counts[id] += 1

        output = {
            'num_predictions': num_predictions,
            'num_accurate': num_accurate,
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
            token_limit: Optional[int] = None,
            k: int = 1,
            count_tokens: bool = False,
            num_top_tokens: int = 50,
            skip_eval: list = [],
            verbose: bool = True,
            dataset_text_label: str = 'content'
            ):

        # Initialize variables
        token_counts = None
        out = {
            "num_predictions": 0,
            "num_accurate": 0,
            "token_counts": None,
        }

        # Loop over the dataset
        for data in dataset:
            # predict next token from text
            text = data[ dataset_text_label ]
            output =  self.evaluate_top_k_performance( text, k=k,
                min_index=10, skip_strings=skip_eval, limit=token_limit )

            # Record performance
            out["num_predictions"] += output['num_predictions']
            out["num_accurate"]    += output['num_accurate']

            # Record token counts
            if count_tokens:
                # Get current token counts
                if token_counts is None:
                    token_counts = np.zeros_like( output['token_counts'] )
                token_counts += output['token_counts']

                # Save token counts
                out["token_counts"] = token_counts
                
                # Get top token counts if verbose
                if verbose:
                    topk = token_counts.argpartition(
                        -num_top_tokens )[-num_top_tokens:]
                    topk = topk[np.argsort(token_counts[topk])][::-1]
                    print( self.batch_decode( topk ) )

            # Print the current prediction accuracy
            if verbose:
                # Print overall average prediction accuracy
                pred, acc = out["num_predictions"], out["num_accurate"]
                percentage = round( 100 * acc / pred, 1 )
                out_str = f'accuracy {percentage}% {acc}/{pred}  '

                # Print current prediction accuracy
                pred, acc = output["num_predictions"], output["num_accurate"]
                percentage = round( 100 * acc / (pred+1), 1 )
                out_str += f'( curr {percentage}% )'

                print( out_str )

        return out