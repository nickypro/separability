import argparse

from model import Model
import torch
import numpy as np
import copy
from torch import Tensor

def test_ff_key_counting( verbose = False ):
  with torch.no_grad():
    model_size = '125m'
    n_layers = 12
    d_model  = 768
    d_ff     = 3072

    # Initialize Model
    opt = Model(model_size, limit=1000)

    # Run text
    text = "for ( var i = 0; i < 10; i++ ) { console.log(i); }"
    input_ids = opt.get_ids( text )
    n_tokens  = input_ids.size()[-1]

    # Make a tensor of the expected_size
    expected_size = torch.Size([ n_layers, n_tokens, d_ff ]) 

    # Run the model
    ff_keys = opt.get_ff_key_activations(input_ids=input_ids)

    # Test that result is as desired
    assert len(ff_keys) == n_layers
    assert ff_keys.size() == expected_size

    if (verbose):
        print( "Text size:", ff_keys.size() )
        print( "Expected :", expected_size )

    return True

def test_delete_ff_keys( verbose: bool = False ):
  with torch.no_grad():
    model_size = '125m'
    n_layers = 12
    d_model = 768

    # Define vectors for testing
    vec : Tensor = torch.tensor( np.random.random(d_model), dtype=torch.float32 )

    # Define a vector that is changed at certain indices
    vec_plus_1 : Tensor = copy.deepcopy( vec )
    vec_plus_2 : Tensor = copy.deepcopy( vec )
    removed_indices   = [ 0, 10, 100 ]
    unremoved_indices = [ 1, 3, 69 ]

    removal_tensor = torch.zeros_like(vec_plus_1)
    for index in removed_indices:
        vec_plus_1[index] = 100
        removal_tensor[index] = True

    for i in unremoved_indices:
        vec_plus_2[i] = 100

    removal_indices = []    
    for _ in range(n_layers):
        removal_indices.append( torch.zeros(d_model) )
    removal_indices = torch.stack(removal_indices)
    
    opt = Model(model_size)

    # Pre-test to make sure that outputs are different on each layer
    for layer in range(opt.n_layers):
        out1 = opt.calculate_ff_out_layer( vec, layer )
        out2 = opt.calculate_ff_out_layer( vec_plus_1, layer )
        out3 = opt.calculate_ff_out_layer( vec_plus_2, layer )
        assert not torch.equal( out1, out2 )
        assert not torch.equal( out2, out3 )
        assert not torch.equal( out3, out1 )

    return

    # Change layer 0 but not other layers
    LAYER = 0
    out_proj = opt.model.decoder.layers[LAYER].self_attn.out_proj

    # Test that the old outputs do care about changes to all indices
    old_vec_out = out_proj(vec)
    old_vec_plus_out = out_proj(vec_plus_1)
    if verbose:
        print( '- vec      :', old_vec_out[:5] )
        print( '- vec+ (1) :', old_vec_plus_out[:5] )
    assert not torch.equal( old_vec_out, old_vec_plus_out )

    opt.delete_ff_keys( LAYER, removal_tensor )
    
    out_proj = opt.model.decoder.layers[LAYER].self_attn.out_proj

    # Test that the new outputs do not care about changes to deleted indices
    # but still care about changes to undeleted indices.
    new_vec_out = out_proj(vec)
    new_vec_plus_out_1 = out_proj(vec_plus_1)
    new_vec_plus_out_2 = out_proj(vec_plus_2)
    if verbose:
        print( '- vec      :', new_vec_out[:5] )
        print( '- vec+ (1) :', new_vec_plus_out_1[:5] )
        print( '- vec+ (2) :', new_vec_plus_out_2[:5] )
    assert torch.equal( new_vec_out, new_vec_plus_out_1 )
    assert not torch.equal( new_vec_plus_out_1, new_vec_plus_out_2 )

    print("Test Passed")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action='store_true', default=False)
    args = parser.parse_args()

    test_ff_key_counting(args.verbose)
    test_delete_ff_keys(args.verbose)