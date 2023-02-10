""" Test the get_ff_keys and delete_ff_keys functions. """

import copy
from torch import Tensor
import torch
import numpy as np

# pylint: disable=import-error
import pytest
from seperability.model_names import test_model_names
from seperability import Model

class TestDeleteFFKeys:
    @pytest.mark.parametrize("model_name", test_model_names)
    def test_ff_key_counting(self, model_name):
        print("# Running Test: test_ff_key_counting")
        n_layers = 12
        d_ff     = 3072 # This is the value for 125m, 4*768

        # Initialize model
        opt = Model(model_name, limit=1000)

        # Run text
        text = "for ( var i = 0; i < 10; i++ ) { console.log(i); }"
        input_ids = opt.get_ids( text )
        n_tokens  = input_ids.size()[-1]

        # Make a tensor of the expected_size
        expected_size = torch.Size([ n_layers, n_tokens, d_ff ])

        # Run the model
        with torch.no_grad():
            ff_keys = opt.get_ff_key_activations(input_ids=input_ids)

        # Test that result is as desired
        assert len(ff_keys) == n_layers
        assert ff_keys.size() == expected_size

        print( "Text size:", ff_keys.size() )
        print( "Expected :", expected_size )

    @pytest.mark.parametrize("model_name", test_model_names)
    def test_delete_ff_keys(self, model_name):
        print("# Running Test: test_delete_ff_keys")

        # Pre-test initialization
        # Define model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        opt = Model(model_name, model_device=device, use_accelerator=False)

        # Define input vectors for testing
        removed_indices   = [ 0, 10, 100 ]
        in_vec : Tensor = torch.tensor(
            np.random.random(opt.d_model), dtype=torch.float32
        ).to( device ).detach()

        # define functions for testing
        # Do not use activation function here for better testing
        def in_to_mid(in_vec, layer):
            return opt.calculate_ff_keys_layer(
                in_vec, layer, use_activation_function=False)

        # use activation function here because not used earlier
        def mid_to_out(mid_vec, layer):
            u = opt.model.decoder.layers[layer]
            x = u.activation_fn(mid_vec)
            return u.fc2(x)

        # Calculate mid layer vectors for testing
        mid_vecs = []
        mid_vecs_removed = []
        for layer in range(opt.n_layers):
            mid_vec = in_to_mid( in_vec, layer )
            mid_vecs.append( mid_vec )
            mid_vecs_removed.append( mid_vec.clone() )
            mid_vecs_removed[-1][removed_indices] = 0.0

        mid_vecs = torch.stack( mid_vecs )
        mid_vecs_removed = torch.stack( mid_vecs_removed )

        # Calculate out layer vectors for testing
        out_vecs = []
        out_vecs_removed = []
        for layer in range(opt.n_layers):
            out_vecs.append( mid_to_out( mid_vecs[layer], layer ) )
            out_vecs_removed.append( mid_to_out( mid_vecs_removed[layer], layer ) )

        out_vecs = torch.stack( out_vecs )
        out_vecs_removed = torch.stack( out_vecs_removed )

        # Define a vector that is changed at certain indices
        removal_tensor = torch.zeros_like(mid_vecs, dtype=torch.bool)
        for layer in range(opt.n_layers):
            removal_tensor[layer][removed_indices] = True

        # Here the test starts
        # Pre-test to make sure that outputs are different on each layer
        print('running post-deletion validation')
        for layer in range(opt.n_layers):
            print('layer ', layer)
            mid_vec_layer = in_to_mid( in_vec, layer )
            assert torch.equal( mid_vec_layer, mid_vecs[layer] )
            assert not torch.equal( mid_vec_layer, mid_vecs_removed[layer] )

            out_vec_layer = opt.calculate_ff_out_layer( in_vec, layer )
            assert torch.equal( out_vec_layer, out_vecs[layer] )

        # Run deletions on the layers
        opt.delete_ff_keys(removal_tensor)

        # Post-test to make sure deletions work as expected
        print('running post-deletion validation')
        for layer in range(opt.n_layers):
            print('layer', layer)
            mid_vec_layer = in_to_mid( in_vec, layer )
            assert not torch.equal( mid_vec_layer, mid_vecs[layer] )
            assert torch.equal( mid_vec_layer, mid_vecs_removed[layer] )

            out_vec_layer = opt.calculate_ff_out_layer( in_vec, layer )
            assert torch.equal( out_vec_layer, out_vecs_removed[layer] )

        # Extra sanity check: make sure that weights are zero where deleted
        for layer in range(opt.n_layers):
            w = opt.model.decoder.layers[layer].fc1.weight
            assert torch.equal(
                removal_tensor[layer],
                ( torch.sum( w, dim=0 ) == 0.0 )
            )
