""" Test the delete_attn_pre_out_layer function. """

import argparse
import copy

from torch import Tensor
import torch
import numpy as np

from model import Model

def test_delete_attn_pre_out_layer( verbose: bool = False ):
    print("# Running Test: test_delete_attn_pre_out_layer")
    with torch.no_grad():
        d_model = 768
        model_size = '125m'

        # Define vectors for testing
        vec : Tensor = torch.tensor( np.random.random(d_model), dtype=torch.float32 )

        # Define a vector that is changed at certain indices
        vec_plus_1 : Tensor = copy.deepcopy( vec )
        vec_plus_2 : Tensor = copy.deepcopy( vec )
        removed_indices   = [0, 10, 100]
        unremoved_indices = [1, 3, 69]

        removal_tensor = torch.zeros_like(vec_plus_1)
        for index in removed_indices:
            vec_plus_1[index] = 100
            removal_tensor[index] = True

        for i in unremoved_indices:
            vec_plus_2[i] = 100

        for add_mean in [True, False]:
            opt = Model(model_size)
            LAYER = 0

            out_proj = opt.model.decoder.layers[LAYER].self_attn.out_proj

            # Test that the old outputs do care about changes to all indices
            old_vec_out = out_proj(vec)
            old_vec_plus_out = out_proj(vec_plus_1)
            if verbose:
                print( '- vec      :', old_vec_out[:5] )
                print( '- vec+ (1) :', old_vec_plus_out[:5] )
            assert not torch.equal( old_vec_out, old_vec_plus_out )

            if verbose:
                print('deleting indices:', removed_indices,
                    '' if add_mean else 'NOT', 'adding mean activation')
            if add_mean:
                opt.delete_attn_pre_out_layer( LAYER, removal_tensor, vec )
            else:
                opt.delete_attn_pre_out_layer( LAYER, removal_tensor )

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

    test_delete_attn_pre_out_layer(args.verbose)
