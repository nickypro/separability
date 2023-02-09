""" Test the delete_attn_pre_out_layer function. """

import copy

from torch import Tensor
import torch
import numpy as np
import pytest

# pylint: disable=import-error
from seperability.test_model_names import model_names
from seperability import Model

class TestDeleteAttnPreOutLayer:
    @pytest.mark.parametrize("model_name", model_names)
    def test_delete_attn_pre_out_layer(self, model_name):
        print("# Running Test: test_delete_attn_pre_out_layer")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        opt = Model(model_name, limit=1000)

        with torch.no_grad():
            d_model = opt.d_model

            # Define vectors for testing
            vec : Tensor = torch.tensor(
                np.random.random(d_model), dtype=torch.float32
            ).to( device )

            # Define a vector that is changed at certain indices
            vec_plus_1 : Tensor = copy.deepcopy( vec )
            vec_plus_2 : Tensor = copy.deepcopy( vec )
            removed_indices   = [0, 10, 100]
            unremoved_indices = [1, 3, 69]

            removal_tensor = torch.zeros_like(vec_plus_1, dtype=torch.bool)
            keep_tensor    = torch.ones_like(vec_plus_1, dtype=torch.bool)
            for index in removed_indices:
                vec_plus_1[index] = 100
                removal_tensor[index] = True
                keep_tensor[index] = False

            for i in unremoved_indices:
                vec_plus_2[i] = 100

            # Start tests
            for add_mean in [True, False]:
                print(f"## Testing outward weight removals - add_mean={add_mean}")
                opt = Model(model_name, model_device=device, use_accelerator=False)
                LAYER = 0

                out_proj = opt.model.decoder.layers[LAYER].self_attn.out_proj

                # Test that the old outputs do care about changes to all indices
                old_vec_out = out_proj(vec)
                old_vec_plus_out = out_proj(vec_plus_1)
                print( '- vec      :', old_vec_out[:5] )
                print( '- vec+ (1) :', old_vec_plus_out[:5] )
                assert not torch.equal( old_vec_out, old_vec_plus_out )

                # Run the deletion
                print('deleting indices:', removed_indices,
                        '' if add_mean else 'NOT', 'adding mean activation')
                if add_mean:
                    opt.delete_attn_pre_out_layer( LAYER, removal_tensor, vec )
                else:
                    opt.delete_attn_pre_out_layer( LAYER, removal_tensor )

                out_proj = opt.model.decoder.layers[LAYER].self_attn.out_proj

                # Test that new outputs do not care about changes to deleted indices
                # but still care about changes to undeleted indices.
                new_vec_out = out_proj(vec)
                new_vec_plus_out_1 = out_proj(vec_plus_1)
                new_vec_plus_out_2 = out_proj(vec_plus_2)
                print( '- vec      :', new_vec_out[:5] )
                print( '- vec+ (1) :', new_vec_plus_out_1[:5] )
                print( '- vec+ (2) :', new_vec_plus_out_2[:5] )
                assert torch.equal( new_vec_out, new_vec_plus_out_1 )
                assert not torch.equal( new_vec_plus_out_1, new_vec_plus_out_2 )

            # Also test inward weight removals
            print("## Testing inward weight removals")
            opt = Model(model_name, model_device=device, use_accelerator=False)
            v_proj = opt.model.decoder.layers[LAYER].self_attn.v_proj

            # Get output vector before deletion
            old_vec_out = v_proj(vec)
            print( '- old vec  :', old_vec_out[:5] )

            # Run the deletion
            print('deleting indices:', removed_indices)
            opt.delete_attn_pre_out_layer( LAYER, removal_tensor )

            v_proj = opt.model.decoder.layers[LAYER].self_attn.v_proj

            # Test that the new outputs do not care about changes to deleted indices
            new_vec_out = v_proj(vec)
            print( '- new vec  :', new_vec_out[:5] )

            assert torch.equal( old_vec_out*keep_tensor, new_vec_out )

            print("Test Passed")
