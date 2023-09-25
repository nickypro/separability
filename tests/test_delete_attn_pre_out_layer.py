""" Test the delete_attn_pre_out_layer function. """

import copy

from torch import Tensor
import torch
import numpy as np

# pylint: disable=import-error
import pytest
from separability.model_repos import test_model_repos
from separability import Model

class TestDeleteAttnPreOutLayer:
    @pytest.mark.parametrize("model_repo", test_model_repos)
    @pytest.mark.parametrize("mask_fn", ["delete", "step"])
    def test_delete_attn_pre_out_layer(self, model_repo, mask_fn):
        # Test deleting the output of the attention layers
        print("# Running Test: test_delete_attn_pre_out_layer")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        opt = Model(model_repo, limit=1000, dtype="fp32", mask_fn=mask_fn)

        with torch.no_grad():
            n_heads, d_head, d_model = \
                opt.cfg.n_heads, opt.cfg.d_head, opt.cfg.d_model

            # Define vectors for testing
            #vec_in: Tensor  = torch.tensor(
            #    np.random.random(d_model), dtype=torch.float32
            #).to( device )
            vec_mid: Tensor = torch.tensor(
                np.random.random((n_heads, d_head)), dtype=torch.float32
            ).to( device )

            # Define a vector that is changed at certain indices
            vec_mid_d0 : Tensor = copy.deepcopy( vec_mid )
            vec_mid_d1 : Tensor = copy.deepcopy( vec_mid )
            removed_indices   = [(0, 0), (0, 10), (1, 10), (5, 31)]
            unremoved_indices = [(0, 1), (1, 0),  (5, 30)]

            removal_tensor = torch.zeros_like(vec_mid_d0, dtype=torch.bool)
            keep_tensor    = torch.ones_like(vec_mid_d1, dtype=torch.bool)
            for (i_head, i_pos) in removed_indices:
                vec_mid_d0[i_head][i_pos] = 100
                removal_tensor[i_head][i_pos] = True
                keep_tensor[i_head][i_pos] = False

            for i_head, i_pos in unremoved_indices:
                vec_mid_d1[i_head][i_pos] = 100

            # Start tests
            for add_mean in [True, False]:
                print(f"## Testing outward weight removals - add_mean={add_mean}")
                opt = Model(model_repo, dtype="fp32", mask_fn=mask_fn,
                    model_device=device, use_accelerator=False)
                LAYER = 0

                out_proj = opt.layers[LAYER]["attn.out_proj"]
                out_proj_orig_weight = out_proj.weight.detach().clone()

                # Test that the old outputs do care about changes to all indices
                old_vec_out = out_proj(vec_mid.flatten())
                old_vec_out_d0 = out_proj(vec_mid_d0.flatten())
                print( '- vec      :', old_vec_out[:5] )
                print( '- vec+ (1) :', old_vec_out_d0[:5] )
                assert not torch.equal( old_vec_out, old_vec_out_d0 )

                # Run the deletion
                print('deleting indices:', removed_indices,
                        '' if add_mean else 'NOT', 'adding mean activation')
                if add_mean:
                    opt.delete_attn_pre_out_layer( LAYER, removal_tensor, vec_mid )
                else:
                    opt.delete_attn_pre_out_layer( LAYER, removal_tensor )

                out_proj = opt.layers[LAYER]["attn.out_proj"]

                # Test that new outputs do not care about changes to deleted indices
                # but still care about changes to undeleted indices.
                new_vec_out = out_proj(vec_mid.flatten())
                new_vec_out_d0 = out_proj(vec_mid_d0.flatten())
                new_vec_out_d1 = out_proj(vec_mid_d1.flatten())
                print( '- vec      :', new_vec_out[:5] )
                print( '- vec+ (1) :', new_vec_out_d0[:5] )
                print( '- vec+ (2) :', new_vec_out_d1[:5] )
                assert torch.equal( new_vec_out, new_vec_out_d0 )
                assert not torch.equal( new_vec_out_d0, new_vec_out_d1 )

                if mask_fn == "delete":
                    assert not torch.equal( out_proj.weight, out_proj_orig_weight )
                if mask_fn == "step":
                    assert torch.equal( out_proj.weight, out_proj_orig_weight )

        return

    @pytest.mark.parametrize("model_repo", test_model_repos)
    @pytest.mark.parametrize("mask_fn", ["delete", "step"])
    def test_delete_attn_value_layer(self, model_repo, mask_fn):
        print("# Running Test: test_delete_attn_value_layer")

        # Define model and parameters
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        LAYER = 0

        opt = Model(model_repo, dtype="fp32", mask_fn=mask_fn,
            model_device=device, use_accelerator=False)
        v_proj = opt.layers[LAYER]["attn.v_proj"]
        v_proj_orig_weight = v_proj.weight.detach().clone()

        n_heads, d_head, d_model = \
            opt.cfg.n_heads, opt.cfg.d_head, opt.cfg.d_model

        # Start test
        with torch.no_grad():
            # Define vec in
            vec_in: Tensor = torch.tensor(
                np.random.random(d_model), dtype=torch.float32
            ).to( device )

            # Choose indices (head, pos) to delete
            removed_indices = [(0, 0), (0, 10), (1, 10), (5, 31)]
            removal_tensor  = \
                torch.zeros((n_heads, d_head), dtype=torch.bool, device=device)
            keep_tensor     = \
                torch.ones((n_heads, d_head), dtype=torch.bool, device=device)
            for (i_head, i_pos) in removed_indices:
                removal_tensor[i_head][i_pos] = True
                keep_tensor[i_head][i_pos]    = False


            # Get output vector before deletion
            old_vec_mid = v_proj(vec_in).reshape((n_heads, d_head))
            print( '- old vec  :', old_vec_mid[:5] )

            # Run the deletion
            print('deleting indices:', removed_indices)
            opt.delete_attn_pre_out_layer( LAYER, removal_tensor )
            v_proj = opt.layers[LAYER]["attn.v_proj"]

            # Get output vector after deletion
            new_vec_mid = v_proj(vec_in).reshape((n_heads, d_head))
            print( '- new vec  :', new_vec_mid[:5] )

            # Test that new outputs do not care about changes to deleted indices
            # Check weight changes
            if mask_fn == "delete":
                # Check that the weights have changed
                assert not torch.equal( v_proj_orig_weight, v_proj.weight )
                # Check that the outputs have changed
                assert torch.equal( old_vec_mid*keep_tensor, new_vec_mid )
                assert not torch.equal( old_vec_mid, new_vec_mid )

            if mask_fn == "step":
                # mask should not affect weights
                assert torch.equal( v_proj_orig_weight, v_proj.weight )
                # TODO: support value matrix output mask
                # Check that the outputs have changed
                #assert torch.equal( old_vec_mid*keep_tensor, new_vec_mid )
                #assert not torch.equal( old_vec_mid, new_vec_mid )

            print("Test Passed")

            return
