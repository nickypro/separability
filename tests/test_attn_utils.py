""" Test the attention operations. """
import torch

# pylint: disable=import-error
import pytest
from separability.model_repos import test_model_repos
from separability import Model

test_model_repos = ["facebook/opt-125m"]

def not_equal(t0, t1):
    return not torch.equal(t0, t1)

def is_zero(t):
    return torch.equal(t, torch.zeros_like(t))

class TestAttnCrossoverDelete:
    @pytest.mark.parametrize("model_repo", test_model_repos)
    def test_svd(self, model_repo):
        with torch.no_grad():
            layer = 0

            opt = Model(model_repo, use_accelerator=False, svd_attn=False)
            d_model, device = opt.cfg.d_model, opt.device
            attn = opt.layers[layer]["attn"]

            # Get example input
            in_0 = torch.randn([1, 3, d_model], device=device)
            mask = torch.tensor(
                [[[[1, 0, 0], [1, 1, 0], [1, 1, 1]]]],
                device=device, dtype=torch.bool
            )

            # Get example output
            out_0, _, (k_0, v_0) = attn(in_0, attention_mask=mask)
            print(out_0)


            # Do SVD stuff
            opt.svd_attention_layers()

            # Check that the output is the same
            out_1, _, (k_1, v_1) = attn(in_0, attention_mask=mask)
            assert torch.allclose(out_0, out_1, 1e-3, 1e-4)

            return

    @pytest.mark.parametrize("model_repo", test_model_repos)
    def test_deletion(self, model_repo):
        with torch.no_grad():
            layer, h_index, i_index = 0, 11, 63

            opt = Model(model_repo, use_accelerator=False, svd_attn=False)
            d_model, d_head, n_heads = opt.cfg.d_model, opt.cfg.d_head, opt.cfg.n_heads
            device = opt.device
            attn = opt.layers[layer]["attn"]

            # Get example input
            in_0 = torch.randn([1, 3, d_model], device=device)
            mask = torch.tensor(
                [[[[1, 0, 0], [1, 1, 0], [1, 1, 1]]]],
                device=device, dtype=torch.bool
            )
            out_biases = torch.stack([
                opt.layers[layer]["attn.b_O"] for _ in range(3)
            ])

            # Get example output
            out_0, _, (k_0, v_0) = attn(in_0, attention_mask=mask)
            # k_0 is [1, 12, 3, 64]
            v_0_mod = v_0.clone()
            v_0_mod[0, h_index, :, i_index] = 0


            # Delete index
            remove_indices = torch.zeros(
                (n_heads, d_head),
                dtype=torch.bool, device=device
            )
            remove_indices[h_index, i_index] = True
            opt.delete_attn_pre_out_layer(layer, remove_indices)

            # Test behaviour is correct
            out_1, _, (k_1, v_1) = attn(in_0, attention_mask=mask)
            assert not torch.equal(v_0, v_1)
            assert torch.equal(v_0_mod, v_1)


            # Delete ALL indices
            remove_indices = torch.ones_like(remove_indices)
            opt.delete_attn_pre_out_layer(layer, remove_indices)

            # Test behaviour is correct
            out_2, _, (k_2, v_2) = attn(in_0, attention_mask=mask)

            print(out_2)

            assert is_zero(attn.v_proj.weight)
            assert is_zero(attn.v_proj.bias)
            assert is_zero(v_2)
            assert torch.allclose(out_2, out_biases)

        return



