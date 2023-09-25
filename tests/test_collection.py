import torch
import numpy as np
import matplotlib.pyplot as plt

# pylint: disable=import-error, pointless-statement
import pytest
from separability.model_repos import test_model_repos
from separability import Model
from separability.activations import get_midlayer_activations

class TestCollection:
    @pytest.mark.parametrize("model_repo", test_model_repos)
    def test_ff_collections(self, model_repo):
        print( "# Running Test: test_ff_collection" )
        opt = Model(model_repo, limit=1000, dtype="fp32")
        n_samples = 1e3
        n_layers, d_mlp = opt.cfg.n_layers, opt.cfg.d_mlp

        data_pile = get_midlayer_activations(opt, "pile", n_samples,
            calculate_ff=False, calculate_attn=False,
            collect_ff=True, use_ff_activation_function=False)
        data_code = get_midlayer_activations(opt, "code", n_samples,
            calculate_ff=False, calculate_attn=False,
            collect_ff=True, use_ff_activation_function=False)

        assert data_pile.raw["ff"].size()[1:] == torch.Size([n_layers, d_mlp])
        assert data_code.raw["ff"].size()[1:] == torch.Size([n_layers, d_mlp])

        ff_pile = data_pile.raw["ff"].permute( (1,2,0) )
        ff_code = data_code.raw["ff"].permute( (1,2,0) )

        assert ff_pile.size()[:-1] == torch.Size([n_layers, d_mlp])
        assert ff_code.size()[:-1] == torch.Size([n_layers, d_mlp])
        assert ff_pile.size()[-1] >= n_samples
        assert ff_code.size()[-1] >= n_samples

        # assert only ff was collected
        with pytest.raises(KeyError):
            data_pile["raw"]["attn"]
        with pytest.raises(KeyError):
            data_code["raw"]["attn"]

        # TODO: Add more tests here to make sure the data is correct

    @pytest.mark.parametrize("model_repo", test_model_repos)
    def test_attn_collections(self, model_repo):
        print( "# Running Test: test_attn_collection" )
        opt = Model(model_repo, limit=1000, dtype="fp32")
        n_samples = 1e3
        n_layers, n_heads, d_head = \
            opt.cfg.n_layers, opt.cfg.n_heads, opt.cfg.d_head
        shape = torch.Size([n_layers, n_heads, d_head])

        data_pile = get_midlayer_activations(opt, "pile", n_samples,
            calculate_ff=False, calculate_attn=False, collect_attn=True)
        data_code = get_midlayer_activations(opt, "code", n_samples,
            calculate_ff=False, calculate_attn=False, collect_attn=True)

        assert data_pile.raw["attn"].size()[1:] == shape
        assert data_code.raw["attn"].size()[1:] == shape

        attn_pile = data_pile.raw["attn"].permute( (1,2,3,0) )
        attn_code = data_code.raw["attn"].permute( (1,2,3,0) )

        assert attn_pile.size()[:-1] == shape
        assert attn_code.size()[:-1] == shape
        assert attn_pile.size()[-1] >= n_samples
        assert attn_code.size()[-1] >= n_samples

        # assert only attention was collected
        with pytest.raises(KeyError):
            data_pile.raw["ff"]
        with pytest.raises(KeyError):
            data_code.raw["ff"]

        # TODO: Add more tests here to make sure the data is correct

    @pytest.mark.parametrize("model_repo", test_model_repos)
    @pytest.mark.parametrize("mask_fn", ["delete", "step"])
    def test_masked_collection(self, model_repo, mask_fn):
        print( "# Running Test: test_masked_collection" )
        opt = Model(model_repo, limit=1000, dtype="fp32", mask_fn=mask_fn)
        n_samples = 1e3

        data_pile = get_midlayer_activations(opt, "pile", n_samples)

        attn_removals = torch.zeros_like(data_pile.attn.orig.mean.shape)
        attn_removals[:, :10] = 1
        mlp_removals  = torch.zeros_like(data_pile.ff.orig.mean.shape)
        mlp_removals[:, :10] = 1

        opt.delete_attn_pre_out(attn_removals)
        opt.delete_ff_keys(mlp_removals)

        # TODO: add tests here

    @pytest.mark.parametrize("model_repo", test_model_repos)
    def test_does_not_collect(self, model_repo):
        print( "# Running Test: test_does_not_collection" )
        opt = Model(model_repo, limit=1000)
        n_samples = 1e3

        with pytest.raises(ValueError):
            _data_pile = get_midlayer_activations(opt, "pile", n_samples,
                calculate_ff=False, calculate_attn=False)
