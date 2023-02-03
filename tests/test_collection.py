import torch
import numpy as np
import matplotlib.pyplot as plt
import pytest

from model import Model
from activations import get_midlayer_activations

class TestCollection:
    def test_ff_collections(self, verbose: bool = False):
        print( "# Running Test: test_ff_collection" )
        opt = Model("125m", limit=1000)
        n_samples = 1e3

        data_pile = get_midlayer_activations(opt, "pile", n_samples,
            calculate_ff=False, calculate_attn=False,
            collect_ff=True, use_ff_activation_function=False)
        data_code = get_midlayer_activations(opt, "code", n_samples,
            calculate_ff=False, calculate_attn=False,
            collect_ff=True, use_ff_activation_function=False)

        assert data_pile["raw"]["ff"].size()[1:] == torch.Size([12, 768*4])
        assert data_code["raw"]["ff"].size()[1:] == torch.Size([12, 768*4])

        ff_pile = data_pile["raw"]["ff"].permute( (1,2,0) )
        ff_code = data_code["raw"]["ff"].permute( (1,2,0) )

        assert ff_pile.size()[:-1] == torch.Size([12, 768*4])
        assert ff_code.size()[:-1] == torch.Size([12, 768*4])
        assert ff_pile.size()[-1] >= n_samples
        assert ff_code.size()[-1] >= n_samples

        # assert only ff was collected
        with pytest.raises(KeyError):
            data_pile["raw"]["attn"]
        with pytest.raises(KeyError):
            data_code["raw"]["attn"]

        if verbose:
            plt.figure()

            ff_pile_count, bins = np.histogram(ff_pile[0][0].numpy(), bins=100)
            ff_pile_mids = (bins[1:] + bins[:-1]) / 2
            plt.plot( ff_pile_mids, ff_pile_count, label="pile" )

            ff_code_count, bins = np.histogram(ff_code[0][0].numpy(), bins=100)
            ff_code_mids = (bins[1:] + bins[:-1]) / 2
            plt.plot( ff_code_mids, ff_code_count, label="code" )

            plt.legend()
            plt.show()

        # TODO: Add more tests here to make sure the data is correct

    def test_attn_collections(self, verbose: bool = False):
        print( "# Running Test: test_attn_collection" )
        opt = Model("125m", limit=1000)
        n_samples = 1e3

        data_pile = get_midlayer_activations(opt, "pile", n_samples,
            calculate_ff=False, calculate_attn=False, collect_attn=True)
        data_code = get_midlayer_activations(opt, "code", n_samples,
            calculate_ff=False, calculate_attn=False, collect_attn=True)

        assert data_pile["raw"]["attn"].size()[1:] == torch.Size([12, 12, 64])
        assert data_code["raw"]["attn"].size()[1:] == torch.Size([12, 12, 64])

        attn_pile = data_pile["raw"]["attn"].permute( (1,2,3,0) )
        attn_code = data_code["raw"]["attn"].permute( (1,2,3,0) )

        assert attn_pile.size()[:-1] == torch.Size([12, 12, 64])
        assert attn_code.size()[:-1] == torch.Size([12, 12, 64])
        assert attn_pile.size()[-1] >= n_samples
        assert attn_code.size()[-1] >= n_samples

        # assert only attention was collected
        with pytest.raises(KeyError):
            data_pile["raw"]["ff"]
        with pytest.raises(KeyError):
            data_code["raw"]["ff"]

        if verbose:
            plt.figure()

            attn_pile_count, bins = np.histogram(attn_pile[0][0].numpy(), bins=100)
            attn_pile_mids = (bins[1:] + bins[:-1]) / 2
            plt.plot( attn_pile_mids, attn_pile_count, label="pile" )

            attn_code_count, bins = np.histogram(attn_code[0][0].numpy(), bins=100)
            attn_code_mids = (bins[1:] + bins[:-1]) / 2
            plt.plot( attn_code_mids, attn_code_count, label="code" )

            plt.legend()
            plt.show()

        # TODO: Add more tests here to make sure the data is correct

    def test_does_not_collect(self):
        print( "# Running Test: test_does_not_collection" )
        opt = Model("125m", limit=1000)
        n_samples = 1e3

        with pytest.raises(ValueError):
            _data_pile = get_midlayer_activations(opt, "pile", n_samples,
                calculate_ff=False, calculate_attn=False)
