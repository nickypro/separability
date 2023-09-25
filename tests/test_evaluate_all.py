""" Test the evaluate_all function. """

# pylint: disable=import-error
import pytest
from separability.model_repos import test_model_repos
from separability import Model
from separability.eval import evaluate, evaluate_all

class TestEvaluate:
    @pytest.mark.parametrize("model_repo", test_model_repos)
    def test_evaluate(self, model_repo):
        print("# Running test: test_evaluate")
        opt = Model(model_repo, limit=1000, dtype="fp32")
        opt.show_details()

        eval_sample_size = 1e4

        # We run the a first time, with a small subset of data
        # Note that for most datasets, we use the "test" strand.
        # In 'code', we use a non-intersecting subset of the "train" strand
        # since there is no "test" strand.
        data_1 = evaluate( opt, 'code', eval_sample_size,
            verbose=True, dataset_tokens_to_skip=eval_sample_size )

        # We check that the data is correct
        keys = data_1.misc.keys()
        expected_keys = [
            'num_predictions',
            'num_skip_predictions',
            'num_accurate',
            'num_skip_accurate',
            'num_topk_accurate',
            'num_topk_skip_accurate',
            'token_counts',
            'percent', # dict
            'loss',
            'log_loss',
#            'loss_data', # dict
        ]
        print(keys)
        assert len(keys) == len(expected_keys)
        for key in keys:
            assert key in expected_keys

        # verify the percent sub dict
        percent_keys = data_1.percent.keys()
        expected_percent_keys = [ "base", "skip", "topk", "topk_skip" ]
        assert len(percent_keys) == 4
        for key in percent_keys:
            assert key in expected_percent_keys

        # verify the loss_data sub dict
        loss_data_keys = data_1.loss_data.keys()
        expected_loss_data_keys = [ "loss", "log_loss" ]
        assert len(loss_data_keys) == 2
        for key in loss_data_keys:
            assert key in expected_loss_data_keys

        # We run the a second time, with a different subset of data
        data_2 = evaluate( opt, 'code', eval_sample_size,
            verbose=True, dataset_tokens_to_skip=2*eval_sample_size )


        # We check that the output is different, since the input was different,
        # and that the output is similar, since the model is the same, and the text
        # is similar (it's the same dataset, just a different subset)
        for key in keys:
            if key == 'percent':
                continue
            if key == 'loss_data':
                continue
            if key == 'token_counts':
                assert data_1.misc[key] is None
                assert data_2.misc[key] is None
                continue
            print( key, data_1.misc[key], data_2.misc[key] )
            assert data_1.misc[key] != data_2.misc[key]
            assert data_1.misc[key] < data_2.misc[key] * 1.5
            assert data_1.misc[key] > data_2.misc[key] * 0.5
        print()

    @pytest.mark.parametrize("model_repo", test_model_repos)
    def test_evaluate_all(self, model_repo):
        print("# Running test: test_evaluate_all")
        opt = Model(model_repo, limit=1000, dtype="fp32")
        opt.show_details()

        data = evaluate_all( opt, 1e3 )

        print( data.keys() )
