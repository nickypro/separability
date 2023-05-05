import warnings
import torch

# pylint: disable=import-error
import pytest
from separability.model_repos import test_model_repos
from separability import Model
from separability.activations import prune_and_evaluate, evaluate_all

class TestPruneAndEvaluate:
    @pytest.mark.parametrize("model_repo", test_model_repos)
    def test_prune_and_evaluate_accelerate(self, model_repo):
        print( "# Running Test: test_prune_and_evaluate_accelerate" )

        opt = Model(model_repo, limit=1000, use_accelerator=True)

        if torch.cuda.device_count() <= 1:
            warnings.warn( "Multi-gpu not available", category=UserWarning )
            return

        data = prune_and_evaluate(opt, 0.05, 0.05, 0.001, 1e4, 1e4)

        pile_loss = data.loss_data['pile']['loss']
        code_loss = data.loss_data['code']['loss']
        assert pile_loss > 1
        assert code_loss > 1

    @pytest.mark.parametrize("model_repo", test_model_repos)
    def test_prune_and_evaluate(self, model_repo):
        opt = Model(model_repo, limit=1000, use_accelerator=False)
        data = prune_and_evaluate(opt, 0.1, 0.1, 0.001, 1e4, 1e4,
                                  do_attn_mean_offset=False)

        pile_loss = data.loss_data['pile']['loss']
        code_loss = data.loss_data['code']['loss']
        assert pile_loss > 1
        assert code_loss > 1

    @pytest.mark.parametrize("model_repo", test_model_repos)
    def test_prune_and_evaluate_mean_offset(self, model_repo):
        # TODO: Fix mean offset
        return
        opt = Model(model_repo, limit=1000, use_accelerator=False)
        data = prune_and_evaluate(opt, 0.1, 0.1, 0.001, 1e4, 1e4,
                                  do_attn_mean_offset=True)

        pile_loss = data.loss_data['pile']['loss']
        code_loss = data.loss_data['code']['loss']
        assert pile_loss > 1
        assert code_loss > 1


    @pytest.mark.parametrize("model_repo", test_model_repos)
    def test_prune_attn_values_and_evaluate(self, model_repo):
        opt = Model(model_repo, limit=1000, use_accelerator=False)
        data = prune_and_evaluate(opt, 0.1, 0.1, 0.001, 1e4, 1e4,
                do_attn_mean_offset=False, attn_mode="values")

        pile_loss = data.loss_data['pile']['loss']
        code_loss = data.loss_data['code']['loss']
        assert pile_loss > 1
        assert code_loss > 1
