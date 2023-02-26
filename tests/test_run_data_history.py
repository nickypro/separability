import torch

# pylint: disable=import-error
import pytest
from separability.model_repos import test_model_repos
from separability import Model
from separability.data_classes import RunDataHistory
from separability.activations import evaluate_all

class TestRunDataHistory:
    @pytest.mark.parametrize("model_repo", test_model_repos)
    def test_run_data_history(self, model_repo):
        opt = Model(model_repo, limit=1000)
        history = RunDataHistory(use_wandb=False)

        eval_data = evaluate_all(opt, 1e4)
        history.add(eval_data)

        assert len(history.history) == 1
        assert history.df['area_base'][0] == 1.0
        assert history.df['area_topk'][0] == 1.0
        assert history.df['area_skip'][0] == 1.0
        assert history.df['area_topk_skip'][0] == 1.0

        # TODO: Test more things
