import torch

from model import Model
from activations import evaluate_all
from data_classes import RunDataHistory

class TestRunDataHistory:
    def test_run_data_history(self):
        opt = Model("125m", limit=1000)
        history = RunDataHistory(use_wandb=False)

        eval_data = evaluate_all(opt, 1e4)
        history.add(eval_data)

        assert len(history.history) == 1
        assert history.df['area_base'][0] == 1.0
        assert history.df['area_topk'][0] == 1.0
        assert history.df['area_skip'][0] == 1.0
        assert history.df['area_topk_skip'][0] == 1.0

        # TODO: Test more things
