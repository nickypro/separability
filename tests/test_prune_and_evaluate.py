import warnings
import torch

# pylint: disable=import-error
import pytest
from seperability.model_names import test_model_names
from seperability import Model
from seperability.activations import prune_and_evaluate

class TestPruneAndEvaluate:
    @pytest.mark.parametrize("model_name", test_model_names)
    def test_prune_and_evaluate_accelerate(self, model_name):
        print( "# Running Test: test_prune_and_evaluate_accelerate" )

        opt = Model(model_name, limit=1000, use_accelerator=True)

        if torch.cuda.device_count() <= 1:
            warnings.warn( "Multi-gpu not available", category=UserWarning )
            return

        data = prune_and_evaluate(opt, 0.05, 0.05, 0.001, 1e4, 1e4)

        pile_loss = data.loss_data['pile']['loss']
        code_loss = data.loss_data['code']['loss']
        assert pile_loss < code_loss

    @pytest.mark.parametrize("model_name", test_model_names)
    def test_prune_and_evaluate(self, model_name):
        print( "# Running Test: test_prune_and_evaluate" )

        opt = Model(model_name, limit=1000, use_accelerator=False)
        data = prune_and_evaluate(opt, 0.05, 0.05, 0.001, 1e4, 1e4)

        pile_loss = data.loss_data['pile']['loss']
        code_loss = data.loss_data['code']['loss']
        assert pile_loss < code_loss
