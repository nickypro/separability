import warnings
import torch

from seperability import Model
from seperability.activations import prune_and_evaluate

class TestPruneAndEvaluate:
    model_name = "facebook/opt-125m"

    def test_prune_and_evaluate_accelerate(self):
        print( "# Running Test: test_prune_and_evaluate_accelerate" )

        opt = Model(self.model_name, limit=1000, use_accelerator=True)

        if torch.cuda.device_count() <= 1:
            warnings.warn( "Multi-gpu not available", category=UserWarning )
            return

        data = prune_and_evaluate(opt, 0.05, 0.05, 0.001, 1e4, 1e4)

        pile_loss = data.loss_data['pile']['loss']
        code_loss = data.loss_data['code']['loss']
        assert pile_loss < code_loss

    def test_prune_and_evaluate(self):
        print( "# Running Test: test_prune_and_evaluate" )

        opt = Model(self.model_name, limit=1000, use_accelerator=False)
        data = prune_and_evaluate(opt, 0.05, 0.05, 0.001, 1e4, 1e4)

        pile_loss = data.loss_data['pile']['loss']
        code_loss = data.loss_data['code']['loss']
        assert pile_loss < code_loss
