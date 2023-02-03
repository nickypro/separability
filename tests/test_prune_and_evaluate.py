from model import Model
from activations import prune_and_evaluate

class TestPruneAndEvaluate:
    def test_prune_and_evaluate(self):
        print( "# Running Test: test_prune_and_evaluate" )

        opt = Model("125m", limit=1000)
        data = prune_and_evaluate(opt, 0.05, 0.05, 0.001, 1e4, 1e4)

        pile_loss = data.loss_data['pile']['loss']
        code_loss = data.loss_data['code']['loss']
        assert pile_loss < code_loss
