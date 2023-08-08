""" Test the calculate_attn_crossover function. """

# pylint: disable=import-error
import pytest
from separability.model_repos import test_model_repos
from separability import Model
from separability.data_classes import ActivationSummary
from separability.activations import get_attn_activations
from separability.scoring import get_attn_crossover
from separability.eval import evaluate_all

class TestAttnCrossoverDelete:
    @pytest.mark.parametrize("model_repo", test_model_repos)
    def test_calculate_attn_crossover_and_delete(self, model_repo):
        print("# Running test: test_calculate_attn_crossover_and_delete")
        # Load model and evaluate
        opt = Model( model_repo, limit=1000 )
        print(" - Initial Evaluation...")
        eval_before = evaluate_all( opt, 1e3 ).misc

        # Get crossover data
        print(" - Initial Evaluation...")

        pile_out: ActivationSummary = \
            get_attn_activations(opt, 'pile', sample_size=1e3)
        code_out: ActivationSummary = \
            get_attn_activations(opt, 'code', sample_size=1e3)
        crossover_multiple = get_attn_crossover(opt, pile_out, code_out)

        # Remove attention heads over crossover threshold (very low threshold)
        removals = crossover_multiple > 1.0
        print("# Deleting Attention Heads...")
        opt.delete_attn_pre_out_heads( removals, pile_out.mean )

        # Make sure attention heads were deleted
        print("# Final Evaluation...")
        eval_after = evaluate_all( opt, 1e3 ).misc
        eval_keys = eval_before.keys()
        for key in eval_keys:
            assert eval_before[key] != eval_after[key]

        print("# Test Passed")
