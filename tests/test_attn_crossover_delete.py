""" Test the calculate_attn_crossover function. """

# pylint: disable=import-error
import pytest
from seperability.model_repos import test_model_repos
from seperability import Model
from seperability.activations import get_attn_activations, \
    get_attn_crossover, evaluate_all

class TestAttnCrossoverDelete:
    @pytest.mark.parametrize("model_repo", test_model_repos)
    def test_calculate_attn_crossover_and_delete(self, model_repo):
        print("# Running test: test_calculate_attn_crossover_and_delete")
        # Load model and evaluate
        opt = Model( model_repo, limit=1000 )
        print(" - Initial Evaluation...")
        eval_before = evaluate_all( opt, 1e3 )

        # Get crossover data
        print(" - Initial Evaluation...")

        pile_out = get_attn_activations(opt, 'pile', sample_size=1e3)
        code_out = get_attn_activations(opt, 'code', sample_size=1e3)
        attn_data = get_attn_crossover(opt, pile_out, code_out)

        # Remove attention heads over crossover threshold (very low threshold)
        removals = attn_data['crossover_multiple'] > 1.0
        print("# Deleting Attention Heads...")
        opt.delete_attn_pre_out_heads( removals, attn_data['pile_means'] )

        # Make sure attention heads were deleted
        print("# Final Evaluation...")
        eval_after = evaluate_all( opt, 1e3 )
        eval_keys = eval_before.keys()
        for key in eval_keys:
            assert eval_before[key] != eval_after[key]

        print("# Test Passed")
