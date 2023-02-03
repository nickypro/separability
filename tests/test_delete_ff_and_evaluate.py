import torch

from model import Model
from activations import count_ff_key_activations, evaluate_all

class TestDeleteFFKeys:
    def test_delete_ff_and_evaluate(self):
        print("# Running test: test_delete_ff_and_evaluate")
        # Load model and evaluate
        opt = Model( "125m", limit=1000 )
        print(" - Initial Evaluation...")
        eval_before = evaluate_all( opt, 1e3 )

        # Get crossover data
        print(" - Initial Evaluation...")

        pile_count = count_ff_key_activations(opt, 'pile', sample_size=1e3)
        code_count = count_ff_key_activations(opt, 'code', sample_size=1e3)

        # Remove attention heads over crossover threshold (very low threshold here)
        removals = code_count > pile_count
        print("# Deleting FF Keys...")

        opt.delete_ff_keys(removals)

        # Make sure attention heads were deleted
        print("# Final Evaluation...")
        eval_after = evaluate_all( opt, 1e3 )
        eval_keys = eval_before.keys()
        for key in eval_keys:
            assert eval_before[key] != eval_after[key]

        print("# Test Passed")
