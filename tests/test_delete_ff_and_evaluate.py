import torch

from model import Model
from activations import count_ff_key_activations, evaluate_all

def test_delete_ff_and_evaluate( verbose: bool = False ):
    print("# Running test: test_delete_ff_and_evaluate")
    # Load model and evaluate
    opt = Model( "125m", limit=1000 )
    if verbose:
        print(" - Initial Evaluation...")
    eval_before = evaluate_all( opt, 1e3 )

    # Get crossover data
    if verbose:
        print(" - Initial Evaluation...")

    pile_count = count_ff_key_activations(opt, 'pile', sample_size=1e3)
    code_count = count_ff_key_activations(opt, 'code', sample_size=1e3)

    # Remove attention heads over crossover threshold (very low threshold here)
    removals = code_count > pile_count
    if verbose:
        print("# Deleting FF Keys...")

    opt.delete_ff_keys(removals)

    # Make sure attention heads were deleted
    if verbose:
        print("# Final Evaluation...")
    eval_after = evaluate_all( opt, 1e3 )
    eval_keys = eval_before.keys()
    for key in eval_keys:
        assert eval_before[key] != eval_after[key]

    print("# Test Passed")
    return True

if __name__ == "__main__":
    test_delete_ff_and_evaluate(True)
