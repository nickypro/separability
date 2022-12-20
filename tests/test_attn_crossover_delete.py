""" Test the calculate_attn_crossover function. """
import argparse
from activations import calculate_attn_crossover, evaluate_all
from model import Model

def test_calculate_attn_crossover_and_delete( verbose: bool = False ):
    print("# Running test: test_calculate_attn_crossover_and_delete")
    # Load model and evaluate
    opt = Model( "125m", limit=1000 )
    if verbose:
        print(" - Initial Evaluation...")
    eval_before = evaluate_all( opt, 1e3 )

    # Get crossover data
    if verbose:
        print(" - Initial Evaluation...")
    attn_data = calculate_attn_crossover(opt, sample_size=1e3)

    # Remove attention heads over crossover threshold
    removals = attn_data['crossover_multiple'] > 1.0
    if verbose:
        print("# Deleting Attention Heads...")
    opt.delete_attn_pre_out_heads( removals, attn_data['pile_means'] )

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
    parser = argparse.ArgumentParser()
    parser.add_argument( "--verbose", "-v", action="store_true", default=False,
        help="Print extra information" )
    args = parser.parse_args()

    test_calculate_attn_crossover_and_delete(args.verbose)
