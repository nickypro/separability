from model import Model
from activations import calculate_attn_crossover, evaluate_all

def test_calculate_attn_crossover_and_delete():
    # Load model and evaluate
    opt = Model( "125m", limit=1000 )
    print("# Initial Evaluation...")
    eval_before = evaluate_all( opt, 1e4 )

    # Get crossover data
    print("# Calculating Crossover Data...")
    attn_data = calculate_attn_crossover(opt, 1.0, sample_size=1e4)

    # Remove attention heads over crossover threshold
    print("# Deleting Attention Heads...")
    opt.delete_attn_pre_out_heads( attn_data['removals'], attn_data['pile_means'] ) 

    # Make sure attention heads were deleted
    print("# Final Evaluation...")
    eval_after = evaluate_all( opt, 1e4 )
    for key in eval_before.keys():
        assert eval_before[key] != eval_after[key]

    print("# Test Passed") 
    return True
    
if __name__ == "__main__":
    test_calculate_attn_crossover_and_delete()