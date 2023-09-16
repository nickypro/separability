prune_gal() {
    poetry run python prune_30.py facebook/galactica-1.3b \
        --wandb_project new-method-compare --focus pile_codeless \
        --cripple code --run_pre_test True --svd_attn False \
        --attn_mode pre-out --token_limit 1000 --name "$@"
}
prune_opt() {
    poetry run python prune_30.py facebook/opt-1.3b \
        --wandb_project new-method-compare --focus pile_codeless \
        --cripple code --run_pre_test True --svd_attn False \
        --attn_mode pre-out --token_limit 1000 --name "$@"
}
prune_pyt() {
    poetry run python prune_30.py facebook/pythia-1.3b \
        --wandb_project new-method-compare --focus pile_codeless \
        --cripple code --run_pre_test True --svd_attn False \
        --attn_mode pre-out --token_limit 1000 --name "$@"
}

prune_gal "gal 1.3b ff abs" --attn_frac 0.0 --ff_frac 0.02 --ff_scoring abs
prune_opt "opt 1.3b ff abs" --attn_frac 0.0 --ff_frac 0.02 --ff_scoring abs
prune_pyt "pythia 1.4b ff abs" --attn_frac 0.0 --ff_frac 0.02 --ff_scoring abs
