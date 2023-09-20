# PRUNE CIVIL TOXIC

prune_gpt2() {
    poetry run python prune_30.py "gpt2-large" \
        --wandb_project civil-toxic \
        --focus civil --cripple toxic --additional_datasets wiki,toxicity \
        --recalculate_activations True \
        --run_pre_test True --svd_attn False \
        --ff_scoring abs --attn_scoring abs \
        --attn_mode pre-out --token_limit 1000 --name "$@"
}

prune_llama2() {
    poetry run python prune_30.py "meta-llama/Llama-2-7b" \
        --wandb_project civil-toxic \
        --focus civil --cripple toxic --additional_datasets wiki,toxicity,mmlu:all \
        --recalculate_activations True \
        --run_pre_test True --svd_attn False \
        --ff_scoring abs --attn_scoring abs \
        --attn_mode pre-out --token_limit 1000 --name "$@"
}

prune_gpt2 "gpt2-large 0% 0.5%" --ff_frac 0.0 --attn_frac 0.005
prune_llama2 "llama 7b 0% 0.5%" --ff_frac 0.0 --attn_frac 0.005

