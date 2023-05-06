# Prune Pythia Models
poetry run python prune_30.py "EleutherAI/pythia-1.4B" -a "sqrt" --ff_frac 0 --attn_frac 0.05 --attn_mode "pre-out" --prune_heads "mean" -n "pythia 1.4b attn h sqrt"
poetry run python prune_30.py "EleutherAI/pythia-1.4B" -a "rms"  --ff_frac 0 --attn_frac 0.05 --attn_mode "pre-out" --prune_heads "mean" -n "pythia 1.4b attn h rms"
poetry run python prune_30.py "EleutherAI/pythia-1.4B" -a "abs"  --ff_frac 0 --attn_frac 0.05 --attn_mode "pre-out" --prune_heads "mean" -n "pythia 1.4b attn h abs"
poetry run python prune_30.py "EleutherAI/pythia-1.4B" -a "std"  --ff_frac 0 --attn_frac 0.05 --attn_mode "pre-out" --prune_heads "mean" -n "pythia 1.4b attn h std"
