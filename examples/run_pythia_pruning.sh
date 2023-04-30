# Prune Pythia Models
poetry run python prune_30.py "EleutherAI/pythia-140m"    -f "std" -n "pythia 140m pc ff code cripple"
poetry run python prune_30.py "EleutherAI/pythia-140m" -r -f "std" -n "pythia 140m pc ff code focus"
poetry run python prune_30.py "EleutherAI/pythia-1.4B"    -f "std" -n "pythia 1.4b pc ff code cripple"
poetry run python prune_30.py "EleutherAI/pythia-1.4B" -r -f "std" -n "pythia 1.4b pc ff code focus"
poetry run python prune_30.py "EleutherAI/pythia-6.9B"    -f "std" -n "pythia 6.9b pc ff code cripple"
poetry run python prune_30.py "EleutherAI/pythia-6.9B" -r -f "std" -n "pythia 6.9b pc ff code focus"