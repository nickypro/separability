# Prune Pythia Models
poetry run python prune_30.py "EleutherAI/pythia-160m"    -f "std" -n "pythia 160m pc ff code cripple std"
poetry run python prune_30.py "EleutherAI/pythia-160m" -r -f "std" -n "pythia 160m pc ff code focus std"
poetry run python prune_30.py "EleutherAI/pythia-1.4B"    -f "std" -n "pythia 1.4b pc ff code cripple std"
poetry run python prune_30.py "EleutherAI/pythia-1.4B" -r -f "std" -n "pythia 1.4b pc ff code focus std"
poetry run python prune_30.py "EleutherAI/pythia-6.9B"    -f "std" -n "pythia 6.9b pc ff code cripple std"
poetry run python prune_30.py "EleutherAI/pythia-6.9B" -r -f "std" -n "pythia 6.9b pc ff code focus std"

poetry run python prune_30.py "EleutherAI/pythia-160m"    -f "freq" -n "pythia 160m pc ff code cripple freq"
poetry run python prune_30.py "EleutherAI/pythia-160m" -r -f "freq" -n "pythia 160m pc ff code focus freq"
poetry run python prune_30.py "EleutherAI/pythia-1.4B"    -f "freq" -n "pythia 1.4b pc ff code cripple freq"
poetry run python prune_30.py "EleutherAI/pythia-1.4B" -r -f "freq" -n "pythia 1.4b pc ff code focus freq"
poetry run python prune_30.py "EleutherAI/pythia-6.9B"    -f "freq" -n "pythia 6.9b pc ff code cripple freq"
poetry run python prune_30.py "EleutherAI/pythia-6.9B" -r -f "freq" -n "pythia 6.9b pc ff code focus freq"