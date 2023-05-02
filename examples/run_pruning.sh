# Prune Pythia Models
poetry run python prune_30.py "EleutherAI/pythia-160m"    -f "rms" -n "pythia 160m pc ff rms code cripple"
poetry run python prune_30.py "EleutherAI/pythia-160m" -r -f "rms" -n "pythia 160m pc ff rms code focus"
poetry run python prune_30.py "EleutherAI/pythia-1.4B"    -f "rms" -n "pythia 1.4b pc ff rms code cripple"
poetry run python prune_30.py "EleutherAI/pythia-1.4B" -r -f "rms" -n "pythia 1.4b pc ff rms code focus"
poetry run python prune_30.py "EleutherAI/pythia-6.9B"    -f "rms" -n "pythia 6.9b pc ff rms code cripple"
poetry run python prune_30.py "EleutherAI/pythia-6.9B" -r -f "rms" -n "pythia 6.9b pc ff rms code focus"

# Prune OPT models
poetry run python prune_30.py "facebook/opt-125m"    -f "rms" -n "opt 125m pc ff rms code cripple"
poetry run python prune_30.py "facebook/opt-125m" -r -f "rms" -n "opt 125m pc ff rms code focus"
poetry run python prune_30.py "facebook/opt-1.3b"    -f "rms" -n "opt 1.3b pc ff rms code cripple"
poetry run python prune_30.py "facebook/opt-1.3b" -r -f "rms" -n "opt 1.3b pc ff rms code focus"
poetry run python prune_30.py "facebook/opt-6.7b"    -f "rms" -n "opt 6.7b pc ff rms code cripple"
poetry run python prune_30.py "facebook/opt-6.7b" -r -f "rms" -n "opt 6.7b pc ff rms code focus"

# Prune Galactica Models
poetry run python prune_30.py "facebook/galactica-125m"    -f "rms" -n "gal 125m pc ff rms code cripple"
poetry run python prune_30.py "facebook/galactica-125m" -r -f "rms" -n "gal 125m pc ff rms code focus"
poetry run python prune_30.py "facebook/galactica-1.3b"    -f "rms" -n "gal 1.3b pc ff rms code cripple"
poetry run python prune_30.py "facebook/galactica-1.3b" -r -f "rms" -n "gal 1.3b pc ff rms code focus"
poetry run python prune_30.py "facebook/galactica-6.7b"    -f "rms" -n "gal 6.7b pc ff rms code cripple"
poetry run python prune_30.py "facebook/galactica-6.7b" -r -f "rms" -n "gal 6.7b pc ff rms code focus"
