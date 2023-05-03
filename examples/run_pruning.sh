# Prune Pythia Models
poetry run python prune_30.py "EleutherAI/pythia-160m"    -f "std" -n "pythia 160m pc ff std py cripple"
poetry run python prune_30.py "EleutherAI/pythia-160m" -r -f "std" -n "pythia 160m pc ff std py focus"
poetry run python prune_30.py "EleutherAI/pythia-1.4B"    -f "std" -n "pythia 1.4b pc ff std py cripple"
poetry run python prune_30.py "EleutherAI/pythia-1.4B" -r -f "std" -n "pythia 1.4b pc ff std py focus"
#poetry run python prune_30.py "EleutherAI/pythia-6.9B"    -f "std" -n "pythia 6.9b pc ff std py cripple"
#poetry run python prune_30.py "EleutherAI/pythia-6.9B" -r -f "std" -n "pythia 6.9b pc ff std py focus"

# Prune OPT models
poetry run python prune_30.py "facebook/opt-125m"    -f "std" -n "opt 125m pc ff std py cripple"
poetry run python prune_30.py "facebook/opt-125m" -r -f "std" -n "opt 125m pc ff std py focus"
poetry run python prune_30.py "facebook/opt-1.3b"    -f "std" -n "opt 1.3b pc ff std py cripple"
poetry run python prune_30.py "facebook/opt-1.3b" -r -f "std" -n "opt 1.3b pc ff std py focus"
#poetry run python prune_30.py "facebook/opt-6.7b"    -f "std" -n "opt 6.7b pc ff std py cripple"
#poetry run python prune_30.py "facebook/opt-6.7b" -r -f "std" -n "opt 6.7b pc ff std py focus"

# Prune Galactica Models
poetry run python prune_30.py "facebook/galactica-125m"    -f "std" -n "gal 125m pc ff std py cripple"
poetry run python prune_30.py "facebook/galactica-125m" -r -f "std" -n "gal 125m pc ff std py focus"
poetry run python prune_30.py "facebook/galactica-1.3b"    -f "std" -n "gal 1.3b pc ff std py cripple"
poetry run python prune_30.py "facebook/galactica-1.3b" -r -f "std" -n "gal 1.3b pc ff std py focus"
#poetry run python prune_30.py "facebook/galactica-6.7b"    -f "std" -n "gal 6.7b pc ff std py cripple"
#poetry run python prune_30.py "facebook/galactica-6.7b" -r -f "std" -n "gal 6.7b pc ff std py focus"
