# Prune OPT models
poetry run python prune_30.py "facebook/opt-125m"
poetry run python prune_30.py "facebook/opt-125m" -r
poetry run python prune_30.py "facebook/opt-1.3b"
poetry run python prune_30.py "facebook/opt-1.3b" -r
poetry run python prune_30.py "facebook/opt-6.7b"
poetry run python prune_30.py "facebook/opt-6.7b" -r

# Prune Galactica Models
poetry run python prune_30.py "facebook/galactica-125m"
poetry run python prune_30.py "facebook/galactica-125m" -r
poetry run python prune_30.py "facebook/galactica-1.3b"
poetry run python prune_30.py "facebook/galactica-1.3b" -r
poetry run python prune_30.py "facebook/galactica-6.7b"
poetry run python prune_30.py "facebook/galactica-6.7b" -r
