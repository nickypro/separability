# Prune OPT models
poetry run python prune_30.py "facebook/opt-125m"    -n "opt 125m pc + code cripple"
poetry run python prune_30.py "facebook/opt-125m" -r -n "opt 125m pc + code focus"
poetry run python prune_30.py "facebook/opt-1.3b"    -n "opt 1.3b pc + code cripple"
poetry run python prune_30.py "facebook/opt-1.3b" -r -n "opt 1.3b pc + code focus"
poetry run python prune_30.py "facebook/opt-6.7b"    -n "opt 6.7b pc + code cripple"
poetry run python prune_30.py "facebook/opt-6.7b" -r -n "opt 6.7b pc + code focus"

# Prune Galactica Models
poetry run python prune_30.py "facebook/galactica-125m"    -n "gal 125m pc + code cripple"
poetry run python prune_30.py "facebook/galactica-125m" -r -n "gal 125m pc + code focus"
poetry run python prune_30.py "facebook/galactica-1.3b"    -n "gal 1.3b pc + code cripple"
poetry run python prune_30.py "facebook/galactica-1.3b" -r -n "gal 1.3b pc + code focus"
poetry run python prune_30.py "facebook/galactica-6.7b"    -n "gal 6.7b pc + code cripple"
poetry run python prune_30.py "facebook/galactica-6.7b" -r -n "gal 6.7b pc + code focus"
