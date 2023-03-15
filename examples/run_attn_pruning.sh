poetry run python prune_30.py -s abs  "facebook/galactica-1.3b"
poetry run python prune_30.py -s sqrt "facebook/galactica-1.3b"
poetry run python prune_30.py -s std  "facebook/galactica-1.3b"
poetry run python prune_30.py --svd -s abs  "facebook/galactica-1.3b"
poetry run python prune_30.py --svd -s sqrt "facebook/galactica-1.3b"
poetry run python prune_30.py --svd -s std  "facebook/galactica-1.3b"

poetry run python prune_30.py -s abs --prune_heads mean "facebook/galactica-1.3b"
poetry run python prune_30.py -s sqrt --prune_heads mean "facebook/galactica-1.3b"
poetry run python prune_30.py -s std --prune_heads mean "facebook/galactica-1.3b"
poetry run python prune_30.py --svd -s abs --prune_heads mean "facebook/galactica-1.3b"
poetry run python prune_30.py --svd -s sqrt --prune_heads mean "facebook/galactica-1.3b"
poetry run python prune_30.py --svd -s std --prune_heads mean "facebook/galactica-1.3b"

