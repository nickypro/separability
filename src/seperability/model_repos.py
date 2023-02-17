""" Supported and Tested Model Repos """

# List of models to use for pytest
test_model_repos = [
    "facebook/opt-125m",
    "facebook/galactica-125m",
]

# List of models to include for 'valid tested models' in Model
supported_model_repos = []

# Meta OPT Models
opt_model_sizes = [ "125m", "1.3b", "2.7b", "6.7b", "13b", "30b", "66b" ]
supported_model_repos.extend([
    f"facebook/opt-{s}"       for s in opt_model_sizes
])

# Meta Galactica Models
galactica_model_sizes = [ "125m", "1.3b", "6.7b", "30b", "120b" ]
supported_model_repos.extend([
    f"facebook/galactica-{s}" for s in galactica_model_sizes
])

if __name__ == "__main__":
    for model_repo in supported_model_repos:
        print(model_repo)
