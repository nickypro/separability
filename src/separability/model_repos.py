""" Supported and Tested Model Repos """

# List of models to use for pytest
test_model_repos = [
    "nickypro/tinyllama-15m"
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

# Meta Llama 2 Models
llama_2_model_sizes = [ "7b", "13b", "34b", "70b" ]
supported_model_repos.extend([
    f"meta-llama/llama-2-{s}-hf" for s in llama_2_model_sizes
])

# Tinyllama models
tinyllama_model_sizes = [ "15m", "42m", "110m" ]
supported_model_repos.extend([
    f"nickypro/tinyllama-{s}" for s in tinyllama_model_sizes
])

if __name__ == "__main__":
    for model_repo in supported_model_repos:
        print(model_repo)
