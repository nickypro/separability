""" Supported and Tested Model Names """

# List of models to use for pytest
test_model_names = [
    "facebook/opt-125m",
    "facebook/galactica-125m",
]

# List of models to include for 'valid tested models' in Model
supported_model_names = []

# Meta OPT Models
opt_model_sizes = [ "125m", "350m", "1.3b", "2.7b", "6.7b", "13b", "30b", "66b" ]
supported_model_names.extend([
    f"facebook/opt-{s}"       for s in opt_model_sizes
])

# Meta Galactica Models
galactica_model_sizes = [ "125m", "1.3b", "6.7b", "30b", "120b" ]
supported_model_names.extend([
    f"facebook/galactica-{s}" for s in galactica_model_sizes
])
