import tasks

# Run the tasks
def generate_list_of_configs(loop_config):
    _configs = [{}]
    for key, value in loop_config.items():
        # if value is list, loop through
        if isinstance(value, list):
            extended_configs = []
            for c in _configs:
                for v in value:
                    extended_configs.append({**c, key: v})
            _configs = extended_configs

        else:
            for c in _configs:
                c[key] = value

    return _configs

base_config = {
    "wandb_project": "new-method-compare",
    "model_repo": [
        "facebook/opt-1.3b",
        "facebook/galactica-1.3b",
        "EleutherAI/pythia-1.4B",
    ],
    "focus": "pile_codeless",
    "cripple": "code",
    "run_pre_test": True,
    "token_limit": 1000,
    "svd_attn": False,
    "attn_mode": "pre-out",
    "reverse": [True, False],
}

attn_configs = generate_list_of_configs({
    **base_config,
    "attn_frac": 0.02,
    "attn_scoring": ["abs", "std", "rms"],
    "ff_frac": 0.0,
})

ff_configs = generate_list_of_configs({
    **base_config,
    "attn_frac": 0.0,
    "ff_frac": 0.02,
    "ff_scoring": ["abs", "std", "rms"],
})

all_configs = [*attn_configs, *ff_configs]

def gen_name(config):
    name = config["model_repo"].split("/")[-1][:3]
    name += " " + config["model_repo"].split("/")[-1].split("-")[-1]
    if config["attn_frac"] > 0:
        name += " attn"
        name += " " + config["attn_mode"][0]
        name += " " + config["attn_scoring"]
        if config["svd_attn"]:
            name += " svd"
    if config["ff_frac"] > 0:
        name += " ff"
        name += " " + config["ff_scoring"][0]
    if config["reverse"]:
        name += " rev"

    return name

for config in attn_configs:
    config["name"] = gen_name(config)
    tasks.prune.delay(config["model_repo"], config)

