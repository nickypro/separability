import subprocess
from separability.data_classes import PruningConfig

def prune(model_repo:str, config:dict):
    args = []
    for k, v in config.items():
        if k in ["model_repo", "model_size"]:
            continue
        if k == "reverse":
            args.append("--reverse")
            continue
        args.append(f"--{k}")
        args.append(str(v))

    # create command
    command = ["poetry", "run", "python", "prune_30.py", model_repo, *args]
    print(" ".join(command))
    subprocess.run(command, check=True)

base_config = {
    "wandb_project": "testing-hyperparam",
    "focus": "pile_codeless",
    "cripple": "code",
    "token_limit": 1000,
    "svd_attn": False,
    "attn_mode": "pre-out",
    "reverse": False,
    "ff_scoring": "abs",
    "attn_scoring": "abs",
    "run_pre_test": True,
    "collection_sample_size": 1e5,
    "eval_sample_size": 1e4,
    "recalculate_activations": False,
}

for attn_frac in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]:
    config = {
        **base_config,
        "ff_frac": 0.05,
        "attn_frac": attn_frac,
    }
    config["name"] = f'FF-{config["ff_frac"]}-ATTN-{config["attn_frac"]}'
    prune("facebook/galactica-1.3b", config)

for ff_frac in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]:
    config = {
        **base_config,
        "ff_frac": ff_frac,
        "attn_frac": 0.05,
    }
    config["name"] = f'FF-{config["ff_frac"]}-ATTN-{config["attn_frac"]}'
    prune("facebook/galactica-1.3b", config)

