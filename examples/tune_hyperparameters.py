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
    "wandb_project": "testing-roberta",
    "focus": "pile_codeless",
    "cripple": "code",
    "token_limit": 512,
    "svd_attn": False,
    "attn_mode": "pre-out",
    "reverse": False,
    "ff_scoring": "abs",
    "attn_scoring": "abs",
    "run_pre_test": True,
    "collection_sample_size": 1e5,
    "eval_sample_size": 1e5,
    "recalculate_activations": True,
    "dtype": "fp32",
}

scale = 0.02
options = [0.00, 0.2, 0.4, 0.6, 0.8, 1.0][::-1]

for ff_frac in options:
    config = {
        **base_config,
        "ff_frac": ff_frac * scale,
        "attn_frac": scale,
    }
    config["name"] = f'FF-{config["ff_frac"]}-ATTN-{config["attn_frac"]}'
    prune("roberta-large", config)

for attn_frac in options:
    config = {
        **base_config,
        "ff_frac": scale,
        "attn_frac": attn_frac * scale,
    }
    config["name"] = f'FF-{config["ff_frac"]}-ATTN-{config["attn_frac"]}'
    prune("roberta-large", config)


