from typing import List
import os
import tempfile
from dask.distributed import Client

def run_training(configs: List[str], vram_requirements: list):
    # Connect to the Dask scheduler
    client = Client('oracle.nicky.pro:8786')

    def train_model(config_str):
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp_path = temp.name

        _path = "cd /root/separability && PATH=/root/.local/bin:$PATH"
        _pre = "poetry run python"
        _cmd = f"examples/prune_30.py {config_str}"
        _post = f"> {temp_path} 2>&1"
        full_cmd = f"{_path} {_pre} {_cmd} {_post}"
        os.system(full_cmd)

        with open(temp_path, 'r') as file:
            logs = file.read()

        os.remove(temp_path)
        print(logs)
        print("OUTPUT OF", full_cmd)

        return logs, f"Training done with config: {config_str}"

    futures = [
        client.submit(train_model, config) for config in configs
    ]
    results = client.gather(futures)

    for result in results:
        print(result)
    client.close()

if __name__ == '__main__':
    configs = ["nickypro/tinyllama-15m", "nickypro/tinyllama-42m"]
    vram_requirements = [0.1, 0.2]
    run_training(configs, vram_requirements)
