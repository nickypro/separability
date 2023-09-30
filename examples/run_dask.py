from typing import List
from dask.distributed import Client
from separability.prune import run_pruning
from separability.data_classes import PruningConfig

def run_training(configs: List[PruningConfig], vram_requirements: list):
    # connect to the dask scheduler
    client = Client('oracle.nicky.pro:8786')

    # some heavy computation that should be distributed across dask workers
    def train_model(_config):
        run_pruning(_config)
        return f"Training done with config: {_config}"

    # submit tasks with associated VRAM requirements
    futures = []
    for i, config in enumerate(configs):
        future = client.submit(train_model, config, resources={"VRAM": vram_requirements[i]})
        futures.append(future)

    # gather results
    results = client.gather(futures)
    for result in results:
        print(result)


configs = [
    PruningConfig("nickypro/tinyllama-15m"),
    PruningConfig("nickypro/tinyllama-42m"),
]

vram_requirements = [
    0.1,
    0.2,
    # your list of VRAM requirements corresponding to each config
]

run_training(configs, vram_requirements)
