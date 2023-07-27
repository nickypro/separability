import os
import subprocess
from time import sleep
from celery import Celery

broker_url = os.getenv('BROKER_URL')
app = Celery('tasks', broker=broker_url)

# simple task to test it is working
@app.task
def reverse_string(text):
    sleep(3)
    return text[::-1]

@app.task
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
    subprocess.run(command, check=True)
