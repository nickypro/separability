# Personal git config settings
if [ "$1" ]
then
  if [ $1 = '-n' ]
  then
    echo "Setting git config to Nicky Pochinkov"
    git config --global user.name "Nicky Pochinkov"
    git config --global user.email "work@nicky.pro"


    # Weights & Biases
    echo "
    machine api.wandb.ai
      login user
      password 7d8ccc80648222cf13a2e1ab9a56484e521be926" >> ~/.netrc

    # Add poetry to PATH
    echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64' >> ~/.bashrc
    echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc
    echo 'alias py="poetry run python"' >> ~/.bashrc

    # Add poetry to current bash session PATH
    export PATH="/root/.local/bin:$PATH"
  fi
  # Sometimes poetry install fails and says "pending"
  if [ $2 = "-k"]
  then
    echo "export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring" >> ~/.bashrc
  fi
fi

# Dependencies (using Poetry rather than pip)
apt install python3 python3-pip vim -y
apt install python-is-python3 -y
sudo ln -s /usr/bin/python3 /usr/bin/python
curl -sSL https://install.python-poetry.org | python -
poetry config virtualenvs.in-project true
poetry install
poetry run pip install torch
poetry run pip install --no-deps detoxify -q
