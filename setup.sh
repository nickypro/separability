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
    echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc
    echo 'alias py="poetry run python"' >> ~/.bashrc

    # Add poetry to current bash session PATH
    export PATH="/root/.local/bin:$PATH"
  fi
fi
# Dependencies (using Poetry rather than pip)
apt install python3.8 python3-pip vim -y
curl -sSL https://install.python-poetry.org | python3.8 -
poetry config virtualenvs.in-project true
poetry install
