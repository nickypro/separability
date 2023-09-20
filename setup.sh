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

    # install Tmux Plugins Manager
    git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm

    # Save TMUX config
    echo "# ~/.tmux.conf
    set -g mouse on

    # These bindings are for X Windows only. If you're using a different
    # window system you have to replace the `xsel` commands with something
    # else. See https://github.com/tmux/tmux/wiki/Clipboard#available-tools
    bind -T copy-mode    DoubleClick1Pane select-pane \; send -X select-word \; send -X copy-pipe-no-clear "xsel -i"
    bind -T copy-mode-vi DoubleClick1Pane select-pane \; send -X select-word \; send -X copy-pipe-no-clear "xsel -i"
    bind -n DoubleClick1Pane select-pane \; copy-mode -M \; send -X select-word \; send -X copy-pipe-no-clear "xsel -i"
    bind -T copy-mode    TripleClick1Pane select-pane \; send -X select-line \; send -X copy-pipe-no-clear "xsel -i"
    bind -T copy-mode-vi TripleClick1Pane select-pane \; send -X select-line \; send -X copy-pipe-no-clear "xsel -i"
    bind -n TripleClick1Pane select-pane \; copy-mode -M \; send -X select-line \; send -X copy-pipe-no-clear "xsel -i"
    bind -n MouseDown2Pane run "tmux set-buffer -b primary_selection \"$(xsel -o)\"; tmux paste-buffer -b primary_selection; tmux delete-buffer -b primary_selection"

    set -g @plugin 'tmux-plugins/tpm'
    set -g @plugin 'tmux-plugins/tmux-yank'
    set -g @yank_action 'copy-pipe-no-clear'
    bind -T copy-mode    C-c send -X copy-pipe-no-clear "xsel -i --clipboard"
    bind -T copy-mode-vi C-c send -X copy-pipe-no-clear "xsel -i --clipboard"

    # Initialize TMUX plugin manager (keep this line at the very bottom of tmux.conf)
    run '~/.tmux/plugins/tpm/tpm'" > ~/.tmux.conf


  fi
  # Sometimes poetry install fails and says "pending"
  if [ $1 = '-k' ]
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
