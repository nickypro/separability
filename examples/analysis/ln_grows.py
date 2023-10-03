from separability import Model
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

plt.figure()
model_name = "EleutherAI/pythia-160m"

def plot_model(model_name):
    model = Model(model_name)
    res = model.get_residual_stream("An example text input")[::2]
    text = "An example text input"
    print("input:", text)

    norms = []
    with torch.no_grad():
        for layer in range(model.cfg.n_layers):
            print(f"L{layer:2} pre  LN",    res[layer].norm(dim=-1).cpu().numpy())

    with torch.no_grad():
        for layer in range(model.cfg.n_layers):
            LN = model.layers[layer]["ln1"]
            norms.append( LN(res[layer]).norm(dim=-1).cpu().numpy() )
            print(f"L{layer:2} post LN", norms[-1])

    plt.semilogy(np.array(norms).mean(axis=-1), "o-", label=model_name)
    plt.xlabel("layer")
    plt.ylabel("mean L2 norm")
    plt.ylim(0.99, None)

for model_name in [
            "EleutherAI/pythia-160m",
            "facebook/opt-125m",
            "facebook/galactica-125m",
            "EleutherAI/pythia-1.4B",
            "facebook/opt-1.3b",
            "facebook/galactica-1.3b",
        ]:
    plot_model(model_name)

plt.legend()
plt.savefig("ln_grows.png")
