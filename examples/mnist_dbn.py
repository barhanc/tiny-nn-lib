# %%
import numpy as np

from tqdm import trange
from mnists import MNIST

from tinynn import ebm
from tinynn.utils import sigmoid, chunks, tiles

np.random.seed(42)

# %%
# Load MNIST dataset
# ------------------
X_train = MNIST().train_images().astype(np.float32) / 255.0
X_train = X_train.reshape(-1, 28 * 28)
np.random.shuffle(X_train)

size = 20_000
X_train = X_train[:size, ...]

# %%
# Define DBN model
# ---------------
params = {
    "pc_size": None,
    "v_activation": sigmoid,
    "h_activation": sigmoid,
    "lr": 0.1,
    "momentum": 0.5,
    "l1_penalty": None,
    "l2_penalty": None,
    "weight_limit": None,
    "init_method": "Xavier",
}

dbn = ebm.DBN(
    ebm.RBM(vsize=784, hsize=200, **params),
    ebm.RBM(vsize=200, hsize=200, **params),
    ebm.RBM(vsize=200, hsize=200, **params),
)

# %%
# Greedy layer-wise training
# --------------------------
batch_size = 128
num_epochs = 50

for layer_idx, rbm in enumerate(dbn.layers):
    print(f"Layer {layer_idx}")
    for epoch in trange(num_epochs):
        for X_batch in chunks(X_train, size=batch_size):
            X_batch = dbn.propagate_up(X_batch, layer_idx)
            ebm.cdk_step(rbm, X_batch, k=5)

# %%
# Sample some digits from the DBN
# -------------------------------
noise = np.random.random(size=(12 * 24, 28 * 28))
samples = dbn.sample(noise, steps=200, verbose=True)
tiles(samples.reshape(12, 24, 28, 28))
