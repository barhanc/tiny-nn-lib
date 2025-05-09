import numpy as np

from math import ceil
from tqdm import tqdm
from mnists import MNIST

from tinynn import ebm
from tinynn.utils import sigmoid, chunks, tiles


# Load MNIST dataset
# ------------------
X_train = MNIST().train_images().astype(np.float32) / 255.0
X_train = X_train.reshape(-1, 28 * 28)
np.random.shuffle(X_train)

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
    ebm.RBM(vsize=784, hsize=512, **params),
    ebm.RBM(vsize=512, hsize=512, **params),
    ebm.RBM(vsize=512, hsize=256, **params),
    ebm.RBM(vsize=256, hsize=256, **params),
)

# Greedy layer-wise training
# --------------------------
batch_size = 512
num_epochs = 50

for layer_idx, rbm in enumerate(dbn.layers):
    print(f"Layer {layer_idx}")

    for epoch in range(num_epochs):
        total = ceil(len(X_train) / batch_size)
        dataloader = chunks(X_train, size=batch_size)

        for X_batch in tqdm(dataloader, desc=f"Epoch {epoch+1:>3d}\t", total=total):
            # --- Use CD-k for RBM training
            minibatch = dbn.propagate_up(X_batch, layer_idx)
            ebm.cdk_step(rbm, minibatch, k=5)


# Sample some digits from the DBN
# -------------------------------
noise = np.random.random(size=(12 * 24, 28 * 28))
samples = dbn.sample(noise, steps=1_000)
tiles(samples.reshape(-1, 24, 28, 28))
