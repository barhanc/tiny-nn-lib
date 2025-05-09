# %%
import numpy as np

from math import ceil
from tqdm import trange
from mnists import MNIST

from tinynn import nn
from tinynn.utils import chunks, tiles, sigmoid

np.random.seed(42)

# %%
# Load MNIST dataset
# ------------------
mnist = MNIST()

# Training set
X_train = mnist.train_images().astype(np.float32) / 255.0
X_train = X_train.reshape(-1, 28 * 28)
np.random.shuffle(X_train)

size = 60_000
X_train = X_train[:size, ...]

# Test set
X_test = mnist.test_images().astype(np.float32) / 255.0
X_test = X_test.reshape(-1, 28 * 28)

tiles(X_train[: 12 * 24, ...].reshape(-1, 24, 28, 28))

# %%
# Define Autoencoder
# ------------------
params = {
    "lr": 0.01,
    "momentum": 0.95,
    "l2_penalty": None,
    "weight_limit": None,
    "init_method": "Xavier",
}

autoencoder = nn.Sequential(
    nn.Linear(784, 256, **params),
    nn.Sigmoid(),
    nn.Linear(256, 128, **params),
    nn.Sigmoid(),
    nn.Linear(128, 64, **params),
    nn.Sigmoid(),
    nn.Linear(64, 10, **params),
    #
    nn.Linear(10, 64, **params),
    nn.Sigmoid(),
    nn.Linear(64, 128, **params),
    nn.Sigmoid(),
    nn.Linear(128, 256, **params),
    nn.Sigmoid(),
    nn.Linear(256, 784, **params),
)


# %%
# Training loop
# -------------
def cross_entropy(targets, logits) -> float:
    batch_size = targets.shape[0]
    σ = sigmoid(logits)
    return -1 / batch_size * np.sum(targets * np.log(σ) + (1 - targets) * np.log(1 - σ))


batch_size = 128
num_epochs = 50

for epoch in (pbar := trange(num_epochs)):
    dataloader = chunks(X_train, size=batch_size)
    total = ceil(len(X_train) / batch_size)

    # --- Training phase
    for X_batch in dataloader:
        # --- Forward pass
        logits = autoencoder.forward(X_batch, training=True)

        # --- Compute gradient of the loss function w.r.t. the outputs of the network, assuming that
        #     the loss function is the *pixelwise binary cross entropy*.
        grad_y = 1 / len(X_batch) * (sigmoid(logits) - X_batch)

        # --- Backward pass
        autoencoder.backward(X_batch, grad_y)

    # --- Validation phase
    valid_loss = cross_entropy(X_test, autoencoder.forward(X_test, training=False))

    # --- Log progress
    pbar.set_description(f"Epoch {epoch+1:>3d} | Valid. Loss {valid_loss:>2.2f} |")

# %%
# Show reconstructions
# --------------------
tiles(X_test[: 8 * 24, ...].reshape(8, 24, 28, 28))
reconstructions = autoencoder.forward(X_test[: 8 * 24, ...], training=False)
reconstructions = sigmoid(reconstructions)
tiles(reconstructions.reshape(8, 24, 28, 28))

# %%
