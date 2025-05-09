# %%
import numpy as np

from math import ceil
from tqdm import trange
from mnists import MNIST

from tinynn import nn
from tinynn.utils import chunks, tiles, onehot, softmax

np.random.seed(42)

# %%
# Load MNIST dataset
# ------------------
mnist = MNIST()

# Training set
X_train = mnist.train_images().astype(np.float32) / 255.0
X_train = X_train.reshape(-1, 28 * 28)
y_train = mnist.train_labels()
y_train = onehot(y_train)

size = 60_000
X_train = X_train[:size, ...]
y_train = y_train[:size, ...]

permute = np.random.permutation(len(X_train))
X_train = X_train[permute, ...]
y_train = y_train[permute, ...]

# Test set
X_test = mnist.test_images().astype(np.float32) / 255.0
X_test = X_test.reshape(-1, 28 * 28)
y_test = mnist.test_labels()

tiles(X_train[: 12 * 24, ...].reshape(-1, 24, 28, 28))

# %%
# Define MLP model
# ---------------
params = {
    "lr": 0.05,
    "momentum": 0.95,
    "weight_limit": 5.0,
    "l2_penalty": None,
    "init_method": "He",
}

mlp = nn.Sequential(
    nn.Linear(vsize=784, hsize=512, **params),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(vsize=512, hsize=256, **params),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(vsize=256, hsize=128, **params),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(vsize=128, hsize=10, **params),
)

# %%
# Training loop
# -------------
batch_size = 128
num_epochs = 200

for epoch in (pbar := trange(num_epochs)):
    dataloader = chunks(X_train, y_train, size=batch_size)
    total = ceil(len(X_train) / batch_size)

    # --- Training phase
    for X_batch, y_batch in dataloader:
        # --- Forward pass
        logits = mlp.forward(X_batch, training=True)

        # --- Compute gradient of the loss function w.r.t. the outputs of the network, assuming that
        #     the loss function is the cross entropy
        grad_y = 1 / len(X_batch) * (softmax(logits) - y_batch)

        # --- Backward pass
        mlp.backward(X_batch, grad_y)

    # --- Validation phase
    logits = mlp.forward(X_test, training=False)
    accuracy = 1 / len(y_test) * np.sum(np.argmax(logits, axis=1) == y_test)

    # --- Log progress
    pbar.set_description(f"Epoch {epoch+1:>3d} | Accuracy {accuracy:>2.2%} |")


# %%
layer: nn.Linear = next(filter(lambda l: isinstance(l, nn.Linear), mlp.layers))
filters = layer.w.T
tiles(filters.reshape(16, -1, 28, 28))

# %%
