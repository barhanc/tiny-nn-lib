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

# %%
# Define MLP model
# ---------------
params = {
    "lr": 0.05,
    "momentum": 0.9,
    "weight_limit": None,
    "l2_penalty": None,
    "init_method": "He",
}

mlp = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(vsize=784, hsize=512, **params),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(vsize=512, hsize=512, **params),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(vsize=512, hsize=10, **params),
)

# %%
# Training loop
# -------------
batch_size = 128
num_epochs = 50

for epoch in (pbar := trange(num_epochs)):
    dataloader = zip(chunks(X_train, batch_size), chunks(y_train, batch_size))
    total = ceil(len(X_train) / batch_size)

    # Training phase
    for X_batch, y_batch in dataloader:
        # Forward pass
        logits = mlp.forward(X_batch, training=True)

        # Compute gradient of the loss function w.r.t. the outputs of the network, assuming that the
        # loss function is the cross entropy
        grad_y = 1 / len(X_batch) * (softmax(logits) - y_batch)

        # Backward pass
        mlp.backward(X_batch, grad_y)

    # Validation phase
    logits = mlp.forward(X_test, training=False)
    accuracy = 1 / len(y_test) * np.sum(np.argmax(logits, axis=1) == y_test)

    # Log progress
    pbar.set_description(f"Epoch {epoch+1:>3d} | Accuracy {accuracy:>2.2%} |")


# %%
layer: nn.Linear = next(filter(lambda l: isinstance(l, nn.Linear), mlp.layers))
filters = layer.w.T
tiles(filters.reshape(16, -1, 28, 28))

# %%
