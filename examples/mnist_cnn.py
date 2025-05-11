# %%
import numpy as np

from math import ceil
from tqdm import trange
from mnists import MNIST

from tinynn import nn
from tinynn.utils import chunks, onehot, softmax

np.random.seed(42)

# %%
# Load MNIST dataset
# ------------------
mnist = MNIST()

# Training set
X_train = mnist.train_images().astype(np.float32) / 255.0
X_train = X_train.reshape(-1, 1, 28, 28)
y_train = mnist.train_labels()
y_train = onehot(y_train)

size = 10_000
X_train = X_train[:size, ...]
y_train = y_train[:size, ...]

permute = np.random.permutation(len(X_train))
X_train = X_train[permute, ...]
y_train = y_train[permute, ...]

# Test set
X_test = mnist.test_images().astype(np.float32) / 255.0
X_test = X_test.reshape(-1, 1, 28, 28)
y_test = mnist.test_labels()

# %%
# Define MLP model
# ---------------
params = {
    "lin": {
        "lr": 0.15,
        "momentum": 0.5,
        "weight_limit": None,
        "l2_penalty": None,
        "init_method": "He",
    },
    "conv": {
        "lr": 0.05,
        "momentum": 0.5,
        "init_method": "He",
    },
}

model = nn.Sequential(
    nn.Conv2D(1, 8, 5, 1, 0, **params["conv"]),
    nn.ReLU(),
    nn.Conv2D(8, 8, 2, 2, 0, **params["conv"]),
    nn.ReLU(),
    nn.Conv2D(8, 16, 5, 1, 0, **params["conv"]),
    nn.ReLU(),
    nn.Conv2D(16, 16, 2, 2, 0, **params["conv"]),
    nn.Flatten(1),
    nn.Dropout(0.2),
    nn.Linear(4 * 4 * 16, 200, **params["lin"]),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(200, 10, **params["lin"]),
)

# %%
# Training loop
# -------------
batch_size = 128
num_epochs = 50

for epoch in (pbar := trange(num_epochs)):
    dataloader = chunks(X_train, y_train, size=batch_size)
    total = ceil(len(X_train) / batch_size)

    # Training phase
    for X_batch, y_batch in dataloader:
        # Forward pass
        logits = model.forward(X_batch, training=True)

        # Compute gradient of the loss function w.r.t. the outputs of the network, assuming that the
        # loss function is the cross entropy
        grad_y = 1 / len(X_batch) * (softmax(logits) - y_batch)

        # Backward pass
        model.backward(X_batch, grad_y)

    # Validation phase
    logits = model.forward(X_test, training=False)
    accuracy = 1 / len(y_test) * np.sum(np.argmax(logits, axis=1) == y_test)

    # Log progress
    pbar.set_description(f"Epoch {epoch+1:>3d} | Accuracy {accuracy:>2.2%} |")
