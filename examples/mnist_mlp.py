import numpy as np

from math import ceil
from tqdm import trange
from mnists import MNIST

from tinynn import nn
from tinynn.utils import chunks, tiles, softmax

# Load MNIST dataset
# ------------------

# Training set
X_train = MNIST().train_images().astype(np.float32) / 255.0
X_train = X_train.reshape(-1, 28 * 28)
y_train = MNIST().train_labels()

permute = np.random.permutation(len(X_train))
X_train = X_train[permute, ...]
y_train = y_train[permute, ...]

# Test set
X_test = MNIST().test_images().astype(np.float32) / 255.0
X_test = X_test.reshape(-1, 28 * 28)
y_test = MNIST().test_images()

# Define MLP model
# ---------------
params = {
    "lr": 0.15,
    "momentum": 0.5,
    "l2_penalty": None,
    "weight_limit": 2.0,
    "init_method": "He",
}

mlp = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(vsize=784, hsize=512, **params),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(vsize=512, hsize=256, **params),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(vsize=256, hsize=10, **params),
)

# Training loop
# -------------
batch_size = 512
num_epochs = 50


def scheduler(epoch: int, lr: float, momentum: float) -> tuple[float, float]:
    return lr, momentum


for epoch in (pbar := trange(num_epochs)):
    dataloader = chunks(X_train, y_train, size=batch_size)
    total = ceil(len(X_train) / batch_size)

    # --- Apply learning rate and momentum scheduler
    for layer in filter(lambda l: isinstance(l, nn.Linear), mlp.layers):
        layer: nn.Linear
        layer.lr, layer.momentum = scheduler(epoch, layer.lr, layer.momentum)

    # --- Training phase
    for X_batch, y_batch in dataloader:
        # --- Forward pass
        logits = mlp.forward(X_batch, training=True)

        # --- Compute gradient of the loss function w.r.t. the outputs of the network, assuming that
        #     the loss function is the cross entropy
        grad_y = 1 / batch_size * (softmax(logits) - y_batch)

        # --- Backward pass
        mlp.backward(X_batch, grad_y)

    # --- Validation phase
    logits = mlp.forward(X_test, training=False)
    accuracy = np.sum(np.argmax(logits, axis=1) == y_test)

    # --- Log progress
    pbar.set_description(f"Epoch {epoch+1:>3d}\tTest accuracy {accuracy:.2%}\t")
