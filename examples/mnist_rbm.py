import numpy as np
import matplotlib.pyplot as plt

from math import ceil
from tqdm import tqdm
from tinynn import nn
from mnists import MNIST
from numpy.typing import NDArray


def chunks(*arrays: NDArray, size: int):
    n = len(arrays[0])
    assert all(len(arr) == n for arr in arrays)
    return (tuple(arr[i : i + size] for arr in arrays) for i in range(0, n, size))


def tiles(examples: NDArray):
    space = 2
    rows, cols, h, w = examples.shape

    img_matrix = np.empty(shape=(rows * (h + space) - space, cols * (h + space) - space))
    img_matrix.fill(np.nan)

    for r in range(rows):
        for c in range(cols):
            x_0 = r * (h + space)
            y_0 = c * (w + space)
            ex_min = np.min(examples[r, c])
            ex_max = np.max(examples[r, c])
            img_matrix[x_0 : x_0 + h, y_0 : y_0 + w] = (examples[r, c] - ex_min) / (ex_max - ex_min)

    plt.matshow(img_matrix, cmap="gray", interpolation="none")
    plt.axis("off")
    plt.show()


mnist = MNIST()
X_train = mnist.train_images().astype(np.float32) / 255.0
np.random.shuffle(X_train)
X_train = X_train
X_train = X_train.reshape(-1, 28 * 28)

rbm = nn.RBM(
    vsize=28 * 28,
    hsize=256,
    pc_size=512,
    v_activation=nn.sigmoid,
    h_activation=nn.sigmoid,
    lr=0.01,
    momentum=0.5,
    init_method="Xavier",
    l1_penalty=None,
    l2_penalty=None,
    weight_limit=None,
)

batch_size = 256
total = ceil(len(X_train) / batch_size)

for epoch in range(50):
    for (X_batch,) in tqdm(chunks(X_train, size=batch_size), desc=f"Epoch {epoch+1:>3d}", total=total):
        nn.pcd_step(rbm, X_batch)

v = np.random.random(size=(12 * 24, 28 * 28))
samples = rbm.sample(v, steps=1_000)
tiles(samples.reshape(-1, 24, 28, 28))
