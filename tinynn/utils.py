import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def zeros(*dims: int) -> NDArray[np.float32]:
    return np.zeros(shape=tuple(dims), dtype=np.float32)


def rand(*dims: int) -> NDArray[np.float32]:
    return np.random.rand(*dims).astype(np.float32)


def randn(*dims: int) -> NDArray[np.float32]:
    return np.random.randn(*dims).astype(np.float32)


def chunks(arr: NDArray, size: int):
    return (arr[i : i + size] for i in range(0, len(arr), size))


def onehot(y: NDArray) -> NDArray:
    one_hot = zeros(y.shape[0], np.max(y) + 1)
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot.astype(np.float32)


def tiles(imgs: NDArray):
    space = 2
    rows, cols, h, w = imgs.shape

    img_matrix = np.empty(shape=(rows * (h + space) - space, cols * (h + space) - space))
    img_matrix.fill(np.nan)

    for r in range(rows):
        for c in range(cols):
            x = r * (h + space)
            y = c * (w + space)
            m = np.min(imgs[r, c])
            M = np.max(imgs[r, c])
            img_matrix[x : x + h, y : y + w] = (imgs[r, c] - m) / (M - m)

    plt.matshow(img_matrix, cmap="gray", interpolation="none")
    plt.axis("off")
    plt.show()


def limit_weights(w: NDArray, limit: float) -> NDArray:
    """Applies the norm limit regularization to weights `w`."""
    if limit == 0:
        return w
    norm = np.linalg.norm(w, ord=2, axis=0)
    mask = norm > limit
    return w * (mask * (limit / norm) + (~mask) * 1.0)


def sigmoid(x: NDArray, sample: bool = False) -> NDArray:
    σ = 1.0 / (1.0 + np.exp(-x))
    if sample:
        return σ > rand(*σ.shape)
    return σ


def relu(x: NDArray, sample: bool = False) -> NDArray:
    if sample:
        return np.maximum(0, x + np.sqrt(sigmoid(x)) * randn(*x.shape))
    return np.maximum(0, x)


def softmax(x: NDArray) -> NDArray:
    m = x.max(axis=1, keepdims=True)
    y: NDArray = np.exp(x - m)
    return y / y.sum(axis=1, keepdims=True)
