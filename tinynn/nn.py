import numpy as np

from abc import abstractmethod, ABC
from typing import Optional, Literal
from numpy.typing import NDArray

from .utils import zeros, limit_weights, rand


class Layer(ABC):
    """
    Interface for any differentiable parametrized tensor function with parameters `θ` that takes a
    single tensor `x` and returns a single tensor `y = Layer(x; θ)`.
    """

    # Outputs of the layer after forward pass
    y: Optional[NDArray]

    @abstractmethod
    def reset(self):
        """Initialize the layer."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: NDArray, training: bool) -> NDArray:
        """Propagate the input `x` forward through the layer and save the output in `self.y`."""
        raise NotImplementedError

    @abstractmethod
    def backward(self, x: NDArray, grad_y: NDArray) -> NDArray:
        """
        Given layer input `x` and ∂Loss/∂y (`grad_y`)
            * compute ∂Loss/∂x and ∂Loss/∂θ (where θ are the layer's params);
            * update parameters using momentum SGD;
            * return ∂Loss/∂x;

        NOTE: We assume that the layers are connected in a simple path (i.e. the computation graph
        is linear) and thus we don't have to keep and accumulate the gradients ∂Loss/∂y in the layer
        itself, but can instead just dynamically pass ∂Loss/∂y while traversing this linear
        computation graph.
        """
        raise NotImplementedError


class Sequential(Layer):
    def __init__(self, *layers: Layer):
        self.layers: tuple[Layer, ...] = layers
        self.reset()

    def reset(self):
        self.y: Optional[NDArray] = None
        for layer in self.layers:
            layer.reset()

    def forward(self, x: NDArray, training: bool) -> NDArray:
        for layer in self.layers:
            x = layer.forward(x, training)
        self.y = x.copy()
        return self.y

    def backward(self, x: NDArray, grad_y: NDArray) -> NDArray:
        for i in reversed(range(len(self.layers))):
            # --- Get input to the i-th layer i.e. output of the (i-1)-th layer (or `x` if i=0)
            y_prev = self.layers[i - 1].y if i > 0 else x
            # --- Propagate the ∂Loss/∂y backward through layer `i` and update params
            grad_y = self.layers[i].backward(y_prev, grad_y)
        return grad_y


class Sigmoid(Layer):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y: Optional[NDArray] = None

    def forward(self, x: NDArray, training: bool) -> NDArray:
        self.y = 1.0 / (1.0 + np.exp(-x))
        return self.y

    def backward(self, x: NDArray, grad_y: NDArray) -> NDArray:
        # --- Compute ∂Loss/∂x
        grad_x = grad_y * (self.y * (1.0 - self.y))
        # --- Propagate ∂Loss/∂x backward
        return grad_x


class ReLU(Layer):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y: Optional[NDArray] = None

    def forward(self, x: NDArray, training: bool) -> NDArray:
        self.y = np.maximum(0, x)
        return self.y

    def backward(self, x: NDArray, grad_y: NDArray) -> NDArray:
        # --- Compute ∂Loss/∂x
        grad_x = grad_y * (x > 0).astype(np.float32)
        # --- Propagate ∂Loss/∂x backward
        return grad_x


class Dropout(Layer):
    def __init__(self, p: float):
        assert 0 < p <= 1
        self.p: float = p
        self.reset()

    def reset(self):
        self.y: Optional[NDArray] = None
        # Dropout mask, required for backpropagation
        self.mask: Optional[NDArray] = None

    def forward(self, x: NDArray, training: bool) -> NDArray:
        if training:
            # --- Save the dropout mask for backward pass
            self.mask = (rand(*x.shape) > self.p).astype(np.float32)
            self.y = x * self.mask
        else:
            self.y = x * (1 - self.p)
        return self.y

    def backward(self, x: NDArray, grad_y: NDArray) -> NDArray:
        # --- Compute ∂Loss/∂x
        grad_x = grad_y * self.mask
        # --- Propagate ∂Loss/∂x backward
        return grad_x


class Linear(Layer):
    def __init__(
        self,
        vsize: int,
        hsize: int,
        lr: float,
        momentum: float,
        l2_penalty: Optional[float],
        weight_limit: Optional[float],
        init_method: Literal["Xavier", "He"],
    ):
        self.vsize: int = vsize
        self.hsize: int = hsize
        self.lr: float = lr
        self.momentum: float = momentum
        self.l2_penalty: Optional[float] = l2_penalty
        self.weight_limit: Optional[float] = weight_limit
        self.init_method: Literal["Xavier", "He"] = init_method
        self.reset()

    def reset(self):
        # Weights initialization
        match self.init_method:
            case "Xavier":
                scale = np.sqrt(6 / (self.vsize + self.hsize))
                self.w = np.random.uniform(-scale, +scale, size=(self.vsize, self.hsize)).astype(np.float32)
            case "He":
                scale = np.sqrt(4 / (self.vsize + self.hsize))
                self.w = np.random.normal(0, scale, size=(self.vsize, self.hsize)).astype(np.float32)
            case _:
                raise ValueError(f"Unrecognised {self.init_method=}")

        # Bias initialization
        self.b = zeros(self.hsize)

        # Velocity (momentum) tensors initialization
        self.m_w = zeros(self.vsize, self.hsize)
        self.m_b = zeros(self.hsize)

        # Outputs
        self.y: Optional[NDArray] = None

    def forward(self, x: NDArray, training: bool) -> NDArray:
        self.y = self.b + x @ self.w
        return self.y

    def backward(self, x: NDArray, grad_y: NDArray) -> NDArray:
        # --- Compute ∂Loss/∂x
        grad_x = grad_y @ self.w.T

        # --- Compute  ∂Loss/∂w and ∂Loss/∂b
        grad_w = x.T @ grad_y
        grad_b = grad_y.sum(axis=0)

        # --- Apply L2 regularization
        if self.l2_penalty:
            grad_w += self.l2_penalty * self.w

        # --- Update params
        self.m_w = self.momentum * self.m_w - self.lr * grad_w
        self.m_b = self.momentum * self.m_b - self.lr * grad_b

        self.w += self.m_w
        self.b += self.m_b

        # --- Apply weight limit normalization
        if self.weight_limit:
            self.w = limit_weights(self.w, self.weight_limit)

        # --- Propagate ∂Loss/∂x backward
        return grad_x


class Conv2D(Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        strides: tuple[int, int],
        padding: tuple[int, int],
        lr: float,
        momentum: float,
        init_method: Literal["Xavier", "He"],
    ):
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: tuple[int, int] = kernel_size
        self.strides: tuple[int, int] = strides
        self.padding: tuple[int, int] = padding

        self.lr: float = lr
        self.momentum: float = momentum
        self.init_method: Literal["Xavier", "He"] = init_method

        self.reset()

    def reset(self):
        # Weights initialization
        nrow = self.out_channels
        ncol = self.in_channels * self.kernel_size[0] * self.kernel_size[1]

        match self.init_method:
            case "Xavier":
                scale = np.sqrt(6 / (nrow + ncol))
                self.w = np.random.uniform(-scale, +scale, size=(nrow, ncol)).astype(np.float32)
            case "He":
                scale = np.sqrt(4 / (nrow + ncol))
                self.w = np.random.normal(0, scale, size=(nrow, ncol)).astype(np.float32)
            case _:
                raise ValueError(f"Unrecognised {self.init_method=}")

        # Bias initialization
        self.b = zeros(self.out_channels, 1)

        # Velocity (momentum) tensors initialization
        self.m_w = zeros(nrow, ncol)
        self.m_b = zeros(nrow, 1)

        # Outputs
        self.y: Optional[NDArray] = None

        # TODO: description
        self.indices: Optional[tuple[NDArray, NDArray, NDArray]] = None

    def _pad(self, x: NDArray) -> NDArray:
        assert len(x.shape) == 4
        pad_h, pad_w = self.padding
        return np.pad(x, pad_width=[(0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)])

    def _unpad(self, x: NDArray) -> NDArray:
        assert len(x.shape) == 4
        *_, H, W = x.shape
        pad_h, pad_w = self.padding
        return x[:, :, pad_h : H - pad_h, pad_w : W - pad_w]

    def forward(self, x: NDArray, training: bool) -> NDArray:
        assert len(x.shape) == 4
        assert x.shape[1] == self.in_channels

        _, C_in, H_in, W_in = x.shape
        H_ker, W_ker = self.kernel_size

        C_out = self.out_channels
        H_out = int(1 + (H_in + 2 * self.padding[0] - self.kernel_size[0]) / self.strides[0])
        W_out = int(1 + (W_in + 2 * self.padding[1] - self.kernel_size[1]) / self.strides[1])

        idx_c, idx_h_ker, idx_w_ker = np.indices((C_in, H_ker, W_ker)).reshape(3, -1)
        idx_h_out, idx_w_out = np.indices((H_out, W_out)).reshape(2, -1)

        idx_c = idx_c.reshape(-1, 1)
        idx_h = idx_h_ker.reshape(-1, 1) + self.strides[0] * idx_h_out
        idx_w = idx_w_ker.reshape(-1, 1) + self.strides[1] * idx_w_out
        self.indices = (idx_c, idx_h, idx_w)

        x = self._pad(x)
        x = x[(..., *self.indices)]
        x = self.b + self.w @ x

        self.y = x.reshape(-1, C_out, H_out, W_out)

        return self.y

    def backward(self, x: NDArray, grad_y: NDArray) -> NDArray:
        _, C_in, H_in, W_in = x.shape
        C_out = self.out_channels
        H_out = int(1 + (H_in + 2 * self.padding[0] - self.kernel_size[0]) / self.strides[0])
        W_out = int(1 + (W_in + 2 * self.padding[1] - self.kernel_size[1]) / self.strides[1])

        grad_x = grad_y.reshape(-1, C_out, H_out * W_out)

        grad_w = (grad_x @ x.transpose(0, 2, 1)).sum(axis=0)
        grad_b = grad_x.sum(axis=(0, 2)).reshape(-1, 1)
        grad_x = self.w.T @ grad_x

        # TODO
        raise NotImplementedError

        grad_x = self._unpad(grad_x)
