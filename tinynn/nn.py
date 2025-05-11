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
            # Get input to the i-th layer i.e. output of the (i-1)-th layer (or `x` if i=0)
            y_prev = self.layers[i - 1].y if i > 0 else x
            # Propagate the ∂Loss/∂y backward through layer `i` and update params
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
        # Compute ∂Loss/∂x
        grad_x = grad_y * (self.y * (1.0 - self.y))
        # Propagate ∂Loss/∂x backward
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
        # Compute ∂Loss/∂x
        grad_x = grad_y * (x > 0).astype(np.float32)
        # Propagate ∂Loss/∂x backward
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
            # Save the dropout mask for backward pass
            self.mask = (rand(*x.shape) > self.p).astype(np.float32)
            self.y = x * self.mask
        else:
            self.y = x * (1 - self.p)
        return self.y

    def backward(self, x: NDArray, grad_y: NDArray) -> NDArray:
        # Compute ∂Loss/∂x
        grad_x = grad_y * self.mask
        # Propagate ∂Loss/∂x backward
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
        # Compute ∂Loss/∂x
        grad_x = grad_y @ self.w.T

        # Compute  ∂Loss/∂w and ∂Loss/∂b
        grad_w = x.T @ grad_y
        grad_b = grad_y.sum(axis=0)

        # Apply L2 regularization
        if self.l2_penalty:
            grad_w += self.l2_penalty * self.w

        # Update params
        self.m_w = self.momentum * self.m_w - self.lr * grad_w
        self.m_b = self.momentum * self.m_b - self.lr * grad_b

        self.w += self.m_w
        self.b += self.m_b

        # Apply weight limit normalization
        if self.weight_limit:
            self.w = limit_weights(self.w, self.weight_limit)

        # Propagate ∂Loss/∂x backward
        return grad_x


class Conv2D(Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        strides: int | tuple[int, int],
        padding: int | tuple[int, int],
        lr: float,
        momentum: float,
        init_method: Literal["Xavier", "He"],
    ):
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: tuple[int, int] = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.strides: tuple[int, int] = (strides, strides) if isinstance(strides, int) else strides
        self.padding: tuple[int, int] = (padding, padding) if isinstance(padding, int) else padding

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
        eps = 1e-2  # Initialize biases to small positive values (for ReLU)
        self.b = zeros(self.out_channels, 1) + eps

        # Velocity (momentum) tensors initialization
        self.m_w = zeros(nrow, ncol)
        self.m_b = zeros(nrow, 1)

        # Outputs
        self.y: Optional[NDArray] = None

        # Cached indices for the im2col/col2im transformation
        self._indices: Optional[tuple[NDArray, NDArray, NDArray]] = None
        self._indices_ravel: Optional[NDArray] = None
        # Input tensor after im2col transformation
        self._x2col: Optional[NDArray] = None
        # Input tensor shape from the last iteration of, respectively the forward and backward
        self._dims_x_f: Optional[tuple[int, int, int, int]] = None
        self._dims_x_b: Optional[tuple[int, int, int, int]] = None

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

        B, C_in, H_in, W_in = x.shape
        C_out = self.out_channels
        H_out = int(1 + (H_in + 2 * self.padding[0] - self.kernel_size[0]) / self.strides[0])
        W_out = int(1 + (W_in + 2 * self.padding[1] - self.kernel_size[1]) / self.strides[1])

        if self._dims_x_f is None or self._dims_x_f[1:] != x.shape[1:]:
            use_cached_indices = False
            self._dims_x_f = x.shape
        else:
            use_cached_indices = True

        # Compute indices for im2col transformation
        if not use_cached_indices:
            idx_c, idx_h_ker, idx_w_ker = np.indices((C_in, *self.kernel_size)).reshape(3, -1)
            idx_h_out, idx_w_out = np.indices((H_out, W_out)).reshape(2, -1)

            idx_c = idx_c.reshape(-1, 1)
            idx_h = idx_h_ker.reshape(-1, 1) + self.strides[0] * idx_h_out
            idx_w = idx_w_ker.reshape(-1, 1) + self.strides[1] * idx_w_out

            self.dims_x_forward = x.shape
            self._indices = (idx_c, idx_h, idx_w)

        # Apply padding transformation
        x = self._pad(x)

        # Apply im2col transformation
        x = x[(..., *self._indices)]  # (B, C_in * H_ker * W_ker, H_out * W_out)

        # Apply affine transformation
        # NOTE: We use these transposes and reshapes to leverage the optimized BLAS GEMM as batched
        # matmul in Numpy (contrary to e.g. PyTorch) with signature (N,K), (B,K,M) -> (B,N,M) is
        # significantly slower than flattening the batch dimension nad using matmul with signature
        # (N,K), (K,B*M) -> (N,B*M)
        #
        # fmt:off
        x = x.transpose(1, 0, 2)              # (C_in * H_ker * W_ker, B, H_out * W_out)
        x = x.reshape(-1, B * H_out * W_out)  # (C_in * H_ker * W_ker, B * H_out * W_out)
        
        self._x2col = x
        x = self.b + self.w @ x               # (C_out, B * H_out * W_ou)
        
        x = x.reshape(-1, B, H_out * W_out)   # (C_out, B, H_out * W_ou)
        x = x.transpose(1, 0, 2)              # (B, C_out, H_out * W_ou)
        # fmt:on

        self.y = x.reshape(-1, C_out, H_out, W_out)

        return self.y

    def backward(self, x: NDArray, grad_y: NDArray) -> NDArray:
        assert len(grad_y.shape) == 4
        assert len(x.shape) == 4
        assert x.shape[1] == self.in_channels
        assert grad_y.shape[1] == self.out_channels

        B, C_in, H_in, W_in = x.shape
        C_out = self.out_channels
        H_out = int(1 + (H_in + 2 * self.padding[0] - self.kernel_size[0]) / self.strides[0])
        W_out = int(1 + (W_in + 2 * self.padding[1] - self.kernel_size[1]) / self.strides[1])

        if self._dims_x_b is None or x.shape != self._dims_x_b:
            use_cached_indices = False
            self._dims_x_b = x.shape
        else:
            use_cached_indices = True

        # --- Compute ∂Loss/∂x, ∂Loss/∂w and ∂Loss/∂b

        # Backpropagate through reshape operation
        grad_y = grad_y.reshape(-1, C_out, H_out * W_out)  # (B, C_out, H_out*W_out)

        # Backpropagate through affine transformation
        # fmt:off
        x = self._x2col

        grad_y = grad_y.transpose(1, 0, 2)                 # (C_out, B, H_out * W_out)
        grad_y = grad_y.reshape(-1, B * H_out * W_out)     # (C_out, B * H_out * W_out)

        grad_w = grad_y @ x.T
        grad_b = grad_y.sum(axis=1, keepdims=True)
        grad_y = self.w.T @ grad_y                         # (C_in * H_ker * W_ker, B * H_out * W_out)

        grad_y = grad_y.reshape(-1, B, H_out * W_out)      # (C_in * H_ker * W_ker, B, H_out * W_out)
        grad_y = grad_y.transpose(1, 0, 2)                 # (B, C_in * H_ker * W_ker, H_out * W_out)
        # fmt:on

        # Backpropagate through im2col operation
        dims_x = (B, C_in, H_in + 2 * self.padding[0], W_in + 2 * self.padding[1])

        if not use_cached_indices:
            multi_index = (np.arange(B).reshape(-1, 1, 1), *self._indices)
            self._indices_ravel = np.ravel_multi_index(multi_index, dims=dims_x)

        grad_x = np.bincount(self._indices_ravel.flatten(), weights=grad_y.flatten())
        grad_x = grad_x.reshape(dims_x)

        # Backpropagate through padding operation
        grad_x = self._unpad(grad_x)

        # --- Update params
        self.m_w = self.momentum * self.m_w - self.lr * grad_w
        self.m_b = self.momentum * self.m_b - self.lr * grad_b

        self.w += self.m_w
        self.b += self.m_b

        # Propagate ∂Loss/∂x backward
        return grad_x


class Flatten(Layer):
    def __init__(self, start_dim: int = 1):
        self.start_dim: int = start_dim
        self.reset()

    def reset(self):
        self.y: Optional[NDArray] = None

    def forward(self, x: NDArray, training: bool) -> NDArray:
        self.y = x.reshape(*x.shape[: self.start_dim], -1)
        return self.y

    def backward(self, x: NDArray, grad_y: NDArray) -> NDArray:
        # Compute ∂Loss/∂x
        grad_x = grad_y.reshape(x.shape)
        # Propagate ∂Loss/∂x backward
        return grad_x
