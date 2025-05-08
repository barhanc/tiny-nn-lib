import numpy as np

from itertools import pairwise
from abc import abstractmethod, ABC
from typing import Optional, Literal, Callable, Sequence
from numpy.typing import NDArray


def _zeros(*dims: int) -> NDArray:
    return np.zeros(shape=tuple(dims), dtype=np.float32)


def _rand(*dims: int) -> NDArray:
    return np.random.rand(*dims).astype(np.float32)


def _randn(*dims: int) -> NDArray:
    return np.random.randn(*dims).astype(np.float32)


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
        return σ > _rand(*σ.shape)
    return σ


def relu(x: NDArray, sample: bool = False) -> NDArray:
    if sample:
        return np.maximum(0, x + sigmoid(x) * _randn(*x.shape))
    return np.maximum(0, x)


def softmax(x: NDArray) -> NDArray:
    m = x.max(axis=1, keepdims=True)
    y: NDArray = np.exp(x - m)
    return y / y.sum(axis=1, keepdims=True)


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
        self.y = x
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

        # Biases initialization
        self.b = _zeros(self.hsize)

        # Velocity (momentum) tensors initialization
        self.m_w = _zeros(self.vsize, self.hsize)
        self.m_b = _zeros(self.hsize)

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
            self.mask = (_rand(*x.shape) > self.p).astype(np.float32)
            self.y = x * self.mask
        else:
            self.y = x * (1 - self.p)
        return self.y

    def backward(self, x: NDArray, grad_y: NDArray) -> NDArray:
        # --- Compute ∂Loss/∂x
        grad_x = grad_y * self.mask
        # --- Propagate ∂Loss/∂x backward
        return grad_x


class RBM:
    def __init__(
        self,
        vsize: int,
        hsize: int,
        pc_size: Optional[int],
        v_activation: Callable[[NDArray, bool], NDArray],
        h_activation: Callable[[NDArray, bool], NDArray],
        lr: float,
        momentum: float,
        l1_penalty: Optional[float],
        l2_penalty: Optional[float],
        weight_limit: Optional[float],
        init_method: Literal["Xavier", "He"],
    ):
        self.vsize: int = vsize
        self.hsize: int = hsize
        self.pc_size: Optional[int] = pc_size
        self.v_activation: Callable[[NDArray, bool], NDArray] = v_activation
        self.h_activation: Callable[[NDArray, bool], NDArray] = h_activation

        # Training hyper-params
        self.lr: float = lr
        self.momentum: float = momentum
        self.l1_penalty: Optional[float] = l1_penalty
        self.l2_penalty: Optional[float] = l2_penalty
        self.weight_limit: Optional[float] = weight_limit

        # Initialize
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
        self.b = _zeros(self.vsize)
        self.c = _zeros(self.hsize)

        # Velocity (momentum) tensor initialization
        self.m_w = _zeros(self.vsize, self.hsize)
        self.m_b = _zeros(self.vsize)
        self.m_c = _zeros(self.hsize)

        # Persistent chain initialization
        if self.pc_size:
            self.pc = _zeros(self.pc_size, self.hsize)

    def probas_v(self, h: NDArray, sample: bool) -> NDArray:
        return self.v_activation(self.b + h @ self.w.T, sample=sample)

    def probas_h(self, v: NDArray, sample: bool) -> NDArray:
        return self.h_activation(self.c + v @ self.w, sample=sample)

    def sample(self, v: NDArray, steps: int) -> NDArray:
        # --- Gibbs sampling
        for k in range(steps):
            h = self.probas_h(v, sample=True)
            v = self.probas_v(h, sample=(k < steps - 1))
        return v


class DBN:
    def __init__(self, rbms: Sequence[RBM]):
        for rbm1, rbm2 in pairwise(rbms):
            assert rbm1.hsize == rbm2.vsize
        self.rbms: Sequence[RBM] = rbms

    def propagate_up(self, v: NDArray, n_layers: int) -> NDArray:
        assert 0 <= n_layers < len(self.rbms)
        for i in range(n_layers):
            v = self.rbms[i].probas_h(v, sample=False)
        return v

    def propagate_dn(self, h: NDArray, n_layers: int) -> NDArray:
        assert 0 <= n_layers < len(self.rbms)
        for i in reversed(range(n_layers)):
            h = self.rbms[i].probas_v(h, sample=False)
        return h


def cdk_step(rbm: RBM, minibatch: NDArray, k: int = 1):
    batch_size = minibatch.shape[0]
    v = minibatch

    # Compute gradients
    # -----------------

    # Positive phase
    σ = rbm.probas_h(v, sample=False)

    grad_w = -1 / batch_size * (v.T @ σ)
    grad_b = -1 / batch_size * (v.sum(axis=0))
    grad_c = -1 / batch_size * (σ.sum(axis=0))

    # Negative phase

    # --- Gibbs sampling
    h = rbm.probas_h(v, sample=True)
    v = rbm.probas_v(h, sample=True)
    for _ in range(k - 1):
        h = rbm.probas_h(v, sample=True)
        v = rbm.probas_v(h, sample=True)

    # --- Negative gradient estimation
    σ = rbm.probas_h(v, sample=False)

    grad_w += 1 / batch_size * (v.T @ σ)
    grad_b += 1 / batch_size * (v.sum(axis=0))
    grad_c += 1 / batch_size * (σ.sum(axis=0))

    # Update params
    # -------------

    # --- Apply L1 / L2 regularization
    if rbm.l1_penalty:
        grad_w += rbm.l1_penalty * np.sign(rbm.w)
    if rbm.l2_penalty:
        grad_w += rbm.l2_penalty * rbm.w

    rbm.m_w = rbm.momentum * rbm.m_w - rbm.lr * grad_w
    rbm.m_b = rbm.momentum * rbm.m_b - rbm.lr * grad_b
    rbm.m_c = rbm.momentum * rbm.m_c - rbm.lr * grad_c

    rbm.w += rbm.m_w
    rbm.b += rbm.m_b
    rbm.c += rbm.m_c

    # --- Apply weight limit normalization
    if rbm.weight_limit:
        rbm.w = limit_weights(rbm.w, rbm.weight_limit)


def pcd_step(rbm: RBM, minibatch: NDArray, k: int = 1):
    batch_size = minibatch.shape[0]
    v = minibatch

    # Compute gradients
    # -----------------

    # Positive phase
    σ = rbm.probas_h(v, sample=False)

    grad_w = -1 / batch_size * (v.T @ σ)
    grad_b = -1 / batch_size * (v.sum(axis=0))
    grad_c = -1 / batch_size * (σ.sum(axis=0))

    # Negative phase

    # --- Gibbs sampling
    h = rbm.pc  # Start from persistent chain
    v = rbm.probas_v(h, sample=True)
    for _ in range(k - 1):
        h = rbm.probas_h(v, sample=True)
        v = rbm.probas_v(h, sample=True)

    # --- Negative gradient estimation
    σ = rbm.probas_h(v, sample=False)

    grad_w += 1 / rbm.pc_size * (v.T @ σ)
    grad_b += 1 / rbm.pc_size * (v.sum(axis=0))
    grad_c += 1 / rbm.pc_size * (σ.sum(axis=0))

    # --- Update persistent chain
    rbm.pc = rbm.probas_h(v, sample=True)

    # Update params
    # -------------

    # --- Apply L1 / L2 regularization
    if rbm.l1_penalty:
        grad_w += rbm.l1_penalty * np.sign(rbm.w)
    if rbm.l2_penalty:
        grad_w += rbm.l2_penalty * rbm.w

    rbm.m_w = rbm.momentum * rbm.m_w - rbm.lr * grad_w
    rbm.m_b = rbm.momentum * rbm.m_b - rbm.lr * grad_b
    rbm.m_c = rbm.momentum * rbm.m_c - rbm.lr * grad_c

    rbm.w += rbm.m_w
    rbm.b += rbm.m_b
    rbm.c += rbm.m_c

    # --- Apply weight limit normalization
    if rbm.weight_limit:
        rbm.w = limit_weights(rbm.w, rbm.weight_limit)
