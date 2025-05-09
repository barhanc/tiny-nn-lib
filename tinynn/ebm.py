import numpy as np

from itertools import pairwise
from typing import Optional, Literal, Callable
from numpy.typing import NDArray

from .utils import *

__all__ = [
    "RBM",
    "DBN",
    "cdk_step",
    "pcd_step",
]


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
        self.b = zeros(self.vsize)
        self.c = zeros(self.hsize)

        # Velocity (momentum) tensor initialization
        self.m_w = zeros(self.vsize, self.hsize)
        self.m_b = zeros(self.vsize)
        self.m_c = zeros(self.hsize)

        # Persistent chain initialization
        if self.pc_size:
            self.pc = zeros(self.pc_size, self.hsize)

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
    def __init__(self, *rbms: RBM):
        for rbm1, rbm2 in pairwise(rbms):
            assert rbm1.hsize == rbm2.vsize
        self.layers: tuple[RBM, ...] = rbms

    def propagate_up(self, v: NDArray, n_layers: int) -> NDArray:
        assert 0 <= n_layers < len(self.layers)
        for i in range(n_layers):
            v = self.layers[i].probas_h(v, sample=False)
        return v

    def propagate_dn(self, h: NDArray, n_layers: int) -> NDArray:
        assert 0 <= n_layers < len(self.layers)
        for i in reversed(range(n_layers)):
            h = self.layers[i].probas_v(h, sample=False)
        return h

    def sample(self, v: NDArray, steps: int) -> NDArray:
        i = len(self.layers) - 1
        v = self.propagate_up(v, i)
        h = self.layers[i].sample(v, steps)
        return self.propagate_dn(h, i)


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
