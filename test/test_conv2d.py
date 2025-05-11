import time
import torch
import torch.nn
import tinynn.nn
import numpy as np

# torch.manual_seed(42)
torch.set_default_device("cpu")

batch_size = 5
in_channels = 3
out_channels = 2
h_in = 8
w_in = 8
kernel_size = (3, 3)
stride = (3, 3)
padding = (1, 1)

h_out = int(1 + (h_in + 2 * padding[0] - kernel_size[0]) / stride[0])
w_out = int(1 + (w_in + 2 * padding[1] - kernel_size[1]) / stride[1])


conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dtype=torch.float64)
params = conv1.parameters()
w = next(params).detach()
b = next(params).detach()

conv2 = tinynn.nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, None, None, "Xavier")
conv2.w = w.numpy().reshape(-1, in_channels * kernel_size[0] * kernel_size[1])
conv2.b = b.numpy().reshape(-1, 1)

x = torch.randn(batch_size, in_channels, h_in, w_in, dtype=torch.float64)
t = time.perf_counter()
y1 = conv1(x).detach().numpy()
t = time.perf_counter() - t
print("Torch ", t)
t = time.perf_counter()
y2 = conv2.forward(x.numpy(), training=False)
t = time.perf_counter() - t
print("Numpy ", t)
assert np.allclose(y1, y2)

# Compute numeric derivative

# ∂Loss/∂x 
randi = np.random.randint
grad_y = np.random.random((batch_size, out_channels, h_out, w_out))
ϵ = 1e-6
eps = np.zeros(x.shape)
idx = (randi(batch_size), randi(out_channels), randi(h_out), randi(w_out))
eps[idx] = ϵ

y_p = conv2.forward(x.numpy() + eps, training=False)
y_n = conv2.forward(x.numpy() - eps, training=False)
print(np.sum(grad_y * (y_p - y_n) / (2 * ϵ)))

conv2.forward(x.numpy(), training=True)
grad_x = conv2.backward(x.numpy(), grad_y)
print(grad_x[idx])

# ∂Loss/∂w
eps = np.zeros(conv2.w.shape)
idx = (randi(conv2.w.shape[0]), randi(conv2.w.shape[1]))
eps[idx] = ϵ
old_w = conv2.w.copy()

conv2.w = old_w + eps
y_p = conv2.forward(x.numpy(), training=False)
conv2.w = old_w - eps
y_n = conv2.forward(x.numpy(), training=False)
print(np.sum(grad_y * (y_p - y_n) / (2 * ϵ)))

conv2.w = old_w
conv2.forward(x.numpy(), training=True)
conv2.backward(x.numpy(), grad_y)
print(conv2.grad_w[idx])

# ∂Loss/∂b
eps = np.zeros(conv2.b.shape)
idx = (randi(conv2.b.shape[0]), randi(conv2.b.shape[1]))
eps[idx] = ϵ
old_b = conv2.b.copy()

conv2.b = old_b + eps
y_p = conv2.forward(x.numpy(), training=False)
conv2.b = old_b - eps
y_n = conv2.forward(x.numpy(), training=False)
print(np.sum(grad_y * (y_p - y_n) / (2 * ϵ)))

conv2.b = old_b
conv2.forward(x.numpy(), training=True)
conv2.backward(x.numpy(), grad_y)
print(conv2.grad_b[idx])