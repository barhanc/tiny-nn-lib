import time
import torch
import torch.nn
import tinynn.nn
import numpy as np

torch.manual_seed(42)
torch.set_default_device("cpu")

in_channels = 3
out_channels = 10
kernel_size = (3, 3)
stride = (3, 3)
padding = (1, 1)

conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dtype=torch.float64)
params = conv1.parameters()
w = next(params).detach()
b = next(params).detach()

conv2 = tinynn.nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, None, None, "Xavier")
conv2.w = w.numpy().reshape(-1, in_channels * kernel_size[0] * kernel_size[1])
conv2.b = b.numpy().reshape(-1, 1)

x = torch.randn(128, in_channels, 64, 64, dtype=torch.float64)
t = time.perf_counter()
y1 = conv1(x).detach().numpy()
t = time.perf_counter() - t
print("Torch ", t)
t = time.perf_counter()
y2 = conv2.forward(x.numpy(), training=False)
t = time.perf_counter() - t
print("Numpy ", t)
print(np.allclose(y1, y2))
