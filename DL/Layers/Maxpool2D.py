import numpy as np
from .interface import Layer

class MaxPool2D(Layer):
    def __init__(self, pool_size=2, stride=2):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        self.input = x
        batch_size, channels, h_in, w_in = x.shape
        pool = self.pool_size
        stride = self.stride

        h_out = (h_in - pool) // stride + 1
        w_out = (w_in - pool) // stride + 1

        out = np.zeros((batch_size, channels, h_out, w_out))
        self.max_indices = {}

        for b in range(batch_size):
            for c in range(channels):
                for i in range(h_out):
                    for j in range(w_out):
                        i0 = i * stride
                        j0 = j * stride
                        region = x[b, c, i0:i0+pool, j0:j0+pool]
                        max_val = np.max(region)
                        out[b, c, i, j] = max_val
                        max_pos = np.unravel_index(np.argmax(region), region.shape)
                        self.max_indices[(b, c, i, j)] = (i0 + max_pos[0], j0 + max_pos[1])
        return out

    def backward(self, grad_output):
        dx = np.zeros_like(self.input)

        for (b, c, i, j), (mi, mj) in self.max_indices.items():
            dx[b, c, mi, mj] += grad_output[b, c, i, j]

        return dx
