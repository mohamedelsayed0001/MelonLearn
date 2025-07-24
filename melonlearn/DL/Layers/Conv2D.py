import numpy as np
from .interface import Layer

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size,stride ,padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.params["W"] = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        self.params["b"] = np.zeros((out_channels, 1))

    def forward(self, x):
        self.input = x  
        batch_size, _, h_in, w_in = x.shape
        k = self.kernel_size
        h_out = (h_in - k + 2 * self.padding) // self.stride + 1
        w_out = (w_in - k + 2 * self.padding) // self.stride + 1

        out = np.zeros((batch_size, self.out_channels, h_out, w_out))

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(h_out):
                    for j in range(w_out):
                        i0 = i * self.stride
                        j0 = j * self.stride
                        region = x[b, :, i0:i0+k, j0:j0+k]
                        out[b, oc, i, j] = np.sum(region * self.params["W"][oc]) + self.params["b"][oc]
        return out

    def backward(self, grad_output):
        x = self.input
        W = self.params["W"]
        k = self.kernel_size
        batch_size, _, h_in, w_in = x.shape
        _, _, h_out, w_out = grad_output.shape

        dx = np.zeros_like(x)
        dW = np.zeros_like(W)
        db = np.zeros_like(self.params["b"])

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(h_out):
                    for j in range(w_out):
                        i0 = i * self.stride
                        j0 = j * self.stride
                        region = x[b, :, i0:i0+k, j0:j0+k]
                        dW[oc] += grad_output[b, oc, i, j] * region
                        db[oc] += grad_output[b, oc, i, j]
                        dx[b, :, i0:i0+k, j0:j0+k] += grad_output[b, oc, i, j] * W[oc]

        self.grads["W"] = dW
        self.grads["b"] = db
        return dx
