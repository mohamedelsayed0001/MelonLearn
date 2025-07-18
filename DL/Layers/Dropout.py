import numpy as np
from .interface import Layer

class Dropout(Layer):
    def __init__(self, rate=0.5):
        super().__init__()
        self.rate = rate
        self.mask = None
        self.training = True  

    def forward(self, x):
        if self.training:
            self.mask = (np.random.rand(*x.shape) >= self.rate).astype(float)
            return x * self.mask / (1.0 - self.rate)
        else:
            return x  # No dropout during inference

    def backward(self, grad_output):
        return grad_output * self.mask / (1.0 - self.rate) if self.training else grad_output
