import numpy as np
from .interface import Layer

class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.input_shape = x.shape  
        return x.reshape(x.shape[0], -1)  

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)
