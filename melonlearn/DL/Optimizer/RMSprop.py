import numpy as np
from .interface import Optimizer

class RMSProp(Optimizer):
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.s = {}

    def update(self, param, grad):
        key = id(param)
        if key not in self.s:
            self.s[key] = np.zeros_like(grad)

        self.s[key] = self.beta * self.s[key] + (1 - self.beta) * grad**2
        param -= self.lr * grad / (np.sqrt(self.s[key]) + self.eps)
