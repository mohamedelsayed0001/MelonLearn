import numpy as np
from .interface import Optimizer

class Momentum(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}
    
    def update(self, param, grad):
        key = id(param)
        if key not in self.velocity:
            self.velocity[key] = np.zeros_like(grad)

        self.velocity[key] = self.momentum * self.velocity[key] - self.lr * grad
        param += self.velocity[key]