import numpy as np
from .interface import Optimizer

class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = {}
    def update(self, param, grad):
        key = id(param)

        if  key not in self.m:
            self.m[key] = np.zeros_like(grad)
            self.v[key] = np.zeros_like(grad)
            self.t[key] = 0
        
        self.t[key] += 1
        self.m[key]  = self.beta1 * self.m[key] + (1-self.beta1) * grad
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

        m_corr = self.m[key] / (1 - self.beta1 ** self.t[key])
        v_corr = self.v[key] / (1 - self.beta2 ** self.t[key])
        

        param -= self.lr * m_corr / (np.sqrt(v_corr) + self.eps)