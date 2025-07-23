from .interface import Optimizer

class SGD(Optimizer):
    def __init__(self,lr = 0.01):
        self.lr = 0.01
    
    def update(self, param, grad):
       param -= self.lr * grad