class Layer:
    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError
