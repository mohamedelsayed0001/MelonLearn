import  numpy as np 
from .interface import Layer
 # no autograd
def relu(x):
    return np.maximum(0,x)

def relu_grad(x):
    return (x>0).astype(float)

def sigmoid(x):
    return 1/ (1+np.exp(-x))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    x_shifted = x - np.mx(x,axis = 1 ,keepdims =True)
    exp = np.exp(x_shifted)
    return exp / np.sum(exp,axis = 1,keepdims = True)

def softmax_grad(x):
    s = softmax(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_grad(x):
    return 1 - np.tanh(x)**2

activation_functions = {
    "relu": (relu, relu_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
    "softmax": (softmax, softmax_grad),
    "tanh": (tanh, tanh_grad),
    None: (lambda x: x, lambda x: np.ones_like(x)) 
}

class Dense(Layer):
    def __init__(self,output_dim,activation=None):
        super().__init__()
        self.output_dim = output_dim
        self.activation, self.activation_grad = activation_functions[activation]
        self.built = False
    
    def build(self, input_dim):
        self.params["W"] = np.random.randn(input_dim, self.output_dim) * 0.01
        self.params["b"] = np.zeros((1, self.output_dim))
        self.built = True
    def forward(self, x):
        if not self.built:
            self.build(x.shape[1])
        self.input = x
        self.z = x @ self.params["W"] + self.params["b"]
        self.output = self.activation(self.z)
        return self.output
    
    def backward(self, grad_output):
        grad_z = grad_output * self.activation_grad(self.z)
        self.grads["w"] = self.input.T @ grad_z
        self.grads["b"] = np.sum(grad_z, axis=0, keepdims=True)
        return grad_z @ self.params["W"].T