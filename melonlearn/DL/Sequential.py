class Sequential:
    def __init__(self, layers):
        self.layers = layers
        self.layers = layers
        self.loss_fn = None
        self.loss_fn_grad = None
        self.optimizer = None
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def get_params_and_grads(self):
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads.get(name)
                yield param, grad

    def compile(self, loss, optimizer):
        self.loss_fn, self.loss_fn_grad = loss
        self.optimizer = optimizer

    def train(self,X,y):

        predictions = self.forward(X)
        loss = self.loss_fn(predictions,y)

        grad_loss = self.loss_fn_grad(predictions,y)
        self.backward(grad_loss)

        for param, grad in self.get_params_and_grads():
            self.optimizer.update(param, grad)

        return loss