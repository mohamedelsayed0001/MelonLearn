import  numpy as np

def mse_loss(y_bar,y):
    return np.mean((y_bar - y)**2)

def mse_grad(y_bar,y):
    return 2 * (y_bar - y) / y.size