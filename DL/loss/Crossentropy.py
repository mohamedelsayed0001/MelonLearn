import numpy as np

def cross_entropy_loss(y_pred, y_true, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def cross_entropy_grad(y_pred, y_true, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -y_true / y_pred / y_true.shape[0]
