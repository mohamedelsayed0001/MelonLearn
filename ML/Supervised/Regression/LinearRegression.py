import numpy as np
import pandas as pd

class LinearRegression:
    """
    Multivariate Linear Regression using Gradient Descent.

    This class supports training on multiple features to fit a model of the form:
        y = w1*x1 + w2*x2 + ... + wn*xn + b

    Methods:
        - train(X, Y): Fits weights and bias using gradient descent.
        - predict(X): Predicts target values for input features.
        - evaluate(X, Y): Computes Mean Squared Error on test data.
    """

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coefficients = None
        self.bias = 0.0

    def train(self, X, Y):
        
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if isinstance(Y, pd.Series) or isinstance(Y, pd.DataFrame):
            Y = Y.values.ravel()

        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0.0

        for _ in range(self.epochs):
            y_bar = np.dot(X, w) + b
            error = y_bar - Y

            dw = (2 / n_samples) * np.dot(X.T, error)
            db = (2 / n_samples) * np.sum(error)

            w -= self.learning_rate * dw
            b -= self.learning_rate * db

        self.coefficients = w
        self.bias = b

    def predict(self, X):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        return np.dot(X, self.coefficients) + self.bias

    def evaluate(self, X, Y):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if isinstance(Y, pd.Series) or isinstance(Y, pd.DataFrame):
            Y = Y.values.ravel()
        y_bar = self.predict(X)
        mse = np.mean((y_bar - Y) ** 2)
        return mse
