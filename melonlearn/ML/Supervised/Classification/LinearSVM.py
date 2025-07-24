import numpy as np
import pandas as pd

class LinearSVM:
    """
    Linear Support Vector Machine (SVM) Classifier using Hinge Loss and Sub-Gradient Descent.

    This class implements a binary SVM from scratch for classification tasks. It uses the hinge loss function:

        L(w, b) = 1/2 ||w||² + C * Σ max(0, 1 - yᵢ(w·xᵢ + b))

    The algorithm trains a linear decision boundary that maximizes the margin between two classes. 
    If a sample lies within the margin or is misclassified, it incurs a penalty weighted by the regularization parameter C.

    Parameters:
        - learning_rate: Step size for gradient updates
        - epochs: Number of training iterations
        - epsilon: ε-insensitive margin
        - C: Regularization strength

    Methods:
        - train(X, Y): Fit the model to training data
        - predict(X): Predict values for new data
        - evaluate(X, Y):  Evaluates model accuracy on test data (X, Y).
    """

    def __init__(self, learning_rate=0.01, epochs=1000, epsilon=0.1, C=1):
        self.epsilon = epsilon
        self.C = C
        self.learning_rate = learning_rate
        self.coefficients = None
        self.bias = 0.0

    def train(self, X, Y):
       
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if isinstance(Y, pd.Series) or isinstance(Y, pd.DataFrame):
            Y = Y.values.ravel()

        Y = np.where(Y <= 0, -1, 1)
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0.0

        for _ in range(self.epochs):
            for i in range(n_samples):
                cond = Y[i]*(np.dot(X[i], w) + b)
                

                if cond < 1:
                    w -= self.learning_rate * (w - self.C * X[i] * Y[i])
                    b += self.learning_rate * self.C *Y[i]
                else:
                    w -= self.learning_rate * w  

        self.coefficients = w
        self.bias = b

    def predict(self, X):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        return np.sign(np.dot(X, self.coefficients) + self.bias)

    def evaluate(self, X, Y):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if isinstance(Y, pd.Series) or isinstance(Y, pd.DataFrame):
            Y = Y.values.ravel()

        y_bar = self.predict(X)
        Y = np.where(Y <= 0, -1, 1)
        return np.mean(y_bar == Y)
