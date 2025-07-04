import numpy as np

class SVR:
    """
    Support Vector Regression (SVR) using ε-insensitive loss and sub-gradient descent.

    This implementation supports:
    - Multivariate (n-dimensional) input features
    - Linear SVR (no kernel)
    - Training via gradient-based updates with regularization (L2)

    Methods:
        - train(X, Y): Fit the model to the training data
        - predict(X): Predict target values for given input features
        - evaluate(X, Y): Compute Mean Squared Error (MSE) on test data
    """

    def __init__(self,learning_rate=0.01,epochs=1000,epsilon=0.1,C=1):
        self.epsilon = epsilon 
        self.C = C
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coefficients = None
        self.bias = 0.0 

    def train(self,X,Y):
        n_samples, n_features = X.shape

        w = np.zeros(n_features)
        b = 0.0

        for _ in range(self.epochs):
            for i in range(n_samples):
                y_bar = np.dot(X[i],w)+b
                error = y_bar - Y[i]
                if error > self.epsilon:
                    w -= self.learning_rate * (w + self.C*X[i])
                    b -= self.learning_rate * self.C
                elif error < -self.epsilon:
                    w -= self.learning_rate *(w - self.C*X[i])
                    b += self.learning_rate* self.C
                else:
                    w -= self.learning_rate * w
        
        self.coefficients = w 
        self.bias = b 

    def predict(self,X):
        return np.dot(X, self.coefficients) + self.bias
    
    def evaluate(self,X,Y):
        y_bar = self.predict(X)
        mse = np.mean((y_bar-Y)**2)
        return mse