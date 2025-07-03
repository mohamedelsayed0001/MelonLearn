import numpy as np

class LassoRegression:

    """
    Multivariate Lasso Regression using Gradient Descent.

    This class supports training on multiple features with L1 regularization (Lasso),
    fitting a model of the form:
        y = w1*x1 + w2*x2 + ... + wn*xn + b

    Methods:
        - train(X, Y): Trains the model using L1-regularized gradient descent.
        - predict(X): Predicts target values for input features.
        - evaluate(X, Y): Computes Mean Squared Error on test data.
    """
     
    def __init__(self,learning_rate=0.01,epochs=1000,Lambda = 2):
        self.Lambda = Lambda
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coefficients = None
        self.biase = 0.0 

    def train(self,X,Y):
        n_samples, n_features = X.shape

        w = np.zeros(n_features)
        b = 0.0

        for _ in range(self.epochs):
            y_bar = np.dot(X, w) + b  
            error = y_bar - Y

            dw = (2 / n_samples) * np.dot(X.T, error)
            db = (2 / n_samples) * np.sum(error)

            dw += self.Lambda * np.sign(w)

            w -= self.learning_rate*dw  
            b -= self.learning_rate*db
        
        self.coefficients = w 
        self.biase = b 

    def predict(self,X):
        return np.dot(X, self.coefficients) + self.biase
    
    def evaluate(self,X,Y):
        y_bar = self.predict(X)
        mse = np.mean((y_bar-Y)**2)
        return mse