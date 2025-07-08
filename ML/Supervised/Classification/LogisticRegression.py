import numpy as np
import pandas as pd

class LogisticRegression :
    def __init__ (self , learning_rate = 0.01,epochs=1000,threshold = 0.5,activation = "sigmoid"):
        self.learning_rate = learning_rate
        self.epochs =epochs
        self.activation = activation
        self.threshold = threshold
        self.weights = None
        self.bias = None

    def _activation(self,z):
        if self.activation == "sigmoid" :
            return 1/(1+np.exp(-z))
        elif self.activation == "tanh":
            return np.tanh(z)
        else :
            return ValueError("nsupported activation: choose 'sigmoid' or 'tanh'") 
    
    def train(self,X,y):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.ravel()
            
        n_samples,n_features =X.shape
        self.weights =  np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            z = np.dot(X,self.weights) + self.bias
            y_bar = self._activation(z)
        
            dw = (1/n_samples) * np.dot(X.T,(y_bar-y))
            db = (1/n_samples) * np.sum(y_bar-y)

            self.weights -= self.learning_rate *dw 
            self.bias -= self.learning_rate*db
    
    def predict(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        z =np.dot(X, self.weights) + self.bias
        y_bar = self._activation(z)

        if self.activation == "tanh":
            return np.where(y_bar >= 0, 1, 0)
        else:
            return np.where(y_bar >= self.threshold, 1, 0)
   
    def evaluate(self, X, Y):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if isinstance(Y, pd.Series) or isinstance(Y, pd.DataFrame):
            Y = Y.values.ravel()

        y_bar = self.predict(X)
        mse = np.mean((y_bar - Y) ** 2)
        return mse