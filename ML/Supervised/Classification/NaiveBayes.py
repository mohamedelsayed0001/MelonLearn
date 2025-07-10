import numpy as np 
import pandas as pd 

class NaiveBayesClassifier:
    def __init__(self):
        self.class_poiors ={}
        self.likelihoods = {}
        self.classes= None 
        self.epsilon = 1e-9
    def train(self,X,y):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.ravel()
        
        n_samples,n_features =  X.shape
        self.classes = np.unique(y)
        self.class_priors = {}
        self.feature_likelihoods = {}


        for c in self.classes :
            X_c = X[y == c]
            self.class_poiors[c] = X_c.shape[0]/n_samples
            self.feature_likelihoods[c] = (np.sum(X_c, axis=0) + 1) / (np.sum(X_c) + n_features)
    
    def _predict_single(self, x):
        posteriors ={}

        for c in self.classes:
            prior = np.log(self.class_priors[c])
            likelihood = np.sum(x * np.log(self.feature_likelihoods[c] + self.epsilon))
            posteriors[c] = prior + likelihood
        
        return max(posteriors,key=posteriors.get)
    
    def predict(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        return np.array([self._predict_single(x) for x in X])

    def evaluate(self, X, y):
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.ravel()
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy