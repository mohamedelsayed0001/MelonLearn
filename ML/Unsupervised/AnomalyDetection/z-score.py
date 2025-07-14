import numpy as np 
import pandas as pd

class ZScore :
    """
    Z-Score Based Anomaly Detection (Model-Free, Univariate)

    Detects anomalies by measuring how many standard deviations each point is from the mean.
    Points with absolute Z-score above a threshold are flagged as anomalies.

    Parameters:
    -----------
    threshold : float
        The Z-score threshold to use for detecting anomalies (commonly 2.5 or 3.0).

    Methods:
    --------
    - fit(X): Calculates the mean and std of the data.
    - score(X): Returns the Z-scores for input X.
    - predict(X): Labels points as -1 (anomaly) or 1 (normal).
    """
    def __init__(self,threshold=3.0):
        self.threshold = threshold
        self.mean_ = None
        self.std_ = None

    def fit (self,X):
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.values
        
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

    def score(self, X):
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.values
        z_scores = (X - self.mean_) / self.std_
        return z_scores

    def predict(self, X):
        z_scores = self.score(X)
        return np.where(np.abs(z_scores) > self.threshold, -1, 1)   