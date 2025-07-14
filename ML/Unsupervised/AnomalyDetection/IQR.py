import numpy as np
import pandas as pd

class IQRAnomalyDetector:
    """
    IQR-Based Anomaly Detection (Model-Free, Univariate)

    Detects anomalies using the Interquartile Range (IQR) method.
    Points lying outside the [Q1 - factor*IQR, Q3 + factor*IQR] range are flagged as outliers.

    Parameters:
    -----------
    factor : float
        The multiplier for IQR to determine the outlier range (default: 1.5).

    Methods:
    --------
    - fit(X): Computes Q1, Q3, and IQR from the data.
    - score(X): Returns a boolean mask of points outside the IQR range.
    - predict(X): Labels points as -1 (anomaly) or 1 (normal).
    """

    def __init__(self, factor=1.5):
        self.factor = factor
        self.q1_ = None
        self.q3_ = None
        self.iqr_ = None
        self.lower_bound_ = None
        self.upper_bound_ = None
    def fit(self,X):
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.values
        self.q1_ = np.percentile(X, 25, axis=0)
        self.q3_ = np.percentile(X, 75, axis=0)
        self.iqr_ = self.q3_ - self.q1_
        self.lower_bound_ = self.q1_ - self.factor * self.iqr_
        self.upper_bound_ = self.q3_ + self.factor * self.iqr_

    def score (self,X):
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.values
        return (X < self.lower_bound_) | (X > self.upper_bound_)
    
    def predict(self, X):
        outliers = self.score(X)
        return np.where(outliers, -1, 1)