import numpy as np
import pandas as pd

class PCA :

    """
        Principal Component Analysis (PCA) - from scratch

        This class reduces the dimensionality of data by projecting it onto the top N 
        principal components (directions of maximum variance).

        Parameters:
        -----------
        n_components : int
            Number of principal components to keep.

        Attributes:
        -----------
        components_ : ndarray
            Principal component directions (eigenvectors).
        explained_variance_ : ndarray
            Amount of variance explained by each component.
        mean_ : ndarray
            Mean of the original data (used for centering).
    """
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.mean_ = None
    
    def fit(self, X):
        if isinstance(X,(pd.DataFrame,pd.Series)):
            X = X.values
        
        if self.n_components > X.shape[1]:
            raise ValueError("n_components must be less than or equal to the number of features")

        
        self.mean_ = np.mean(X,axis = 0)
        X_centered = X - self.mean_
        cov = np.cov(X_centered,rowvar = False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:,sorted_idx]

        self.explained_variance_ = eigenvalues[:self.n_components]
        self.components_ = eigenvectors[:,:self.n_components]

    def transform(self,X):
        if isinstance(X,(pd.DataFrame,pd.Series)):
            X = X.values

        if self.components_ is None or self.mean_ is None:
            raise ValueError("The PCA model must be fitted before calling transform.")

        X_centered  = X - self.mean_
        return np.dot(X_centered,self.components_)
    
    def fit_transform(self ,X):
        self.fit(X)
        return self.transform(X)
