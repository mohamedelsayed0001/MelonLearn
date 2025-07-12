import numpy as np
import pandas as pd

class ICA:
    """
    Independent Component Analysis (ICA) - using FastICA algorithm

    Parameters:
    -----------
    n_components : int
        Number of independent components to extract.

    Attributes:
    -----------
    components_ : ndarray
        The estimated independent components.
    mean_ : ndarray
        The mean of the input data (used for centering).
    """
    def __init__(self, n_components, max_iter=200, tol=1e-5):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.components_ = None
        self.mean_ = None

    def _g(self, x):
        # Non-linearity (log-cosh function)
        return np.tanh(x)

    def _g_prime(self, x):
        return 1 - np.tanh(x) ** 2

    def fit(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values

        
        self.mean_ = np.mean(X, axis=0)
        X = X - self.mean_


        cov = np.cov(X, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        D = np.diag(1. / np.sqrt(eigvals[-self.n_components:]))
        E = eigvecs[:, -self.n_components:]
        X_white = (X @ E) @ D


        W = np.random.randn(self.n_components, self.n_components)

        for _ in range(self.max_iter):
            W_old = W.copy()
            WX = X_white @ W.T
            gwx = self._g(WX)
            g_wx_prime = self._g_prime(WX)

            W = (gwx.T @ X_white) / X_white.shape[0] - np.diag(g_wx_prime.mean(axis=0)) @ W

            W = self._orthogonalize(W)

    
            if np.max(np.abs(np.abs(np.diag(W @ W_old.T)) - 1)) < self.tol:
                break

        self.components_ = W

    def transform(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        X = X - self.mean_
        return (X @ self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def _orthogonalize(self, W):
        # Gram-Schmidt process
        U, _, Vt = np.linalg.svd(W, full_matrices=False)
        return U @ Vt
