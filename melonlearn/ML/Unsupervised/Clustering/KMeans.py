import numpy as np
import pandas as pd 

class KMeans :

    """
        K-Means Clustering Algorithm (from scratch, NumPy + Pandas-compatible)

        This class implements the K-Means clustering algorithm, which partitions data into `k` clusters 
        based on similarity (Euclidean distance). The algorithm iteratively assigns data points to the 
        nearest cluster center and updates cluster centers based on the mean of assigned points.

        Parameters:
        -----------
        k : int
            The number of clusters to form.
        max_iters : int
            Maximum number of iterations for updating centroids.
        tol : float
            Tolerance threshold to check for convergence (based on centroid movement).

        Attributes:
        -----------
        centroids : ndarray
            Coordinates of the cluster centers.
        labels : ndarray
            Cluster labels assigned to each data point.

        Methods:
        --------
        fit(X):
            Fits the K-Means model to the input data `X` (can be a NumPy array or Pandas DataFrame).
        
        predict(X):
            Predicts the closest cluster each data point in `X` belongs to.
    """
    def __init__(self , k = 3 , max_iters = 100 ,tol = 1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
    
    def fit(self, X):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        np.random.seed(42)
        starting_idx = np.random.permutation(len(X))[:self.k]
        self.centroids = X[starting_idx]

        for i in range(self.max_iters):
            dist = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(dist, axis = 1)

            new_centroids  = np.array([X[self.labels == j].mean(axis=0) for j in range(self.k)])
            if np.all(np.linalg.norm(self.centroids - new_centroids, axis=1) < self.tol):
                break

            self.centroids = new_centroids

    def predict(self,X):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        dist = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(dist, axis=1)