import numpy as np
import pandas as pd
from collections import deque

class DBSCAN:
    """
    DBSCAN clustering algorithm (brute-force distance calculation)

    Parameters:
    -----------
    eps : float
        Maximum distance to consider a neighbor.
    min_samples : int
        Minimum number of neighbors to form a core point.

    Attributes:
    -----------
    labels_ : ndarray
        Cluster labels for each point (-1 for noise).
    """
    def __init__(self , eqs = 0.5 , min_samples = 5):
        self.eqs = eqs
        self.min_samples = min_samples
        self.labels_ = None 

    def fit(self,X): 
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        n=len(X)
        self.labels_ = np.full(n, -1) 
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True

            neighbors  = self._region_query(X, i)

            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1  # Noise
            else:
                self._expand_cluster(X, i, neighbors, cluster_id, visited)
                cluster_id += 1
    def _region_query(self, X, i):
        distances = np.linalg.norm(X - X[i], axis=1)
        return np.where(distances <= self.eps)[0]
    
    def _expand_cluster(self, X, i, neighbors, cluster_id, visited):
        self.labels_[i] = cluster_id
        queue = deque(neighbors)

        while queue:
            j = queue.popleft()
            if not visited[j]:
                visited[j] = True 
                new_neighbors  = self._region_query(X, j)
                if len(new_neighbors) >= self.min_samples:
                    queue.extend(new_neighbors)
            if self.labels_[j] == -1:
                self.labels_[j] = cluster_id