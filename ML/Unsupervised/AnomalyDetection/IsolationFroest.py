import numpy as np
import pandas as pd

class IsolationForest:
    """
        Isolation Forest for Anomaly Detection (Model-free)

        This implementation detects anomalies based on the principle that 
        outliers are easier to isolate using random splits in a tree structure.

        Classes:
        --------
        - IsolationForest:
            An ensemble of isolation trees used to compute anomaly scores.
            
            Parameters:
                - n_estimators (int): Number of trees to build.
                - max_samples (int): Number of samples per tree.
            
            Methods:
                - fit(X): Build the isolation forest.
                - anomaly_score(X): Return anomaly score for each sample.
                - predict(X, threshold=0.6): Label samples as -1 (anomaly) or 1 (normal).

        - _IsolationTree (extra tree regresson):
            A binary tree that recursively isolates points by random feature and split.

            Methods:
                - fit(X): Recursively build the tree until max depth or 1 sample.
                - path_length(x): Compute how deep a sample x goes before isolation.
    """
    def __init__(self, n_estimators=100, max_samples=256):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.trees = []
    def fit(self,X):
        if isinstance(X,(pd.DataFrame, pd.Series)):
            X = X.values
        height_limit = int(np.ceil(np.log2(self.max_samples)))
        self.trees = []

        for _ in range(self.n_estimators):
            if len(X) > self.max_samples:
                idx = np.random.choice(len(X), self.max_samples, replace=False)
                X_sample = X[idx]
            else: 
                X_sample = X
            
            tree = _IsolationTree(height_limit)
            tree.fit(X_sample)
            self.trees.append(tree)
    
    def anomaly_score(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values

        scores = []
        for x in X:
            path_lengths = [tree.path_length(x) for tree in self.trees]
            avg_path_length = np.mean(path_lengths)
            c_n = _IsolationTree._c(self.max_samples)
            score = 2 ** (-avg_path_length / c_n)
            scores.append(score)
        return np.array(scores)
    def predict(self, X, threshold=0.6):
        scores = self.anomaly_score(X)
        return np.where(scores > threshold, -1, 1) 


class _IsolationTree :
    def __init__ (self ,height_limit):
        self.height_limit  = height_limit
        self.left = None
        self.right = None
        self.split_attr = None
        self.split_value = None
        self.size = None
        self.is_leaf = False
    def fit(self, X, current_height=0):
        self.size = len(X)
        if current_height >= self.height_limit or self.size <= 1:
            self.is_leaf = True
            return

        q = np.random.randint(X.shape[1])  # random feature
        min_val, max_val = np.min(X[:, q]), np.max(X[:, q])
        if min_val == max_val:
            self.is_leaf = True
            return

        p = np.random.uniform(min_val, max_val)  # random split value
        self.split_attr = q
        self.split_value = p

        left_mask = X[:, q] < p
        right_mask = ~left_mask

        self.left = _IsolationTree(self.height_limit)
        self.left.fit(X[left_mask], current_height + 1)

        self.right = _IsolationTree(self.height_limit)
        self.right.fit(X[right_mask], current_height + 1)

    def path_length(self, x, current_height=0):
        if self.is_leaf or self.split_attr is None:
            return current_height + self._c(self.size)
        if x[self.split_attr] < self.split_value:
            return self.left.path_length(x, current_height + 1)
        else:
            return self.right.path_length(x, current_height + 1)

    @staticmethod
    def _c(n):
        # Average path length of unsuccessful search in a binary tree
        if n <= 1:
            return 0
        return 2 * np.log(n - 1) + 0.5772156649  # Euler-Mascheroni constant