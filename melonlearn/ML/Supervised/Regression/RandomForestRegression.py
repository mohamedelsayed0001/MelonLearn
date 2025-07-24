import numpy as np 
import pandas as pd 
from RegressionTree import RegressionTree

class RandomForestRegressor:
    """
    Random Forest Regressor using custom RegressionTree.

    Parameters:
        n_estimators (int): Number of trees.
        max_depth (int): Max depth of each tree.
        min_samples_split (int): Minimum samples to split.
        max_features (int or float or 'sqrt' or 'log2'): Number of features to consider per tree.

    Methods:
        - train(X, y): Train the forest.
        - predict(X): Predict by averaging predictions of trees.
        - evaluate(X, y): Return MSE on given data.
    """
    def __init__(self,n_estimators=10,max_depth=7,min_samples_split=6,max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
    def _get_feature(self,n_features):
        if isinstance(self.max_features, int):
            return np.random.choice(n_features, self.max_features, replace=False)
        elif self.max_features == 'sqrt':
            return np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
        elif self.max_features == 'log2':
            return np.random.choice(n_features, int(np.log2(n_features)), replace=False)
        else:
            return np.arange(n_features)
    
    def train(self,X,y):
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.ravel()
        n_samples,n_features = X.shape
        self.feature_pre_tree = [] 

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples , size = n_samples,replace = True)
            X_sample = X[indices]
            y_sample = y[indices]

            feature_indices = self._get_feature(n_features=n_features)
            X_sample_sub = X_sample[:,feature_indices]

            tree = RegressionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.train(X_sample_sub, y_sample)

            self.trees.append(tree)
            self.feature_pre_tree.append(feature_indices)
        
    def predict(self,X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        all_preds = []

        for tree, feat_idx in zip(self.trees, self.feature_indices_per_tree):
            preds = tree.predict(X[:, feat_idx])
            all_preds.append(preds)

        return np.mean(all_preds,axis = 0)
    def evaluate(self, X, y):
        y_bar = self.predict(X)
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.ravel()
        return np.mean((y_bar - y) ** 2)