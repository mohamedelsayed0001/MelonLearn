import numpy as np
import pandas as pd
from DecisionTree import DecisionTree  # Import your classification tree

class RandomForestClassifier:
    """
    Random Forest Classifier using custom DecisionTree (based on Gini Impurity).

    Parameters:
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum depth of each tree.
        min_samples_split (int): Minimum samples required to split a node.
        max_features (int, float, 'sqrt', or 'log2'): Number of features to consider at each split.

    Methods:
        - train(X, y): Train the forest on input data.
        - predict(X): Predict class labels by majority vote.
        - evaluate(X, y): Return accuracy on test data.
    """
    def __init__(self, n_estimators=10, max_depth=7, min_samples_split=6, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.feature_indices_per_tree = []

    def _get_feature_indices(self, n_features):
        if isinstance(self.max_features, int):
            return np.random.choice(n_features, self.max_features, replace=False)
        elif self.max_features == 'sqrt':
            return np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
        elif self.max_features == 'log2':
            return np.random.choice(n_features, int(np.log2(n_features)), replace=False)
        else:
            return np.arange(n_features)

    def train(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.ravel().astype(int)

        n_samples, n_features = X.shape

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

      
            feature_indices = self._get_feature_indices(n_features)
            X_sample_subset = X_sample[:, feature_indices]

       
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.train(X_sample_subset, y_sample)

            self.trees.append(tree)
            self.feature_indices_per_tree.append(feature_indices)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        all_preds = []
        for tree, feature_indices in zip(self.trees, self.feature_indices_per_tree):
            X_subset = X[:, feature_indices]
            preds = tree.predict(X_subset)
            all_preds.append(preds)

        # Transpose to get predictions per sample across all trees
        all_preds = np.array(all_preds).T
        final_preds = [np.bincount(row).argmax() for row in all_preds]  # majority vote
        return np.array(final_preds)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.ravel()
        return np.mean(y_pred == y)
