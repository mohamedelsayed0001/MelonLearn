import numpy as np
import pandas as pd 
from collections import Counter

class KNNClassifier :
    """
    K-Nearest Neighbors (KNN) Classifier from Scratch

    This implementation supports:
        - Classification of binary and multi-class datasets
        - Three distance metrics: 'euclidean', 'manhattan', and 'minkowski'
        - Flexible value of k (number of nearest neighbors)

        Methods:
        - train(X, y): Store the training data
        - predict(X): Predict labels for new input data
        - evaluate(X, y): Return classification accuracy on test data
    """
    
    def __init__(self ,k =3,distance_metric='euclidean', p=3):
        self.k = k
        self.distance_metric = distance_metric
        self.p = p
        self.X_train = None
        self.y_train = None
    def train(self,X,y ):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.ravel()

        self.X_train = X
        self.y_train = y

    def _distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == 'minkowski':
            return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)
        else:
            raise ValueError("Invalid distance metric. Use 'euclidean', 'manhattan', or 'minkowski'.")
    
    def _predict_single(self, x):
        distances = [self._distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]
    

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