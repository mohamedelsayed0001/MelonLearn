import numpy as np
import pandas as pd

class TreeNode:
    def __init__ (self,feature=None, threshold=None, value=None, left=None, right=None):
        self.feature = feature        
        self.threshold = threshold    
        self.value = value            
        self.left = left 
        self.right = right


class RegressionTree:
    """
    Regression Tree using MSE as the split criterion.

    Parameters:
        max_depth (int): Maximum depth of the tree.
        min_samples_split (int): Minimum number of samples required to split a node.

    Methods:
        - train(X, y): Fit the tree to training data.
        - predict(X): Predict target values.
        - evaluate(X, y): Compute Mean Squared Error on test data.
    """
     
    def __init__(self,max_depth = 7,min_samples_split = 6):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root =None

    def _mse(self,y):
        if len(y) == 0:
            return 0 
        return np.mean((y-np.mean(y))**2)
    
    def _best_split(self,X,Y):
        best_feature = None
        best_threshold = None
        best_mse = float('inf')
        n_samples,n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:,feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold
            
                if len(Y[left_mask]) == 0 or len(Y[right_mask]) ==0:
                    continue
                
                left_mse = self._mse(Y[left_mask])
                right_mse = self._mse(Y[right_mask])
                weighted_mse = (len(Y[left_mask]) * left_mse + len(Y[right_mask]) * right_mse) / len(Y)

                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold
    
    def _build_tree(self,X,y,depth = 0):
        if (len(y) < self.min_samples_split or
            depth >= self.max_depth or
            len(np.unique(y)) == 1):
            return TreeNode(value=np.mean(y))  # leaf
        
        feature, threshold = self._best_split(X, y)

        if feature is None:
            return TreeNode(value=np.mean(y))
        
        left_mask = X[:,feature] <= threshold
        right_mask = X[:,feature] > threshold

        left = self._build_tree(X[left_mask],y[left_mask],depth=depth+1)
        right = self._build_tree(X[right_mask],y[right_mask],depth=depth+1)

        return TreeNode(feature=feature, threshold=threshold,left=left ,right=right)
    def train(self, X, y):
        
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.values.ravel()

        self.root = self._build_tree(X, y)
    
    def _predict_sample(self, x, node): 
        
        """
        Traverse the tree like binary search to find the prediction value.
        """

        while node.value is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values

        return np.array([self._predict_sample(x, self.root) for x in X])

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.values.ravel()
        return np.mean((y_pred - y) ** 2)


