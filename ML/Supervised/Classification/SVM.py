import numpy as np
import pandas as pd





class KernelSVM:
    """
        Support Vector Machine classifier using kernel trick and simplified SMO algorithm.
        
        Supports:
        - Polynomial kernel
        - RBF (Gaussian) kernel
        
        Parameters:
        -----------
        kernel : str
            Type of kernel to use: 'linear', 'polynomial', or 'rbf'
        C : float
            Regularization strength
        tol : float
            Numerical tolerance for alpha updates
        max_passes : int
            Max number of passes without changes before stopping

        Methods:
        --------
        - train(X, Y): Fit the model on training data
        - predict(X): Predict class labels for given input
        - evaluate(X, Y): Returns accuracy score on test data
    """

    def __init__(self, kernel='linear', C=1.0, tol=1e-3, max_passes=5):
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.alphas = None
        self.b = 0
        self.X = None
        self.Y = None

        
        if kernel == 'polynomial':
            self.kernel = lambda x, y: self.__polynomial_kernel(x, y, degree=3)
        elif kernel == 'rbf':
            self.kernel = lambda x, y: self.__rbf_kernel(x, y, gamma=0.5)
        else:
            raise ValueError("Unsupported kernel.")


    def __polynomial_kernel(self,x1, x2, degree=3):
        return (1 + np.dot(x1, x2)) ** degree

    def __rbf_kernel(self,x1, x2, gamma=0.1):
        diff = x1 - x2
        return np.exp(-gamma * np.dot(diff, diff))
    
    def train(self, X, Y):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(Y, (pd.DataFrame, pd.Series)):
            Y = Y.values.ravel()

        Y = np.where(Y <= 0, -1, 1)

        n_samples = X.shape[0]
        self.alphas = np.zeros(n_samples)
        self.b = 0
        self.X = X
        self.Y = Y

        passes = 0
        while passes < self.max_passes:
            alpha_changed = 0
            for i in range(n_samples):
                xi, yi = X[i], Y[i]
                Ei = self._decision(xi) - yi

                # Check if this alpha violates KKT conditions
                if (yi * Ei < -self.tol and self.alphas[i] < self.C) or (yi * Ei > self.tol and self.alphas[i] > 0):
                    j = np.random.choice([x for x in range(n_samples) if x != i])
                    xj, yj = X[j], Y[j]
                    Ej = self._decision(xj) - yj

                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]

                    # Compute bounds for alpha_j
                    if yi != yj:
                        L = max(0, alpha_j_old - alpha_i_old)
                        H = min(self.C, self.C + alpha_j_old - alpha_i_old)
                    else:
                        L = max(0, alpha_i_old + alpha_j_old - self.C)
                        H = min(self.C, alpha_i_old + alpha_j_old)

                    if L == H:
                        continue

                    # Compute eta (second derivative)
                    eta = 2 * self.kernel(xi, xj) - self.kernel(xi, xi) - self.kernel(xj, xj)
                    if eta >= 0:
                        continue

                    # Update alpha_j
                    self.alphas[j] -= yj * (Ei - Ej) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)

                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue

                    # Update alpha_i
                    self.alphas[i] += yi * yj * (alpha_j_old - self.alphas[j])

                    # Update bias term
                    b1 = self.b - Ei - yi * (self.alphas[i] - alpha_i_old) * self.kernel(xi, xi) \
                         - yj * (self.alphas[j] - alpha_j_old) * self.kernel(xi, xj)
                    b2 = self.b - Ej - yi * (self.alphas[i] - alpha_i_old) * self.kernel(xi, xj) \
                         - yj * (self.alphas[j] - alpha_j_old) * self.kernel(xj, xj)

                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    alpha_changed += 1

            # If no alphas changed, increase pass count
            passes = passes + 1 if alpha_changed == 0 else 0

    def _decision(self, x):
        """Internal method: computes the decision value for input x."""
        result = 0
        for i in range(len(self.alphas)):
            if self.alphas[i] > 0:
                result += self.alphas[i] * self.Y[i] * self.kernel(self.X[i], x)
        return result + self.b

    def predict(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        return np.sign(np.array([self._decision(x) for x in X]))

    def evaluate(self, X, Y):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(Y, (pd.DataFrame, pd.Series)):
            Y = Y.values.ravel()
        Y = np.where(Y <= 0, -1, 1)
        preds = self.predict(X)
        return np.mean(preds == Y)
