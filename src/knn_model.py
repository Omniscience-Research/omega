import numpy as np
from collections import Counter

class KNearestNeighbors:
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        Fit the k-nearest neighbors classifier.

        Parameters:
        X_train (array-like): Training data of shape (n_samples, n_features)
        y_train (array-like): Target values of shape (n_samples,)

        Returns:
        self: Returns an instance of self.
        """
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        return self

    def predict(self, X_test):
        """
        Predict the class labels for the provided data.

        Parameters:
        X_test (array-like): Test samples of shape (n_samples, n_features)

        Returns:
        y_pred (array): Predicted class label for each sample in X_test
        """
        X_test = np.array(X_test)
        y_pred = [self._predict_single(x) for x in X_test]
        return np.array(y_pred)

    def _predict_single(self, x):
        """
        Predict the class for a single sample.
        """
        distances = self._calculate_distances(x)
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_nearest_labels = self.y_train[k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _calculate_distances(self, x):
        """
        Calculate distances between a single sample and all training samples.
        """
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(self.X_train - x), axis=1)
        else:
            raise ValueError("Unsupported metric. Use 'euclidean' or 'manhattan'.")