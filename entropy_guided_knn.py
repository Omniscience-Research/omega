import numpy as np
from scipy.stats import entropy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class EntropyGuidedKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, min_k=3, max_k=20, entropy_threshold=0.5):
        self.n_neighbors = n_neighbors
        self.min_k = min_k
        self.max_k = max_k
        self.entropy_threshold = entropy_threshold

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Store the training data
        self.X_ = X
        self.y_ = y
        
        # Fit NearestNeighbors for later use
        self.nn_ = NearestNeighbors(n_neighbors=self.max_k, metric='euclidean')
        self.nn_.fit(X)
        
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Get nearest neighbors and distances
        distances, indices = self.nn_.kneighbors(X)

        # Predict for each input sample
        y_pred = np.empty(X.shape[0], dtype=self.classes_.dtype)
        for i in range(X.shape[0]):
            # Calculate local entropy
            local_entropy = self._calculate_entropy(self.y_[indices[i]])
            
            # Determine adaptive k
            adaptive_k = self._get_adaptive_k(local_entropy)
            
            # Get the labels of the k nearest neighbors
            k_nearest_labels = self.y_[indices[i][:adaptive_k]]
            
            # Predict the most common class
            y_pred[i] = np.argmax(np.bincount(k_nearest_labels))

        return y_pred

    def _calculate_entropy(self, labels):
        _, counts = np.unique(labels, return_counts=True)
        return entropy(counts, base=2)

    def _get_adaptive_k(self, local_entropy):
        if local_entropy > self.entropy_threshold:
            # High entropy: increase k to reduce noise
            return min(self.max_k, int(self.n_neighbors * 1.5))
        else:
            # Low entropy: decrease k to capture fine-grained patterns
            return max(self.min_k, int(self.n_neighbors * 0.5))