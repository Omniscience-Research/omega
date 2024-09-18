import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import euclidean_distances

class DirectionalWeightedKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, weights='uniform', p=2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Store the training data
        self.X_ = X
        self.y_ = y
        
        # Calculate the centroid for each class
        self.centroids_ = {}
        for c in self.classes_:
            self.centroids_[c] = np.mean(X[y == c], axis=0)
        
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Compute distances between input samples and training samples
        distances = euclidean_distances(X, self.X_)
        
        # Get indices of k-nearest neighbors
        indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        
        # Predict for each input sample
        y_pred = np.zeros(X.shape[0], dtype=self.y_.dtype)
        for i, neighbors in enumerate(indices):
            # Calculate directional weights
            weights = self._calculate_directional_weights(X[i], neighbors)
            
            # Get labels of neighbors
            neighbor_labels = self.y_[neighbors]
            
            # Weighted voting
            unique_labels, label_counts = np.unique(neighbor_labels, return_counts=True)
            weighted_counts = np.zeros_like(label_counts, dtype=float)
            for j, label in enumerate(unique_labels):
                weighted_counts[j] = np.sum(weights[neighbor_labels == label])
            
            # Predict the class with the highest weighted count
            y_pred[i] = unique_labels[np.argmax(weighted_counts)]
        
        return y_pred

    def _calculate_directional_weights(self, query_point, neighbor_indices):
        weights = np.ones(len(neighbor_indices))
        
        # Calculate the average direction to class centroids
        avg_direction = np.zeros_like(query_point)
        for c in self.classes_:
            avg_direction += self.centroids_[c] - query_point
        avg_direction /= len(self.classes_)
        
        # Normalize the average direction
        avg_direction /= np.linalg.norm(avg_direction)
        
        for i, idx in enumerate(neighbor_indices):
            # Calculate the direction from query point to neighbor
            neighbor_direction = self.X_[idx] - query_point
            neighbor_direction /= np.linalg.norm(neighbor_direction)
            
            # Calculate the dot product to determine if the neighbor is "in front of" the query point
            dot_product = np.dot(avg_direction, neighbor_direction)
            
            # Adjust weight based on dot product
            if dot_product > 0:
                weights[i] *= (1 + dot_product)
            else:
                weights[i] *= (1 - abs(dot_product))
        
        return weights