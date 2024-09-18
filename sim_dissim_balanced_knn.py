import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import euclidean_distances

class BalancedSimDissimKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, sim_weight=0.7, dissim_weight=0.3):
        self.n_neighbors = n_neighbors
        self.sim_weight = sim_weight
        self.dissim_weight = dissim_weight

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Store the training data
        self.X_ = X
        self.y_ = y
        
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Calculate distances between all test points and all training points
        distances = euclidean_distances(X, self.X_)
        
        # Get indices of k nearest neighbors for similarity
        sim_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        
        # Get indices of k farthest neighbors for dissimilarity
        dissim_indices = np.argsort(distances, axis=1)[:, -self.n_neighbors:]
        
        # Initialize predictions array
        predictions = np.zeros(X.shape[0], dtype=self.y_.dtype)
        
        for i in range(X.shape[0]):
            # Get labels of similar and dissimilar neighbors
            sim_labels = self.y_[sim_indices[i]]
            dissim_labels = self.y_[dissim_indices[i]]
            
            # Count occurrences of each class in similar neighbors
            sim_counts = np.bincount(sim_labels, minlength=len(self.classes_))
            
            # Count occurrences of each class in dissimilar neighbors
            dissim_counts = np.bincount(dissim_labels, minlength=len(self.classes_))
            
            # Calculate weighted score for each class
            scores = self.sim_weight * sim_counts - self.dissim_weight * dissim_counts
            
            # Predict the class with the highest score
            predictions[i] = self.classes_[np.argmax(scores)]
        
        return predictions