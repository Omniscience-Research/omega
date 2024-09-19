import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class InteractionAwareKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, interaction_degree=2, interaction_weight=0.5):
        self.n_neighbors = n_neighbors
        self.interaction_degree = interaction_degree
        self.interaction_weight = interaction_weight

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Create polynomial features for interactions
        self.poly = PolynomialFeatures(degree=self.interaction_degree, include_bias=False)
        X_poly = self.poly.fit_transform(X)
        
        # Combine original features with interaction features
        self.X_combined_ = np.hstack([X, self.interaction_weight * X_poly])
        
        # Fit the nearest neighbors model
        self.nn_ = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.nn_.fit(self.X_combined_)
        
        # Store the training labels
        self.y_ = y

        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        
        # Transform input data with polynomial features
        X_poly = self.poly.transform(X)
        
        # Combine original features with interaction features
        X_combined = np.hstack([X, self.interaction_weight * X_poly])
        
        # Find nearest neighbors
        distances, indices = self.nn_.kneighbors(X_combined)
        
        # Predict labels based on nearest neighbors
        y_pred = np.array([np.argmax(np.bincount(self.y_[ind])) for ind in indices])
        
        return y_pred

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        
        # Transform input data with polynomial features
        X_poly = self.poly.transform(X)
        
        # Combine original features with interaction features
        X_combined = np.hstack([X, self.interaction_weight * X_poly])
        
        # Find nearest neighbors
        distances, indices = self.nn_.kneighbors(X_combined)
        
        # Calculate probabilities based on nearest neighbors
        probas = np.array([np.bincount(self.y_[ind], minlength=len(self.classes_)) 
                           for ind in indices])
        probas = probas / self.n_neighbors
        
        return probas