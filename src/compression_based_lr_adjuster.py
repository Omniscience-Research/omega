import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler
import gzip

class CompressibilityAdaptiveLR(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.01, n_iterations=1000, compression_threshold=0.5, tol=1e-4, random_state=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.compression_threshold = compression_threshold
        self.tol = tol
        self.random_state = random_state

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def _compress(self, data):
        return len(gzip.compress(data.tobytes())) / len(data.tobytes())

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        if len(self.classes_) != 2:
            raise ValueError("This classifier is only for binary classification.")
        
        # Store the number of features
        self.n_features_in_ = X.shape[1]
        
        # Initialize the scaler
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Initialize weights
        rng = np.random.RandomState(self.random_state)
        self.coef_ = rng.normal(0, 0.01, self.n_features_in_)
        self.intercept_ = rng.normal(0, 0.01, 1)
        
        for _ in range(self.n_iterations):
            z = np.dot(X_scaled, self.coef_) + self.intercept_
            y_pred = self._sigmoid(z)
            
            error = y_pred - y
            grad_coef = np.dot(X_scaled.T, error) / y.size
            grad_intercept = np.mean(error)
            
            compression_ratio = self._compress(np.concatenate([self.coef_, self.intercept_]))
            adjusted_lr = self.learning_rate * (1 - compression_ratio) if compression_ratio < self.compression_threshold else self.learning_rate
            
            coef_new = self.coef_ - adjusted_lr * grad_coef
            intercept_new = self.intercept_ - adjusted_lr * grad_intercept
            
            if np.all(np.abs(coef_new - self.coef_) < self.tol) and np.abs(intercept_new - self.intercept_) < self.tol:
                break
            
            self.coef_ = coef_new
            self.intercept_ = intercept_new
        
        # Return the classifier
        return self

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        
        # Check if the input features match the number of features from training
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but CompressibilityAdaptiveLR is expecting {self.n_features_in_} features.")
        
        # Apply the same scaling as during fit
        X_scaled = self.scaler_.transform(X)
        
        # Compute probabilities
        z = np.dot(X_scaled, self.coef_) + self.intercept_
        proba = self._sigmoid(z)
        return np.vstack((1 - proba, proba)).T

    def predict(self, X):
        # Predict class labels
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]