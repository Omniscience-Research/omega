import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class BiasVarianceOptimizedKNNEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, k_values=None, n_estimators=5, cv=5):
        self.k_values = k_values or [3, 5, 7, 9, 11]
        self.n_estimators = n_estimators
        self.cv = cv
        self.knn_classifiers = []
        self.meta_learner = LogisticRegression(multi_class='ovr', solver='lbfgs')
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train):
        # Scale the input features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Create and fit KNN classifiers
        self.knn_classifiers = [
            KNeighborsClassifier(n_neighbors=k)
            for k in self.k_values[:self.n_estimators]
        ]
        
        # Fit each KNN classifier
        for knn in self.knn_classifiers:
            knn.fit(X_scaled, y_train)
        
        # Generate meta-features using cross-validation
        meta_features = np.zeros((X_scaled.shape[0], len(self.knn_classifiers)))
        for i, knn in enumerate(self.knn_classifiers):
            meta_features[:, i] = cross_val_predict(knn, X_scaled, y_train, cv=self.cv, method='predict_proba')[:, 1]
        
        # Fit meta-learner
        self.meta_learner.fit(meta_features, y_train)
        
        return self

    def predict(self, X_test):
        # Scale the input features
        X_scaled = self.scaler.transform(X_test)
        
        # Generate predictions from base classifiers
        base_predictions = np.array([knn.predict_proba(X_scaled)[:, 1] for knn in self.knn_classifiers]).T
        
        # Estimate bias and variance for each classifier
        bias = np.abs(base_predictions - np.mean(base_predictions, axis=1, keepdims=True))
        variance = np.var(base_predictions, axis=1, keepdims=True)
        
        # Calculate weights based on bias-variance trade-off
        weights = 1 / (bias + variance + 1e-10)  # Add small constant to avoid division by zero
        weights /= np.sum(weights, axis=1, keepdims=True)  # Normalize weights
        
        # Apply weights to base predictions
        weighted_predictions = np.sum(base_predictions * weights, axis=1)
        
        # Use meta-learner for final prediction
        meta_features = base_predictions  # Use all base predictions as meta-features
        final_predictions = self.meta_learner.predict(meta_features)
        
        return final_predictions

    def predict_proba(self, X_test):
        # Scale the input features
        X_scaled = self.scaler.transform(X_test)
        
        # Generate predictions from base classifiers
        base_predictions = np.array([knn.predict_proba(X_scaled)[:, 1] for knn in self.knn_classifiers]).T
        
        # Estimate bias and variance for each classifier
        bias = np.abs(base_predictions - np.mean(base_predictions, axis=1, keepdims=True))
        variance = np.var(base_predictions, axis=1, keepdims=True)
        
        # Calculate weights based on bias-variance trade-off
        weights = 1 / (bias + variance + 1e-10)  # Add small constant to avoid division by zero
        weights /= np.sum(weights, axis=1, keepdims=True)  # Normalize weights
        
        # Apply weights to base predictions
        weighted_predictions = np.sum(base_predictions * weights, axis=1)
        
        # Use meta-learner for final prediction probabilities
        meta_features = base_predictions  # Use all base predictions as meta-features
        final_proba = self.meta_learner.predict_proba(meta_features)
        
        return final_proba