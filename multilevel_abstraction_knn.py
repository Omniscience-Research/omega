import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class MultiLevelAbstractionKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, n_levels=3, abstraction_ratios=None):
        self.n_neighbors = n_neighbors
        self.n_levels = n_levels
        self.abstraction_ratios = abstraction_ratios or [0.1, 0.3, 1.0]
        self.models = []
        self.scalers = []
        self.pcas = []

    def fit(self, X_train, y_train):
        self.models = []
        self.scalers = []
        self.pcas = []
        
        for level in range(self.n_levels):
            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            self.scalers.append(scaler)
            
            # Apply PCA for feature abstraction
            n_components = max(1, int(X_train.shape[1] * self.abstraction_ratios[level]))
            pca = PCA(n_components=n_components)
            X_abstracted = pca.fit_transform(X_scaled)
            self.pcas.append(pca)
            
            # Train KNN model for this level
            knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
            knn.fit(X_abstracted, y_train)
            self.models.append(knn)
        
        return self

    def predict(self, X_test):
        predictions = []
        
        for level in range(self.n_levels):
            # Scale the features
            X_scaled = self.scalers[level].transform(X_test)
            
            # Apply PCA for feature abstraction
            X_abstracted = self.pcas[level].transform(X_scaled)
            
            # Get predictions for this level
            level_predictions = self.models[level].predict(X_abstracted)
            predictions.append(level_predictions)
        
        # Combine predictions from all levels
        final_predictions = self._combine_predictions(predictions)
        return final_predictions

    def _combine_predictions(self, predictions):
        # Simple majority voting to combine predictions
        stacked_predictions = np.stack(predictions, axis=1)
        final_predictions = []
        
        for sample_predictions in stacked_predictions:
            unique, counts = np.unique(sample_predictions, return_counts=True)
            final_predictions.append(unique[np.argmax(counts)])
        
        return np.array(final_predictions)