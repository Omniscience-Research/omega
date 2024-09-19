import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class AdaptiveDensityCompressedKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, compression_factor=0.1, density_threshold=0.5):
        self.n_neighbors = n_neighbors
        self.compression_factor = compression_factor
        self.density_threshold = density_threshold
        self.X_compressed = None
        self.y_compressed = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        # Standardize the input features
        X_scaled = self.scaler.fit_transform(X)
        
        # Compute local density for each point
        nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        nn.fit(X_scaled)
        distances, _ = nn.kneighbors(X_scaled)
        local_density = 1 / np.mean(distances, axis=1)
        
        # Normalize local density
        local_density = (local_density - local_density.min()) / (local_density.max() - local_density.min())
        
        # Determine compression level for each point
        compression_level = np.where(local_density > self.density_threshold,
                                     self.compression_factor * local_density,
                                     self.compression_factor * self.density_threshold)
        
        # Compress the data
        X_compressed = []
        y_compressed = []
        
        for i in range(len(X_scaled)):
            if np.random.random() < compression_level[i]:
                X_compressed.append(X_scaled[i])
                y_compressed.append(y[i])
        
        self.X_compressed = np.array(X_compressed)
        self.y_compressed = np.array(y_compressed)
        
        return self

    def predict(self, X):
        if self.X_compressed is None or self.y_compressed is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' before using 'predict'.")
        
        X_scaled = self.scaler.transform(X)
        
        nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        nn.fit(self.X_compressed)
        
        _, indices = nn.kneighbors(X_scaled)
        
        predictions = []
        for neighbors in indices:
            neighbor_labels = self.y_compressed[neighbors]
            prediction = np.argmax(np.bincount(neighbor_labels))
            predictions.append(prediction)
        
        return np.array(predictions)