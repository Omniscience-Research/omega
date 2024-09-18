import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class SimilarityFeatureFuser(BaseEstimator, ClassifierMixin):
    def __init__(self, n_clusters=10, similarity_threshold=0.8, random_state=None):
        self.n_clusters = n_clusters
        self.similarity_threshold = similarity_threshold
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.feature_groups = None
        self.fused_features = None
    
    def fit(self, X_train, y_train):
        # Standardize the features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Compute feature similarity matrix
        similarity_matrix = cosine_similarity(X_scaled.T)
        
        # Group similar features
        self.feature_groups = self._group_similar_features(similarity_matrix)
        
        # Fuse similar features
        X_fused = self._fuse_features(X_scaled)
        
        # Apply K-means clustering on fused features
        self.kmeans.fit(X_fused)
        
        return self
    
    def predict(self, X_test):
        X_scaled = self.scaler.transform(X_test)
        X_fused = self._fuse_features(X_scaled)
        return self.kmeans.predict(X_fused)
    
    def _group_similar_features(self, similarity_matrix):
        n_features = similarity_matrix.shape[0]
        feature_groups = []
        used_features = set()
        
        for i in range(n_features):
            if i not in used_features:
                group = {i}
                for j in range(i+1, n_features):
                    if j not in used_features and similarity_matrix[i, j] >= self.similarity_threshold:
                        group.add(j)
                
                feature_groups.append(list(group))
                used_features.update(group)
        
        return feature_groups
    
    def _fuse_features(self, X):
        fused_features = []
        
        for group in self.feature_groups:
            if len(group) == 1:
                fused_features.append(X[:, group[0]])
            else:
                fused_feature = np.mean(X[:, group], axis=1)
                fused_features.append(fused_feature)
        
        return np.column_stack(fused_features)

# Example usage:
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=50, n_informative=10, n_redundant=10, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the SimilarityFeatureFuser
fuser = SimilarityFeatureFuser(n_clusters=2, similarity_threshold=0.8, random_state=42)
fuser.fit(X_train, y_train)

# Make predictions on the test set
y_pred = fuser.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")