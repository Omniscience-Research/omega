import numpy as np
from scipy.spatial.distance import euclidean
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class HybridKNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, continuous_weight=0.5, discrete_weight=0.5):
        self.n_neighbors = n_neighbors
        self.continuous_weight = continuous_weight
        self.discrete_weight = discrete_weight
        self.X_train = None
        self.y_train = None
        self.continuous_scaler = StandardScaler()
        self.feature_types = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.feature_types = self._determine_feature_types(X_train)
        
        # Scale continuous features
        continuous_mask = [i for i, ft in enumerate(self.feature_types) if ft == 'continuous']
        self.continuous_scaler.fit(X_train[:, continuous_mask])
        
        return self

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            distances = self._calculate_distances(x)
            k_nearest_indices = np.argsort(distances)[:self.n_neighbors]
            k_nearest_labels = self.y_train[k_nearest_indices]
            y_pred.append(Counter(k_nearest_labels).most_common(1)[0][0])
        return np.array(y_pred)

    def _determine_feature_types(self, X):
        feature_types = []
        for col in X.T:
            if np.issubdtype(col.dtype, np.number) and len(np.unique(col)) > 10:
                feature_types.append('continuous')
            else:
                feature_types.append('discrete')
        return feature_types

    def _calculate_distances(self, x):
        distances = []
        for x_train in self.X_train:
            continuous_dist = self._continuous_distance(x, x_train)
            discrete_dist = self._discrete_distance(x, x_train)
            combined_dist = (self.continuous_weight * continuous_dist +
                             self.discrete_weight * discrete_dist)
            distances.append(combined_dist)
        return np.array(distances)

    def _continuous_distance(self, x1, x2):
        continuous_mask = [i for i, ft in enumerate(self.feature_types) if ft == 'continuous']
        x1_cont = self.continuous_scaler.transform([x1[continuous_mask]])[0]
        x2_cont = self.continuous_scaler.transform([x2[continuous_mask]])[0]
        return euclidean(x1_cont, x2_cont)

    def _discrete_distance(self, x1, x2):
        discrete_mask = [i for i, ft in enumerate(self.feature_types) if ft == 'discrete']
        return np.sum(x1[discrete_mask] != x2[discrete_mask])

    def get_params(self, deep=True):
        return {
            "n_neighbors": self.n_neighbors,
            "continuous_weight": self.continuous_weight,
            "discrete_weight": self.discrete_weight
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self