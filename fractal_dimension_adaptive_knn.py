import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import euclidean_distances

class FractalDimensionAdaptiveKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, k_max=20, box_sizes=None, adaptive_method='set_k', min_fd=1, max_fd=2):
        self.k_max = k_max
        self.box_sizes = box_sizes if box_sizes is not None else [1, 2, 4, 8, 16]
        self.adaptive_method = adaptive_method
        self.min_fd = min_fd
        self.max_fd = max_fd

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        distances = euclidean_distances(X, self.X_)
        y_pred = np.zeros(X.shape[0], dtype=self.y_.dtype)

        for i, dist in enumerate(distances):
            fd = self._estimate_fractal_dimension(dist)
            fd = np.clip(fd, self.min_fd, self.max_fd)

            if self.adaptive_method == 'set_k':
                k = max(1, min(int(self.k_max * fd), self.k_max))
                k_nearest = np.argsort(dist)[:k]
                y_pred[i] = np.argmax(np.bincount(self.y_[k_nearest]))
            elif self.adaptive_method == 'weight':
                weights = np.power(dist + 1e-10, -fd)
                y_pred[i] = np.argmax(np.bincount(self.y_, weights=weights))

        return y_pred

    def _estimate_fractal_dimension(self, distances):
        counts = []
        for size in self.box_sizes:
            count = np.sum(distances <= size)
            counts.append(float(count))  # Ensure counts are floats
        
        log_counts = np.log(counts)
        log_sizes = np.log(self.box_sizes)
        
        # Use only finite and positive values for linear regression
        valid_mask = np.isfinite(log_counts) & np.isfinite(log_sizes) & (counts > 0)
        if np.sum(valid_mask) < 2:
            return self.min_fd  # Return minimum fd if not enough valid points

        slope, _ = np.polyfit(log_sizes[valid_mask], log_counts[valid_mask], 1)
        return max(slope, 0)  # Ensure non-negative fractal dimension