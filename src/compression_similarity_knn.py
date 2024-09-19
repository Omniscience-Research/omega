import numpy as np
from collections import defaultdict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class CompressibilitySimilarityKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

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

        # Compute similarities
        similarities = self._compute_similarities(X)
        
        # Find k nearest neighbors
        neighbors = similarities.argsort()[:, -self.n_neighbors:]
        
        # Predict labels
        y_pred = np.array([
            np.argmax(np.bincount(self.y_[neighbor]))
            for neighbor in neighbors
        ])
        
        return y_pred

    def _compute_similarities(self, X):
        similarities = np.zeros((X.shape[0], self.X_.shape[0]))
        for i, x in enumerate(X):
            for j, x_train in enumerate(self.X_):
                similarities[i, j] = self._compressibility_similarity(x, x_train)
        return similarities

    def _compressibility_similarity(self, x1, x2):
        concat = np.concatenate([x1, x2])
        compressed_size = len(self._lzw_compress(concat))
        return 1 / compressed_size  # Higher similarity for smaller compressed size

    def _lzw_compress(self, data):
        # Convert data to string for compression
        data_str = ''.join(map(str, data))
        
        dict_size = 256
        dictionary = {chr(i): i for i in range(dict_size)}

        w = ""
        result = []
        for c in data_str:
            wc = w + c
            if wc in dictionary:
                w = wc
            else:
                result.append(dictionary[w])
                dictionary[wc] = dict_size
                dict_size += 1
                w = c

        if w:
            result.append(dictionary[w])

        return result