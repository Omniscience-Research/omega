import numpy as np
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self):
        self.class_priors = {}
        self.feature_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.classes = None
        self.n_features = None

    def fit(self, X_train, y_train):
        """
        Fit the Naive Bayes classifier using training data.

        Parameters:
        X_train (array-like): Training feature vectors
        y_train (array-like): Training labels
        """
        n_samples, self.n_features = X_train.shape
        self.classes = np.unique(y_train)

        # Calculate class priors
        for c in self.classes:
            self.class_priors[c] = np.sum(y_train == c) / n_samples

        # Calculate feature probabilities for each class
        for c in self.classes:
            X_c = X_train[y_train == c]
            for feature in range(self.n_features):
                feature_values = np.unique(X_c[:, feature])
                for value in feature_values:
                    count = np.sum(X_c[:, feature] == value)
                    self.feature_probs[c][feature][value] = (count + 1) / (len(X_c) + len(feature_values))

    def predict(self, X_test):
        """
        Predict class labels for the input samples.

        Parameters:
        X_test (array-like): Test feature vectors

        Returns:
        array-like: Predicted class labels
        """
        y_pred = []
        for x in X_test:
            class_scores = {}
            for c in self.classes:
                class_scores[c] = np.log(self.class_priors[c])
                for feature, value in enumerate(x):
                    class_scores[c] += np.log(self.feature_probs[c][feature].get(value, 1e-10))
            y_pred.append(max(class_scores, key=class_scores.get))
        return np.array(y_pred)