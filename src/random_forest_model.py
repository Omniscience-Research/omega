import numpy as np
from collections import Counter

class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        if (depth == self.max_depth or n_samples < self.min_samples_split or n_classes == 1):
            return DecisionTreeNode(value=Counter(y).most_common(1)[0][0])

        feature_indices = np.random.choice(n_features, n_features, replace=False)

        best_feature, best_threshold = self._best_split(X, y, feature_indices)

        left_indices = X[:, best_feature] < best_threshold
        right_indices = ~left_indices

        left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return DecisionTreeNode(feature_index=best_feature, threshold=best_threshold, left=left, right=right)

    def _best_split(self, X, y, feature_indices):
        best_gain = -1
        best_feature, best_threshold = None, None

        for feature_index in feature_indices:
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature_index], threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)

        left_indices = X_column < threshold
        right_indices = ~left_indices

        n = len(y)
        n_left, n_right = np.sum(left_indices), np.sum(right_indices)
        e_left, e_right = self._entropy(y[left_indices]), self._entropy(y[right_indices])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        return parent_entropy - child_entropy

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def predict(self, X):
        return np.array([self._predict_tree(x, self.root) for x in X])

    def _predict_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature_index] < node.threshold:
            return self._predict_tree(x, node.left)
        else:
            return self._predict_tree(x, node.right)

class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X_train, y_train):
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            X_sample, y_sample = self._bootstrap_sample(X_train, y_train)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X_test):
        tree_predictions = np.array([tree.predict(X_test) for tree in self.trees])
        return np.array([Counter(predictions).most_common(1)[0][0] for predictions in tree_predictions.T])

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]