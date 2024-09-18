import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mutual_info_score

class DimAwareSplitInteractionModel(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1, 
                 interaction_threshold=0.1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.interaction_threshold = interaction_threshold
        self.tree = None
        self.feature_interactions = None

    def _calculate_feature_interactions(self, X):
        n_features = X.shape[1]
        interactions = np.zeros((n_features, n_features))

        for i in range(n_features):
            for j in range(i+1, n_features):
                mi = mutual_info_score(X[:, i], X[:, j])
                interactions[i, j] = mi
                interactions[j, i] = mi

        return interactions

    def _custom_criterion(self, y, X):
        impurity = self._gini(y)
        n_features = X.shape[1]
        interaction_penalty = 0

        for i in range(n_features):
            for j in range(i+1, n_features):
                if self.feature_interactions[i, j] > self.interaction_threshold:
                    interaction_penalty += self.feature_interactions[i, j]

        return impurity + interaction_penalty

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]

        # Calculate feature interactions
        self.feature_interactions = self._calculate_feature_interactions(X)

        # Create and fit the decision tree with custom criterion
        self.tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            criterion='gini'  # We'll use gini and add our custom criterion in the splitting process
        )

        # Override the splitting criterion
        def custom_criterion(self, y, **kwargs):
            return self._custom_criterion(y, kwargs['X'])

        self.tree.criterion = custom_criterion.__get__(self.tree)
        
        # Fit the tree
        self.tree.fit(X, y)

        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Predict
        return self.tree.predict(X)

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Predict probabilities
        return self.tree.predict_proba(X)