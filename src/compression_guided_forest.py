import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import entropy

class CompressionGuidedForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        
    def _calculate_compression_gain(self, y):
        """Calculate compression gain based on entropy reduction."""
        parent_entropy = entropy(np.bincount(y) / len(y), base=2)
        return parent_entropy
    
    def _build_tree(self, X, y):
        """Build a single decision tree guided by compression gain."""
        tree = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state
        )
        
        # Custom splitting criterion based on compression gain
        def custom_criterion(y, **kwargs):
            return -self._calculate_compression_gain(y)  # Negative because sklearn minimizes
        
        tree._criterion = custom_criterion
        tree.fit(X, y)
        return tree
    
    def fit(self, X, y):
        """
        Build a forest of trees from the training set (X, y).
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        
        Returns:
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        
        self.estimators_ = []
        for _ in range(self.n_estimators):
            tree = self._build_tree(X, y)
            self.estimators_.append(tree)
        
        return self
    
    def predict(self, X):
        """
        Predict class for X.
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        Returns:
        y : array-like of shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        y_pred = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x, minlength=self.n_classes_)),
            axis=0,
            arr=predictions
        )
        
        return self.classes_[y_pred]

    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        Returns:
        p : array of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        all_proba = np.array([tree.predict_proba(X) for tree in self.estimators_])
        avg_proba = np.mean(all_proba, axis=0)
        
        return avg_proba