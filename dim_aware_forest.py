import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class DimAwareForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_features='sqrt', min_samples_leaf=1, 
                 max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        
    def _adapt_feature_selection(self, X, y):
        n_samples, n_features = X.shape
        
        if n_features > n_samples:
            # High-dimensional case: use mutual information for feature selection
            mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
            selected_features = np.argsort(mi_scores)[::-1][:int(np.sqrt(n_features))]
        else:
            # Low-dimensional case: use all features
            selected_features = np.arange(n_features)
        
        return selected_features
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        self.selected_features_ = self._adapt_feature_selection(X, y)
        X_selected = X[:, self.selected_features_]
        
        self.estimators_ = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_features=self.max_features,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            tree.fit(X_selected, y)
            self.estimators_.append(tree)
        
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        X_selected = X[:, self.selected_features_]
        
        predictions = np.array([tree.predict(X_selected) for tree in self.estimators_])
        maj_vote = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x, minlength=self.n_classes_)),
            axis=0,
            arr=predictions
        )
        
        return self.classes_[maj_vote]

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        X_selected = X[:, self.selected_features_]
        
        probas = np.array([tree.predict_proba(X_selected) for tree in self.estimators_])
        avg_proba = np.mean(probas, axis=0)
        
        return avg_proba