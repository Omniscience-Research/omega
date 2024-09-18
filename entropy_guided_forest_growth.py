import numpy as np
from scipy.stats import entropy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class EntropyGuidedForestGrowth(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features='sqrt', random_state=None,
                 entropy_threshold=1e-4):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.entropy_threshold = entropy_threshold
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.estimators_ = []
        
        current_entropy = np.inf
        
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state
            )
            
            tree.fit(X, y)
            self.estimators_.append(tree)
            
            y_pred = self.predict(X)
            new_entropy = self._calculate_entropy(y, y_pred)
            
            if new_entropy > current_entropy or (current_entropy - new_entropy) < self.entropy_threshold:
                self.estimators_.pop()
                break
            
            current_entropy = new_entropy
        
        self.n_estimators_ = len(self.estimators_)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        all_proba = np.array([tree.predict_proba(X) for tree in self.estimators_])
        return np.mean(all_proba, axis=0)

    def _calculate_entropy(self, y_true, y_pred):
        cm = np.zeros((self.n_classes_, self.n_classes_))
        for i in range(len(y_true)):
            cm[y_true[i]][y_pred[i]] += 1
        
        cm = cm / cm.sum(axis=1, keepdims=True)
        entropies = np.apply_along_axis(entropy, axis=1, arr=cm)
        return np.mean(entropies)