import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class DynamicBiasVarianceForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        
    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Initialize trees and out-of-bag scores
        self.estimators_ = []
        self.oob_scores_ = []
        
        n_samples = X.shape[0]
        
        for i in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            oob_indices = np.setdiff1d(np.arange(n_samples), np.unique(indices))
            
            # Train a decision tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state
            )
            tree.fit(X[indices], y[indices])
            self.estimators_.append(tree)
            
            # Calculate out-of-bag score
            if len(oob_indices) > 0:
                oob_pred = tree.predict(X[oob_indices])
                oob_score = np.mean(oob_pred == y[oob_indices])
                self.oob_scores_.append(oob_score)
            
            # Dynamically adjust max_depth based on OOB score
            if len(self.oob_scores_) > 1:
                if self.oob_scores_[-1] < self.oob_scores_[-2]:
                    if self.max_depth is not None:
                        self.max_depth = max(self.max_depth - 1, 1)
                else:
                    if self.max_depth is None:
                        self.max_depth = 1
                    else:
                        self.max_depth += 1
            
            # Dynamically adjust number of trees
            if i > 10 and np.mean(self.oob_scores_[-10:]) < np.mean(self.oob_scores_[-20:-10]):
                break
        
        return self
    
    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['estimators_', 'classes_'])
        
        # Input validation
        X = check_array(X)
        
        # Collect predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        
        # Return the class with the most votes
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['estimators_', 'classes_'])
        
        # Input validation
        X = check_array(X)
        
        # Collect probability predictions from all trees
        all_proba = np.array([tree.predict_proba(X) for tree in self.estimators_])
        
        # Average probabilities across all trees
        return np.mean(all_proba, axis=0)