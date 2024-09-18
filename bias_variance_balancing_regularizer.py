import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

class BiasVarianceBalancer(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10, max_depth_range=(1, 10), validation_split=0.2, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth_range = max_depth_range
        self.validation_split = validation_split
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_split, random_state=self.random_state
        )

        best_depth = None
        best_score = float('inf')

        for depth in range(self.max_depth_range[0], self.max_depth_range[1] + 1):
            estimators = [
                DecisionTreeClassifier(max_depth=depth, random_state=self.random_state)
                for _ in range(self.n_estimators)
            ]
            
            for estimator in estimators:
                estimator.fit(X_train, y_train)

            y_pred = np.array([estimator.predict(X_val) for estimator in estimators])
            y_pred_mean = np.mean(y_pred, axis=0)

            bias = np.mean((y_val - y_pred_mean)**2)
            variance = np.mean(np.var(y_pred, axis=0))
            
            balance_score = bias + variance

            if balance_score < best_score:
                best_score = balance_score
                best_depth = depth

        self.estimators_ = [
            DecisionTreeClassifier(max_depth=best_depth, random_state=self.random_state)
            for _ in range(self.n_estimators)
        ]
        
        for estimator in self.estimators_:
            estimator.fit(X, y)

        self.best_depth_ = best_depth
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)

        probas = np.array([estimator.predict_proba(X) for estimator in self.estimators_])
        return np.mean(probas, axis=0)