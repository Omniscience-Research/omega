import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.tree import DecisionTreeClassifier

class DirectionalForest(BaseEstimator, ClassifierMixin):
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
        
        # Calculate feature directionality
        self.feature_directions_ = self._calculate_feature_directions(X, y)
        
        # Create the forest
        self.estimators_ = []
        
        for _ in range(self.n_estimators):
            tree = self._grow_tree(X, y)
            self.estimators_.append(tree)
        
        return self
    
    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Apply directionality to features
        X_directional = X * self.feature_directions_
        
        # Make predictions
        predictions = np.array([tree.predict(X_directional) for tree in self.estimators_])
        
        # Return the most common prediction for each sample
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)
    
    def _grow_tree(self, X, y):
        # Create a decision tree
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state
        )
        
        # Apply directionality to features
        X_directional = X * self.feature_directions_
        
        # Fit the tree with directional features
        tree.fit(X_directional, y)
        
        return tree
    
    def _calculate_feature_directions(self, X, y):
        # Calculate the mean of each feature for each class
        class_means = [np.mean(X[y == c], axis=0) for c in self.classes_]
        
        # Calculate the overall mean of each feature
        overall_mean = np.mean(X, axis=0)
        
        # Calculate directionality based on the difference between class means and overall mean
        directions = np.sign(np.sum([cm - overall_mean for cm in class_means], axis=0))
        
        return directions