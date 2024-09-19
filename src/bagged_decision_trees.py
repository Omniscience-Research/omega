import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

class BaggedDecisionTrees:
    def __init__(self, n_estimators=10, max_samples=1.0, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []

    def fit(self, X_train, y_train):
        """
        Fit the bagged decision trees on the training data.

        Parameters:
        X_train (array-like): Training input samples.
        y_train (array-like): Target values.

        Returns:
        self: Returns an instance of self.
        """
        n_samples = X_train.shape[0]
        
        for _ in range(self.n_estimators):
            # Create a bootstrap sample
            if isinstance(self.max_samples, float):
                n_samples_bootstrap = int(self.max_samples * n_samples)
            else:
                n_samples_bootstrap = self.max_samples

            X_bootstrap, y_bootstrap = resample(X_train, y_train, 
                                                n_samples=n_samples_bootstrap,
                                                random_state=self.random_state)
            
            # Train a decision tree on the bootstrap sample
            tree = DecisionTreeClassifier(max_depth=self.max_depth, 
                                          random_state=self.random_state)
            tree.fit(X_bootstrap, y_bootstrap)
            
            # Add the trained tree to our ensemble
            self.trees.append(tree)

        return self

    def predict(self, X_test):
        """
        Predict class for X_test.

        Parameters:
        X_test (array-like): The input samples.

        Returns:
        y_pred (array-like): The predicted class labels.
        """
        # Get predictions from all trees
        tree_preds = np.array([tree.predict(X_test) for tree in self.trees])
        
        # Use majority voting to get final predictions
        y_pred = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)), 
            axis=0, 
            arr=tree_preds)
        
        return y_pred