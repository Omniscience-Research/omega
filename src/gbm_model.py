import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, ClassifierMixin

class GradientBoostingMachine(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators_ = []
        
    def _logistic_loss(self, y, pred):
        return np.log(1 + np.exp(-y * pred))
    
    def _negative_gradient(self, y, pred):
        return y / (1 + np.exp(y * pred))
    
    def fit(self, X, y):
        # Initialize predictions
        y = np.where(y <= 0, -1, 1)  # Convert to -1 and 1
        F = np.zeros(len(y))
        
        for _ in range(self.n_estimators):
            negative_gradient = self._negative_gradient(y, F)
            
            # Fit a regression tree to the negative gradient
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X, negative_gradient)
            
            # Update F
            update = self.learning_rate * tree.predict(X)
            F += update
            
            # Store the estimator
            self.estimators_.append(tree)
        
        return self
    
    def predict_proba(self, X):
        # Compute raw predictions
        F = np.zeros((X.shape[0], 2))
        for estimator in self.estimators_:
            F[:, 1] += self.learning_rate * estimator.predict(X)
        F[:, 0] = -F[:, 1]
        
        # Apply sigmoid function to get probabilities
        proba = 1 / (1 + np.exp(-F))
        return proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

# Example usage:
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Generate a random binary classification problem
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the GradientBoostingMachine
    gbm = GradientBoostingMachine(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbm.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = gbm.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")