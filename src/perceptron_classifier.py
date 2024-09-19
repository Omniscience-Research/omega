import numpy as np

class PerceptronClassifier:
    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights = None
        self.bias = None

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        
        # Initialize random number generator
        rng = np.random.RandomState(self.random_state)
        
        # Initialize weights and bias
        self.weights = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias = 0
        
        # Training loop
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X_train):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._step_function(linear_output)
                
                # Update weights and bias
                update = self.learning_rate * (y_train[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
        
        return self

    def predict(self, X_test):
        linear_output = np.dot(X_test, self.weights) + self.bias
        y_predicted = self._step_function(linear_output)
        return y_predicted

    def _step_function(self, x):
        return np.where(x >= 0, 1, 0)