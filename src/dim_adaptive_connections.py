import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.exceptions import NotFittedError

class DimensionalityAdaptiveConnector(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neurons=100, learning_rate=0.01, n_iterations=1000, 
                 adaptation_rate=0.001, min_dim=1, max_dim=10, random_state=None):
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.adaptation_rate = adaptation_rate
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.random_state = random_state

    def _initialize_weights(self, n_features):
        rng = np.random.RandomState(self.random_state)
        self.weights_ = rng.randn(self.n_neurons, n_features)
        self.biases_ = np.zeros((self.n_neurons, 1))
        self.output_weights_ = rng.randn(1, self.n_neurons)
        self.output_bias_ = np.zeros((1, 1))
        self.dimensions_ = np.ones(self.n_neurons) * self.max_dim

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -709, 709)))

    def _forward_pass(self, X):
        weighted_sum = np.dot(self.weights_, X.T) + self.biases_
        activations = self._sigmoid(weighted_sum)
        output = np.dot(self.output_weights_, activations) + self.output_bias_
        return activations, output

    def _backward_pass(self, X, y, activations, output):
        m = X.shape[0]
        d_output = output - y
        d_output_weights = np.dot(d_output, activations.T) / m
        d_output_bias = np.sum(d_output, axis=1, keepdims=True) / m
        d_hidden = np.dot(self.output_weights_.T, d_output) * activations * (1 - activations)
        d_weights = np.dot(d_hidden, X) / m
        d_biases = np.sum(d_hidden, axis=1, keepdims=True) / m
        return d_output_weights, d_output_bias, d_weights, d_biases

    def _adapt_dimensions(self, d_weights):
        grad_norm = np.linalg.norm(d_weights, axis=1)
        self.dimensions_ += self.adaptation_rate * (grad_norm - self.dimensions_)
        self.dimensions_ = np.clip(self.dimensions_, self.min_dim, self.max_dim)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        
        if len(self.classes_) != 2:
            raise ValueError("This classifier only supports binary classification.")
        
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)
        
        y = (y == self.classes_[1]).astype(int)
        
        for _ in range(self.n_iterations):
            activations, output = self._forward_pass(X)
            d_output_weights, d_output_bias, d_weights, d_biases = self._backward_pass(X, y, activations, output)
            
            self.output_weights_ -= self.learning_rate * d_output_weights
            self.output_bias_ -= self.learning_rate * d_output_bias
            self.weights_ -= self.learning_rate * d_weights
            self.biases_ -= self.learning_rate * d_biases
            
            self._adapt_dimensions(d_weights)
            self.weights_ *= np.expand_dims(self.dimensions_, axis=1)
        
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        _, output = self._forward_pass(X)
        probabilities = self._sigmoid(output.T)
        return np.column_stack((1 - probabilities, probabilities))

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]