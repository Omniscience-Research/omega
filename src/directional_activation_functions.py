import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class DirectionalActivationNet(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(100,), learning_rate=0.01, max_iter=200, random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state

    def _directional_activation(self, X):
        """Custom directional activation function"""
        return np.tanh(X) * X

    def _forward_pass(self, X):
        """Perform forward pass through the network"""
        activations = [X]
        for i, layer in enumerate(self.weights):
            z = np.dot(activations[-1], layer) + self.biases[i]
            if i == len(self.weights) - 1:
                a = self._softmax(z)
            else:
                a = self._directional_activation(z)
            activations.append(a)
        return activations

    def _backward_pass(self, X, y, activations):
        """Perform backward pass and update weights"""
        m = X.shape[0]
        delta = activations[-1] - y
        for i in reversed(range(len(self.weights))):
            dW = np.dot(activations[i].T, delta) / m
            db = np.sum(delta, axis=0) / m
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * (1 - activations[i]**2) * activations[i]

    def _softmax(self, X):
        """Compute softmax activation"""
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)

    def fit(self, X, y):
        """Fit the DirectionalActivationNet to the training data"""
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # Initialize weights and biases
        np.random.seed(self.random_state)
        self.weights = []
        self.biases = []
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [n_classes]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i+1])))

        # One-hot encode the target values
        y_encoded = np.eye(n_classes)[y]

        # Training loop
        for _ in range(self.max_iter):
            activations = self._forward_pass(X)
            self._backward_pass(X, y_encoded, activations)

        return self

    def predict(self, X):
        """Predict class labels for samples in X"""
        check_is_fitted(self)
        X = check_array(X)
        activations = self._forward_pass(X)
        y_pred = np.argmax(activations[-1], axis=1)
        return self.classes_[y_pred]

    def predict_proba(self, X):
        """Predict class probabilities for samples in X"""
        check_is_fitted(self)
        X = check_array(X)
        activations = self._forward_pass(X)
        return activations[-1]