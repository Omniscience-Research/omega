import numpy as np
from scipy.special import expit

class NeuralNetClassifier:
    def __init__(self, hidden_layer_sizes=(100,), learning_rate=0.001, max_iter=200, random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.weights = []
        self.biases = []
        
    def _initialize_parameters(self, n_features, n_classes):
        np.random.seed(self.random_state)
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [n_classes]
        
        for i in range(1, len(layer_sizes)):
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i])))
    
    def _forward_propagation(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = expit(z) if i < len(self.weights) - 1 else self._softmax(z)
            activations.append(a)
        return activations
    
    def _backward_propagation(self, X, y, activations):
        m = X.shape[0]
        one_hot_y = self._one_hot_encode(y)
        
        dZ = activations[-1] - one_hot_y
        dW = [np.dot(activations[-2].T, dZ) / m]
        db = [np.sum(dZ, axis=0, keepdims=True) / m]
        
        for i in reversed(range(1, len(self.weights))):
            dA = np.dot(dZ, self.weights[i].T)
            dZ = dA * activations[i] * (1 - activations[i])
            dW.insert(0, np.dot(activations[i-1].T, dZ) / m)
            db.insert(0, np.sum(dZ, axis=0, keepdims=True) / m)
        
        return dW, db
    
    def _update_parameters(self, dW, db):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _one_hot_encode(self, y):
        n_classes = len(np.unique(y))
        return np.eye(n_classes)[y]
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        self._initialize_parameters(n_features, n_classes)
        
        for _ in range(self.max_iter):
            activations = self._forward_propagation(X)
            dW, db = self._backward_propagation(X, y, activations)
            self._update_parameters(dW, db)
        
        return self
    
    def predict(self, X):
        activations = self._forward_propagation(X)
        y_pred = np.argmax(activations[-1], axis=1)
        return y_pred