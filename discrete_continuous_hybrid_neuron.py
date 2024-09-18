import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class DiscreteContHybridNeuron(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(100,), learning_rate=0.01, max_iter=200, 
                 tol=1e-4, random_state=None, discrete_threshold=0.5):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.discrete_threshold = discrete_threshold

    def _initialize_parameters(self):
        np.random.seed(self.random_state)
        self.weights_ = [np.random.randn(self.n_features_, self.hidden_layer_sizes[0]) / np.sqrt(self.n_features_)]
        self.biases_ = [np.zeros((1, self.hidden_layer_sizes[0]))]
        
        for i in range(1, len(self.hidden_layer_sizes)):
            self.weights_.append(np.random.randn(self.hidden_layer_sizes[i-1], self.hidden_layer_sizes[i]) / np.sqrt(self.hidden_layer_sizes[i-1]))
            self.biases_.append(np.zeros((1, self.hidden_layer_sizes[i])))
        
        self.weights_.append(np.random.randn(self.hidden_layer_sizes[-1], self.n_classes_) / np.sqrt(self.hidden_layer_sizes[-1]))
        self.biases_.append(np.zeros((1, self.n_classes_)))

    def _hybrid_activation(self, z):
        continuous = 1 / (1 + np.exp(-np.clip(z, -709, 709)))
        discrete = (z > self.discrete_threshold).astype(float)
        return np.where(np.abs(z) > self.discrete_threshold, discrete, continuous)

    def _forward_pass(self, X):
        activations = [X]
        for i in range(len(self.weights_)):
            z = np.dot(activations[-1], self.weights_[i]) + self.biases_[i]
            a = self._hybrid_activation(z) if i < len(self.weights_) - 1 else self._softmax(z)
            activations.append(a)
        return activations

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _backward_pass(self, activations, y):
        m = y.shape[0]
        one_hot_y = np.eye(self.n_classes_)[y]
        
        dz = activations[-1] - one_hot_y
        for i in range(len(self.weights_) - 1, -1, -1):
            dw = np.dot(activations[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            
            self.weights_[i] -= self.learning_rate * dw
            self.biases_[i] -= self.learning_rate * db
            
            if i > 0:
                da = np.dot(dz, self.weights_[i].T)
                dz = da * (activations[i] * (1 - activations[i]))

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        
        self._initialize_parameters()
        
        for _ in range(self.max_iter):
            activations = self._forward_pass(X)
            self._backward_pass(activations, y)
            
            loss = -np.mean(np.log(activations[-1][np.arange(len(y)), y] + 1e-8))
            if loss < self.tol:
                break
        
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        activations = self._forward_pass(X)
        return self.classes_[np.argmax(activations[-1], axis=1)]

    def predict_proba(self, X):
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        activations = self._forward_pass(X)
        return activations[-1]