import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy

class CompressionGuidedPruner(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', 
                 learning_rate_init=0.001, max_iter=200, 
                 compression_ratio=0.5, n_pruning_iterations=5):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.compression_ratio = compression_ratio
        self.n_pruning_iterations = n_pruning_iterations
        
    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = np.unique(y)
        
        # Preprocess the data
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Initialize and fit the MLP
        self.mlp_ = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes,
                                  activation=self.activation,
                                  learning_rate_init=self.learning_rate_init,
                                  max_iter=self.max_iter)
        
        self.mlp_.fit(X_scaled, y)
        
        # Perform iterative pruning
        for _ in range(self.n_pruning_iterations):
            self._prune_neurons(X_scaled)
            self.mlp_.fit(X_scaled, y)
        
        return self
    
    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['mlp_', 'scaler_'])
        
        # Input validation
        X = check_array(X)
        
        # Preprocess the data
        X_scaled = self.scaler_.transform(X)
        
        # Predict
        return self.mlp_.predict(X_scaled)
    
    def _prune_neurons(self, X):
        weights = self.mlp_.coefs_
        
        for layer_idx in range(len(weights) - 1):
            layer_weights = weights[layer_idx]
            n_neurons = layer_weights.shape[1]
            
            importance = self._compute_neuron_importance(X, layer_idx)
            n_keep = int(n_neurons * (1 - self.compression_ratio))
            keep_indices = np.argsort(importance)[-n_keep:]
            
            weights[layer_idx] = layer_weights[:, keep_indices]
            weights[layer_idx + 1] = weights[layer_idx + 1][keep_indices, :]
            
            if hasattr(self.mlp_, 'intercepts_'):
                self.mlp_.intercepts_[layer_idx] = self.mlp_.intercepts_[layer_idx][keep_indices]
        
        self.mlp_.coefs_ = weights
    
    def _compute_neuron_importance(self, X, layer_idx):
        activations = self._get_layer_activations(X, layer_idx)
        importance = np.apply_along_axis(lambda x: entropy(x), 0, activations)
        return importance
    
    def _get_layer_activations(self, X, layer_idx):
        activations = X
        for i in range(layer_idx + 1):
            activations = self.mlp_.activation(
                np.dot(activations, self.mlp_.coefs_[i]) + self.mlp_.intercepts_[i]
            )
        return activations