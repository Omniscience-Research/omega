import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class AbstractionGuidedEnsembleNet(BaseEstimator, ClassifierMixin):
    def __init__(self, n_networks=5, hidden_layer_sizes=(100,), max_iter=200, random_state=None):
        self.n_networks = n_networks
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self.networks = []
        self.meta_learner = None
        self.scaler = StandardScaler()

    def _create_network(self, hidden_layer_sizes):
        return MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=self.max_iter,
            random_state=self.random_state
        )

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=self.random_state)

        # Create and train base networks with varying levels of abstraction
        for i in range(self.n_networks):
            hidden_layers = tuple([max(1, self.hidden_layer_sizes[0] // (2 ** i))] * len(self.hidden_layer_sizes))
            network = self._create_network(hidden_layers)
            network.fit(X_train, y_train)
            self.networks.append(network)

        # Generate meta-features
        meta_features = np.column_stack([network.predict_proba(X_val) for network in self.networks])

        # Train meta-learner
        self.meta_learner = MLPClassifier(
            hidden_layer_sizes=(50,),
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.meta_learner.fit(meta_features, y_val)

        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        meta_features = np.column_stack([network.predict_proba(X_scaled) for network in self.networks])
        return self.meta_learner.predict(meta_features)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        meta_features = np.column_stack([network.predict_proba(X_scaled) for network in self.networks])
        return self.meta_learner.predict_proba(meta_features)