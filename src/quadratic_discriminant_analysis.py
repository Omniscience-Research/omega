import numpy as np
from scipy.stats import multivariate_normal

class QuadraticDiscriminantAnalysis:
    def __init__(self):
        self.classes = None
        self.priors = None
        self.means = None
        self.covariances = None

    def fit(self, X_train, y_train):
        """
        Fit the QDA model to the training data.

        Parameters:
        X_train (array-like): Training data of shape (n_samples, n_features)
        y_train (array-like): Target values of shape (n_samples,)
        """
        self.classes = np.unique(y_train)
        n_classes = len(self.classes)
        n_features = X_train.shape[1]

        self.priors = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_features))
        self.covariances = np.zeros((n_classes, n_features, n_features))

        for i, c in enumerate(self.classes):
            X_c = X_train[y_train == c]
            self.priors[i] = X_c.shape[0] / X_train.shape[0]
            self.means[i] = np.mean(X_c, axis=0)
            self.covariances[i] = np.cov(X_c, rowvar=False)

    def predict(self, X_test):
        """
        Predict class labels for samples in X_test.

        Parameters:
        X_test (array-like): Samples to predict, of shape (n_samples, n_features)

        Returns:
        array-like: Predicted class labels for each sample
        """
        n_samples = X_test.shape[0]
        n_classes = len(self.classes)
        log_probs = np.zeros((n_samples, n_classes))

        for i, c in enumerate(self.classes):
            mvn = multivariate_normal(mean=self.means[i], cov=self.covariances[i])
            log_probs[:, i] = np.log(self.priors[i]) + mvn.logpdf(X_test)

        return self.classes[np.argmax(log_probs, axis=1)]

    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in X_test.

        Parameters:
        X_test (array-like): Samples to predict, of shape (n_samples, n_features)

        Returns:
        array-like: Predicted class probabilities for each sample
        """
        n_samples = X_test.shape[0]
        n_classes = len(self.classes)
        log_probs = np.zeros((n_samples, n_classes))

        for i, c in enumerate(self.classes):
            mvn = multivariate_normal(mean=self.means[i], cov=self.covariances[i])
            log_probs[:, i] = np.log(self.priors[i]) + mvn.logpdf(X_test)

        # Normalize probabilities
        probs = np.exp(log_probs - np.max(log_probs, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)

        return probs