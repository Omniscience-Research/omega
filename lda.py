import numpy as np
from scipy.linalg import eigh

class LinearDiscriminantAnalysis:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.classes = None
        self.means = None
        self.priors = None
        self.eigenvectors = None
        self.scalings = None

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.classes = np.unique(y_train)
        n_classes = len(self.classes)

        if self.n_components is None:
            self.n_components = min(n_features, n_classes - 1)

        # Compute class means and priors
        self.means = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        for idx, cls in enumerate(self.classes):
            X_cls = X_train[y_train == cls]
            self.means[idx] = X_cls.mean(axis=0)
            self.priors[idx] = X_cls.shape[0] / n_samples

        # Compute within-class scatter matrix
        S_w = np.zeros((n_features, n_features))
        for idx, cls in enumerate(self.classes):
            X_cls = X_train[y_train == cls]
            X_centered = X_cls - self.means[idx]
            S_w += X_centered.T @ X_centered

        # Compute between-class scatter matrix
        overall_mean = X_train.mean(axis=0)
        S_b = np.zeros((n_features, n_features))
        for idx, cls in enumerate(self.classes):
            n_samples_cls = np.sum(y_train == cls)
            mean_diff = self.means[idx] - overall_mean
            S_b += n_samples_cls * np.outer(mean_diff, mean_diff)

        # Solve the generalized eigenvalue problem
        eigvals, eigvecs = eigh(S_b, S_w)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
        
        # Select the top n_components eigenvectors
        self.eigenvectors = eigvecs[:, :self.n_components]
        
        # Compute scaling
        self.scalings = np.dot(self.eigenvectors.T, S_w) @ self.eigenvectors
        self.scalings = np.linalg.inv(np.sqrt(self.scalings))
        
        return self

    def predict(self, X_test):
        # Project data onto LDA space
        X_lda = np.dot(X_test - self.means.mean(axis=0), self.eigenvectors)
        X_lda = np.dot(X_lda, self.scalings)
        
        # Compute log-likelihood for each class
        log_likelihood = np.zeros((X_test.shape[0], len(self.classes)))
        for idx, cls in enumerate(self.classes):
            mean_lda = np.dot(self.means[idx] - self.means.mean(axis=0), self.eigenvectors)
            mean_lda = np.dot(mean_lda, self.scalings)
            log_likelihood[:, idx] = (
                -0.5 * np.sum(np.square(X_lda - mean_lda), axis=1)
                + np.log(self.priors[idx])
            )
        
        # Return predicted class
        return self.classes[np.argmax(log_likelihood, axis=1)]