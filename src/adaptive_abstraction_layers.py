import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class AdaptiveAbstractionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_abstractions=3):
        super(AdaptiveAbstractionLayer, self).__init__()
        self.num_abstractions = num_abstractions
        self.layers = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(num_abstractions)])
        self.abstraction_weights = nn.Parameter(torch.ones(num_abstractions) / num_abstractions)

    def forward(self, x):
        outputs = [layer(x) for layer in self.layers]
        weighted_outputs = [w * out for w, out in zip(self.abstraction_weights, outputs)]
        return sum(weighted_outputs)

class AdaptiveAbstractionNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, num_abstractions=3):
        super(AdaptiveAbstractionNet, self).__init__()
        self.layers = nn.ModuleList()
        
        for i, (in_dim, out_dim) in enumerate(zip([input_dim] + hidden_dims[:-1], hidden_dims)):
            self.layers.append(AdaptiveAbstractionLayer(in_dim, out_dim, num_abstractions))
            self.layers.append(nn.ReLU())
        
        self.output_layer = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)

class AdaptiveAbstractionNetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_dims=[64, 32], num_abstractions=3, learning_rate=0.001, epochs=100, batch_size=32):
        self.hidden_dims = hidden_dims
        self.num_abstractions = num_abstractions
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        self.n_classes_ = len(self.classes_)

        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        # Initialize the network
        self.network_ = AdaptiveAbstractionNet(self.n_features_in_, self.hidden_dims, self.n_classes_, self.num_abstractions)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.network_.parameters(), lr=self.learning_rate)

        # Training loop
        self.network_.train()
        for epoch in range(self.epochs):
            for i in range(0, len(X), self.batch_size):
                batch_X = X_tensor[i:i+self.batch_size]
                batch_y = y_tensor[i:i+self.batch_size]

                optimizer.zero_grad()
                outputs = self.network_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X_tensor = torch.FloatTensor(X)
        
        self.network_.eval()
        with torch.no_grad():
            outputs = self.network_(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        return self.classes_[predicted.numpy()]

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X_tensor = torch.FloatTensor(X)
        
        self.network_.eval()
        with torch.no_grad():
            outputs = self.network_(X_tensor)
            probabilities = nn.functional.softmax(outputs, dim=1)
        return probabilities.numpy()