# %%

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

# %%

class AutoHPSearchNN(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn compatible wrapper for PyTorch neural networks.
    Handles pandas DataFrames/Series and numpy arrays as input.
    """
    def __init__(self, hidden_layers=(64, 32), activation='relu', dropout_rate=0.2,
                 learning_rate=0.001, optimizer='adam', batch_size=32, epochs=100):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.input_dim = None
        self.classes_ = None
        
    def _get_activation(self):
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'sigmoid':
            return nn.Sigmoid()
        elif self.activation == 'elu':
            return nn.ELU()
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            return nn.ReLU()  # Default
    
    def _build_model(self):
        layers = []
        activation = self._get_activation()
        
        # Input layer
        layers.append(nn.Linear(self.input_dim, self.hidden_layers[0]))
        layers.append(activation)
        if self.dropout_rate > 0:
            layers.append(nn.Dropout(self.dropout_rate))
        
        # Hidden layers
        for i in range(len(self.hidden_layers) - 1):
            layers.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]))
            layers.append(activation)
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
        
        # Output layer (sigmoid for binary classification)
        layers.append(nn.Linear(self.hidden_layers[-1], 1))
        layers.append(nn.Sigmoid())
        
        model = nn.Sequential(*layers)
        model.to(self.device)
        return model
    
    def _get_optimizer(self, model_parameters):
        if self.optimizer == 'adam':
            return optim.Adam(model_parameters, lr=self.learning_rate)
        elif self.optimizer == 'sgd':
            return optim.SGD(model_parameters, lr=self.learning_rate)
        elif self.optimizer == 'rmsprop':
            return optim.RMSprop(model_parameters, lr=self.learning_rate)
        else:
            return optim.Adam(model_parameters, lr=self.learning_rate)  # Default
    
    def _convert_to_numpy(self, X):
        """Convert pandas DataFrame or Series to numpy array of float64 type"""
        import pandas as pd
        
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            return X.to_numpy().astype('float64')
        return X
    
    def fit(self, X, y):
        # Convert pandas objects to numpy arrays if necessary
        X = self._convert_to_numpy(X)
        y = self._convert_to_numpy(y)
        
        # Ensure y is 1D
        if len(y.shape) > 1 and y.shape[1] == 1:
            y = y.ravel()
        
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1))
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Store input dimension and classes
        self.input_dim = X.shape[1]
        self.classes_ = np.unique(y)
        
        # Build the model
        self.model = self._build_model()
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = self._get_optimizer(self.model.parameters())
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Optional: print progress every 10 epochs
            if (epoch + 1) % 10 == 0 and hasattr(self, 'verbose') and self.verbose:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(dataloader):.4f}')
                
        return self
    
    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        
        # Convert pandas objects to numpy arrays if necessary
        X = self._convert_to_numpy(X)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Get predictions
        with torch.no_grad():
            y_prob = self.model(X_tensor).cpu().numpy()
        
        # Format for scikit-learn (needs probabilities for both classes in binary classification)
        proba = np.hstack([1 - y_prob, y_prob])
        return proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    

# %%
