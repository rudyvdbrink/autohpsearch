# %%

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

# %%
# %%

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import numpy as np

# %%

class AutoHPSearchNN(BaseEstimator):
    """
    A scikit-learn compatible wrapper for PyTorch neural networks.
    Handles pandas DataFrames/Series and numpy arrays as input.
    Supports both classification and regression tasks.
    """
    def __init__(self, hidden_layers=(64, 32), activation='relu', dropout_rate=0.2,
                 learning_rate=0.001, optimizer='adam', batch_size=32, epochs=100,
                 task_type='classification', output_activation=None, n_outputs=1):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.task_type = task_type  # 'classification' or 'regression'
        self.output_activation = output_activation  # None, 'sigmoid', 'softmax', etc.
        self.n_outputs = n_outputs  # Number of output nodes
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
    
    def _get_output_activation(self):
        # For regression, often no activation is used at the output
        if self.output_activation is None:
            return nn.Identity()
        elif self.output_activation == 'sigmoid':
            return nn.Sigmoid()
        elif self.output_activation == 'softmax':
            return nn.Softmax(dim=1)
        elif self.output_activation == 'tanh':
            return nn.Tanh()
        else:
            return nn.Identity()
    
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
        
        # Output layer
        layers.append(nn.Linear(self.hidden_layers[-1], self.n_outputs))
        
        # Apply output activation only if specified
        if self.task_type == 'classification':
            if self.n_outputs == 1:  # Binary classification
                layers.append(nn.Sigmoid())
            else:  # Multi-class classification
                layers.append(nn.Softmax(dim=1))
        else:  # Regression - typically no activation on output
            layers.append(self._get_output_activation())
        
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
    
    def _get_loss_function(self):
        if self.task_type == 'classification':
            if self.n_outputs == 1:  # Binary classification
                return nn.BCELoss()
            else:  # Multi-class classification
                return nn.CrossEntropyLoss()
        else:  # Regression
            return nn.MSELoss()  # Mean Squared Error for regression
    
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
        
        # Configure the output dimension based on task_type and y
        if self.task_type == 'classification':
            self.classes_ = np.unique(y)
            if len(self.classes_) > 2:  # Multiclass classification
                self.n_outputs = len(self.classes_)
        
        # Ensure y is properly shaped
        if self.task_type == 'classification':
            if self.n_outputs == 1:  # Binary classification
                # Ensure y is 1D for binary classification
                if len(y.shape) > 1 and y.shape[1] == 1:
                    y = y.ravel()
                y_tensor = torch.FloatTensor(y.reshape(-1, 1))
            else:  # Multiclass classification - convert to one-hot
                # Convert to integer class indices
                y = y.astype(int)
                y_tensor = torch.LongTensor(y)
        else:  # Regression
            # For regression, reshape to (-1, n_outputs)
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            y_tensor = torch.FloatTensor(y)
        
        # Convert features to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Store input dimension
        self.input_dim = X.shape[1]
        
        # Build the model
        self.model = self._build_model()
        
        # Define loss function and optimizer
        criterion = self._get_loss_function()
        optimizer = self._get_optimizer(self.model.parameters())
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # For multiclass with CrossEntropyLoss, targets need to be class indices
                if self.task_type == 'classification' and self.n_outputs > 1:
                    loss = criterion(outputs, targets)
                else:
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
        """
        Return probability estimates for classification tasks.
        Only valid for classification tasks.
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
            
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
        
        # Format based on number of classes
        if self.n_outputs == 1:  # Binary classification
            # Format for scikit-learn (needs probabilities for both classes)
            proba = np.hstack([1 - y_prob, y_prob])
        else:  # Multi-class classification
            proba = y_prob
            
        return proba
    
    def predict(self, X):
        """
        Return predictions for classification or regression tasks.
        """
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
            predictions = self.model(X_tensor).cpu().numpy()
        
        if self.task_type == 'classification':
            if self.n_outputs == 1:  # Binary classification
                return (predictions >= 0.5).astype(int).ravel()
            else:  # Multi-class classification
                return np.argmax(predictions, axis=1)
        else:  # Regression
            return predictions


# Create specific variants that inherit from scikit-learn mixin classes
class AutoHPSearchClassifier(AutoHPSearchNN, ClassifierMixin):
    """Neural network classifier compatible with scikit-learn."""
    
    def __init__(self, hidden_layers=(64, 32), activation='relu', dropout_rate=0.2,
                 learning_rate=0.001, optimizer='adam', batch_size=32, epochs=100,
                 output_activation=None, n_outputs=1):
        super().__init__(
            hidden_layers=hidden_layers, 
            activation=activation,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            optimizer=optimizer,
            batch_size=batch_size,
            epochs=epochs,
            task_type='classification',
            output_activation=output_activation,
            n_outputs=n_outputs
        )

class AutoHPSearchRegressor(AutoHPSearchNN, RegressorMixin):
    """Neural network regressor compatible with scikit-learn."""
    
    def __init__(self, hidden_layers=(64, 32), activation='relu', dropout_rate=0.2,
                 learning_rate=0.001, optimizer='adam', batch_size=32, epochs=100):
        super().__init__(
            hidden_layers=hidden_layers, 
            activation=activation,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            optimizer=optimizer,
            batch_size=batch_size,
            epochs=epochs,
            task_type='regression',
            output_activation=None,
            n_outputs=1
        )

# %%