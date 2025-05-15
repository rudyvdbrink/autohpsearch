
# AutoHPSearch

A Python package for automatic hyperparameter tuning of machine learning models for cross-sectional data.

AutoHPSearch simplifies the process of hyperparameter optimization for various machine learning models. It provides a unified interface to tune hyperparameters across multiple model types including:

- Logistic Regression
- Random Forest
- Gradient Boosting
- SVM
- K-Nearest Neighbors
- XGBoost
- Feed Forward Neural Networks (custom PyTorch implementation)

## Installation

```bash
pip install autohpsearch
```

Or install directly from the repository:

```bash
git clone https://github.com/rudyvdbrink/autohpsearch.git
cd autohpsearch
pip install -e .
```


## Usage

### Basic Example

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from autohpsearch.search.hptuing import tune_hyperparameters, generate_hypergrid

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Generate hyperparameter grid for multiple models
hypergrid = generate_hypergrid(['logistic_regression', 'random_forest', 'xgboost'])

# Tune hyperparameters
results = tune_hyperparameters(
    X_train_scaled, y_train, 
    X_test_scaled, y_test, 
    hypergrid=hypergrid, 
    scoring='balanced_accuracy',
    cv=5
)

# Access best model and results
best_model = results['best_model'] # The winning model
optimal_params = results['optimal_params'] # Best paramters for each model
performance_results = results['results'] # cross-validation and test score table

print(f"Best model: {type(best_model).__name__}")
print(f"Optimal parameters: {optimal_params}")
print(f"Results summary:\n{performance_results}")
```

### Using Neural Network Models

```python
from autohpsearch.models.nn import AutoHPSearchClassifier

# Create a neural network classifier with custom parameters
nn_clf = AutoHPSearchClassifier(
    hidden_layers=(64, 32),
    activation='relu',
    dropout_rate=0.2,
    learning_rate=0.001,
    optimizer='adam',
    batch_size=32,
    epochs=100
)

# Train the model
nn_clf.fit(X_train_scaled, y_train)

# Make predictions
y_pred = nn_clf.predict(X_test_scaled)
```

## Available Models

AutoHPSearch supports the following model types:

1. **logistic_regression**: Logistic regression classifier
2. **random_forest**: Random forest classifier
3. **gradient_boosting**: Gradient boosting classifier
4. **svm**: Support vector machine classifier
5. **knn**: K-nearest neighbors classifier
6. **xgboost**: XGBoost classifier
7. **dnn**: Deep neural network classifier/regressor

## Hyperparameter Tuning

The `generate_hypergrid()` function creates a comprehensive grid of hyperparameters for each model type. You can:

- Generate grids for all supported models: `generate_hypergrid()`
- Generate a grid for a specific model: `generate_hypergrid('random_forest')`
- Generate grids for multiple models: `generate_hypergrid(['logistic_regression', 'xgboost'])`

The `tune_hyperparameters()` function performs grid search cross-validation on the specified models and returns:

- The best overall model
- Optimal parameters for each model
- Performance metrics for each model

## Neural Network Models

AutoHPSearch includes custom neural network implementations that are compatible with scikit-learn:

- `AutoHPSearchClassifier`: For classification tasks
- `AutoHPSearchRegressor`: For regression tasks

These models provide flexibility in architecture design and training configuration while maintaining the familiar scikit-learn API.

## Author

Rudy van den Brink