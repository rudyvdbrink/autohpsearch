
# AutoHPSearch

A Python package for automatic hyperparameter tuning of machine learning models for cross-sectional data. AutoHPSearch simplifies the process of hyperparameter optimization for various machine learning models by providing a unified interface to tune hyperparameters across multiple model types.

The search space is navigated with either grid or random search. The latter is faster but provides a less comprehensive coverage of the search space. CUDA-enabled computing for neural network implementations is included.

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

To enable CUDA you need to manually install the right version of torch+cuda depending on your GPU and system.

## Usage

### Examples Scripts
- [Classification](https://github.com/rudyvdbrink/autohpsearch/blob/main/examples/classification_basic.ipynb) - Demonstrates simple binary classification
- [Regression](https://github.com/rudyvdbrink/autohpsearch/blob/main/examples/regression_basic.ipynb) - Simple regression example
- [Neural Network Usage](https://github.com/rudyvdbrink/autohpsearch/blob/main/examples/nn_usage.py) - Syntax examples for using scikit-learn compatible neural networks
- [Iris example](https://github.com/rudyvdbrink/autohpsearch/blob/main/examples/iris_example.py) - Examples of both classification and regression solving using real data

### Basic Example

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from autohpsearch.search.hptuing import tune_hyperparameters, generate_hypergrid

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Generate hyperparameter grid for multiple models
hypergrid = generate_hypergrid(['logistic_regression', 'random_forest_clf', 'xgboost_clf'])

# Tune hyperparameters
results = tune_hyperparameters(
    X_train, y_train, 
    X_test, y_test, 
    hypergrid=hypergrid, 
    scoring='balanced_accuracy',
    search_type='random',
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

### Classification Models
1. **logistic_regression**: Logistic regression classifier (including L1 / L2 / elastic net regularization)
2. **random_forest_clf**: Random forest classifier
3. **gradient_boosting_clf**: Gradient boosting classifier
4. **svm_clf**: Support vector machine classifier
5. **knn_clf**: K-nearest neighbors classifier
6. **xgboost_clf**: XGBoost classifier
7. **dnn_clf**: Deep neural network classifier

### Regression Models
1. **linear_regression**: Linear regression
2. **ridge**: Ridge regression
3. **lasso**: Lasso regression
4. **elastic_net**: Elastic Net regression
5. **random_forest_reg**: Random forest regressor
6. **gradient_boosting_reg**: Gradient boosting regressor
7. **svr**: Support vector regression
8. **knn_reg**: K-nearest neighbors regressor
9. **xgboost_reg**: XGBoost regressor
10. **dnn_reg**: Deep neural network regressor

## Hyperparameter Tuning

The `generate_hypergrid()` function creates a comprehensive grid of hyperparameters for each model type. You can:

- Generate grids for all supported models: `generate_hypergrid(task_type='classification')`
- Generate a grid for a specific model: `generate_hypergrid('random_forest_clf')` or `generate_hypergrid('random_forest_reg', task_type='regression')`
- Generate grids for multiple models: `generate_hypergrid(['logistic_regression', 'xgboost_clf'])`

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