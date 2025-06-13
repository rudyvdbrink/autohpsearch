# %% import libraries

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
 
from autohpsearch import tune_hyperparameters, generate_hypergrid
from autohpsearch.vis.evaluation_plots import regression_prediction_plot, regression_residual_plot

# %% generate data

# Generate synthetic regression data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% run hyperparameter tuning

# Generate hyperparameter grid for regression models
hypergrid = generate_hypergrid(['ridge', 'random_forest_reg'], task_type='regression')

# Tune hyperparameters
results = tune_hyperparameters(
    X_train, y_train, 
    X_test, y_test, 
    hypergrid=hypergrid, 
    scoring='neg_root_mean_squared_error',
    cv=5,
    task_type='regression',
)

# Access best model and results
best_model = results['best_model']
optimal_params = results['optimal_params']
performance_results = results['results']

print(f"Best model: {type(best_model).__name__}")
print(f"Optimal parameters: {optimal_params}")
print(f"Results summary:\n{performance_results}")

# %% evaluate model performance

regression_prediction_plot(y_test, best_model.predict(X_test))
regression_residual_plot(y_test, best_model.predict(X_test))