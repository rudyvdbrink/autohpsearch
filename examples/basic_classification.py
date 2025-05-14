
# %% import libraries

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from autohpsearch.search.hptuing import tune_hyperparameters, generate_hypergrid

# %% generate data

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% run hyperparameter tuning

# Generate hyperparameter grid for two models
hypergrid = generate_hypergrid(['logistic_regression', 'random_forest_clf'])

# Tune hyperparameters
results = tune_hyperparameters(
    X_train_scaled, y_train, 
    X_test_scaled, y_test, 
    hypergrid=hypergrid, 
    scoring='balanced_accuracy',
    cv=5
)

# Access best model and results
best_model = results['best_model']
optimal_params = results['optimal_params']
performance_results = results['results']

print(f"Best model: {type(best_model).__name__}")
print(f"Optimal parameters: {optimal_params}")
print(f"Results summary:\n{performance_results}")