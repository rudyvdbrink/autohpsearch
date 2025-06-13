# %% libraries

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import AutoHPSearch functions
from autohpsearch import tune_hyperparameters, generate_hypergrid
from autohpsearch.vis.evaluation_plots import (    
    regression_prediction_plot,
    regression_residual_plot,
    plot_confusion_matrix,
    plot_ROC_curve,
    bar_plot_results_df
)

# %% Load and prepare data

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Split the data for both classification and regression examples
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% Classification example

# Generate hyperparameter grid for classification models
clf_hypergrid = generate_hypergrid(task_type='classification')

# Tune hyperparameters for classification with balanced_accuracy
clf_results = tune_hyperparameters(
    X_train_scaled, y_train,
    X_test_scaled, y_test,
    hypergrid=clf_hypergrid,
    scoring='balanced_accuracy',
    cv=5,
    search_type='random',
    n_iter=30,
    verbose=False
)

# Print the results:
# It is possible (likely) that here some models are tied in performance on the test set. 
# Models are selected based on performance in cross-validation (computed on the trainning data).
# In this case of ties,autohpsearch will additionally sort models by their training time. 
# A third criterion for sorting is the computation time for model predcition.
print(clf_results['results'])

# Get the best model
best_clf = clf_results['best_model']
best_model_key = clf_results['results'].index[0] if not clf_results['results'].empty else None

# Make predictions with the best model
y_clf_pred = best_clf.predict(X_test_scaled)
y_clf_proba = best_clf.predict_proba(X_test_scaled)

# Plot confusion matrix
fig = plot_confusion_matrix(y_test, y_clf_pred, labels=range(len(target_names)))

# Plot ROC curve
fig = plot_ROC_curve(y_test, y_clf_proba, labels=target_names)

# Plot some timing information
fig = bar_plot_results_df(clf_results['results'], 'prediction_time_ms')


# %% Regression example

# For regression, we'll predict the petal length from the other features
# Use feature at index 2 (petal length) as target and others as features
X_reg = np.delete(X, 2, axis=1)
y_reg = X[:, 2]
reg_feature_names = [name for i, name in enumerate(feature_names) if i != 2]
reg_target_name = feature_names[2]

# Split the data for regression
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Standardize features
reg_scaler = StandardScaler()
X_reg_train_scaled = reg_scaler.fit_transform(X_reg_train)
X_reg_test_scaled = reg_scaler.transform(X_reg_test)

# Generate hyperparameter grid for regression models
regression_models = ['random_forest_reg', 'gradient_boosting_reg', 'linear_regression']
reg_hypergrid = generate_hypergrid(regression_models, task_type='regression')

# Tune hyperparameters for regression using r2 scoring
reg_results = tune_hyperparameters(
    X_reg_train_scaled, y_reg_train,
    X_reg_test_scaled, y_reg_test,
    hypergrid=reg_hypergrid,
    scoring='r2',
    cv=5,
    task_type='regression',
    search_type='random',
    n_iter=10,
    verbose=False
)

# Get the best regression model
best_reg = reg_results['best_model']

# Make predictions with the best model
y_reg_pred = best_reg.predict(X_reg_test_scaled)

# Plot regression prediction plot
fig = regression_prediction_plot(y_reg_test, y_reg_pred)

# Plot regression residuals
fig = regression_residual_plot(y_reg_test, y_reg_pred)

# %%
