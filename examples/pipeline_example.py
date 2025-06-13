# %% libraries

from autohpsearch.datasets.dataloaders import fetch_housing
from autohpsearch.pipeline.pipeline import AutoMLPipeline

# %% Load dataset

X_train, X_test, y_train, y_test = fetch_housing()

# %% Create and fit an end-to-end pipeline

# Fit the pipeline: this will clean the data run hyperparameter search, train the model, and evaluate it
pipeline = AutoMLPipeline(task_type='regression',
                          remove_outliers=True,
                          target_transform='log1p',
                          search_type='grid')

pipeline.fit(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)

# Write a report in markdown format that contains information and plots on the data, the model, and the evaluation metrics
pipeline.generate_data_report()
