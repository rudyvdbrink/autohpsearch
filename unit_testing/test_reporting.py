# %%

# %%

from sklearn.datasets import fetch_california_housing, load_breast_cancer
from autohpsearch.pipeline.pipeline import AutoMLPipeline
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Load classification dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Add some missing values
rows, cols = X.shape
mask = np.random.random((rows, cols)) < 0.05  # 5% missing values
X = X.mask(mask)

# Convert some numerical columns to categorical to demonstrate categorical encoding
X['mean radius_cat'] = pd.cut(X['mean radius'], 
                                bins=[0, 10, 15, 20, 100], 
                                labels=['tiny', 'small', 'medium', 'large'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

pipeline = AutoMLPipeline(
    task_type='classification',
    remove_outliers=True,
    outlier_method='zscore',
    outlier_threshold=3,
    num_imputation_strategy='mean',
    cat_imputation_strategy='most_frequent',
    cat_encoding_method='auto',
    scaling_method='standard',
    model_name=['random_forest_clf', 'gradient_boosting_clf', 'dnn_clf'],  # Specify models
    scoring='balanced_accuracy',
    search_type='random',
    n_iter=20,
    cv=5,
    verbose=True
)

pipeline.fit(X_train, y_train, X_test, y_test)

pipeline.generate_data_report()
