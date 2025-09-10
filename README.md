
# AutoHPSearch

A Python package for automatic hyperparameter tuning of machine learning models for cross-sectional data. AutoHPSearch simplifies the process of hyperparameter optimization for various machine learning models by providing a unified interface to tune hyperparameters across multiple model types.

AutoHPSearch also contains functionality for full end-to-end pipelines that include cleaning, parameter search, model evaluation, automated production of data reports in markdown format ([example here](https://github.com/rudyvdbrink/autohpsearch/blob/main/example_reports/data_report_v0001_20250612_200805.md)), as well as fine tuning large language models (LLMs) with just a few lines of code.  

The hyperparameter search space is navigated with grid, random, or bayesian search. Random search is faster but provides a less comprehensive coverage of the search space. CUDA-enabled computing for neural network implementations is included.

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
- [Classification](https://github.com/rudyvdbrink/autohpsearch/blob/main/examples/basic_classification.py) - Demonstrates simple binary classification
- [Regression](https://github.com/rudyvdbrink/autohpsearch/blob/main/examples/basic_regression.py) - Simple regression example
- [Neural Network Usage](https://github.com/rudyvdbrink/autohpsearch/blob/main/examples/nn_usage.py) - Syntax examples for using scikit-learn compatible neural networks
- [Iris Example](https://github.com/rudyvdbrink/autohpsearch/blob/main/examples/iris_example.py) - Examples of both classification and regression solving using real data
- [Pipeline Example](https://github.com/rudyvdbrink/autohpsearch/blob/main/examples/pipeline_example.py) - An example of a full automated end-to-end pipeline
- [LLM Example](https://github.com/rudyvdbrink/autohpsearch/blob/main/examples/llm_classification.py) - An example of how to fine tune an LLM for a sequence classification task

### Creating and Fitting a Full End-To-End Automatic Pipeline

```python
# Import requirements
from autohpsearch.datasets.dataloaders import fetch_housing
from autohpsearch import AutoMLPipeline

# Load an example dataset
X_train, X_test, y_train, y_test = fetch_housing()

# Fit the pipeline: this will clean the data run hyperparameter search, train models, and evaluate them
pipeline = AutoMLPipeline(task_type='regression')
pipeline.fit(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)
```
### Automated Reports on Data Distributions And Model Performance

AutoHPSearch can generate a report on the data that includes plots of feature distributions before and after data cleaning, and statistics on requested properties of the data such as the number of outliers etc. It will also include plots for the best performing model to examine its performance on the test set. You can find an example report [here](https://github.com/rudyvdbrink/autohpsearch/blob/main/example_reports/data_report_v0001_20250612_200805.md). To create a report, simply run:

```python
# Write a report in markdown format 
pipeline.generate_data_report()
```
### Example Classification With Specified Models

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from autohpsearch import tune_hyperparameters, generate_hypergrid

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

### Fitting Neural Network Models

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

### Fine Tuning Large Language Models

AutoHPSearch includes functionality for low-rank adaptation of large language models. The fitting process is integrated with the transformers library, so pre-trained base models are downloaded from huggingface. Model classes also contain methods for pushing trained models to hugginface hub. 

```python
from autohpsearch import AutoLoraForSeqClass
from autohpsearch.datasets.dataloaders import fetch_imdb

# Get the data (a selection of imdb reviews, which can be positive or negative)
dataset = fetch_imdb()

# Initialize the model with a base model, and LoRA parameters
model = AutoLoraForSeqClass(base_model='bert-base-uncased',
                            r=2,
                            train_batch_size=8,
                            eval_batch_size=8,
                            num_train_epochs=3,
                            )

# Fit the model on the dataset
model.fit(dataset)

# Push the model to hugging face hub
model.push()
```

## Available Models

AutoHPSearch supports the following model types for end-to-end training:

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

### Large Language Models

- The base models for sequence tasks are drawn from [HuggingFace.co](https://www.huggingface.co), so any model that is hosted there is supported in principle. These include popular pre-trained models such as Meta's Llama models, Mistral, GPT-Neo, and others. 

## Hyperparameter Tuning

The `generate_hypergrid()` function creates a comprehensive grid of hyperparameters for each model type. You can:

- Generate grids for all supported models: `generate_hypergrid(task_type='classification')`
- Generate a grid for a specific model: `generate_hypergrid('random_forest_clf')` or `generate_hypergrid('random_forest_reg', task_type='regression')`
- Generate grids for multiple models: `generate_hypergrid(['logistic_regression', 'xgboost_clf'])`

The `tune_hyperparameters()` function performs grid search cross-validation on the specified models and returns:

- The best overall model
- Optimal parameters for each model
- Performance metrics for each model

## Neural Network Models / LLMs

AutoHPSearch includes custom neural network implementations that are compatible with scikit-learn:

- `AutoHPSearchClassifier`: For classification tasks
- `AutoHPSearchRegressor`: For regression tasks

These models provide flexibility in architecture design and training configuration while maintaining the familiar scikit-learn API.

Large language model classes:
- `AutoLoraForSeqClass`: For sequence classificiation tasks
- `AutoLoraForSeqReg`: For sequence regression tasks
- `AutoLoraForSeqDual`: for dual task models that fit both a regression and classification head simultaneously

## Author

[Rudy van den Brink](https://www.brinkdatascience.com)