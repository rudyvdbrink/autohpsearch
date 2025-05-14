import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (make_scorer, 
                             balanced_accuracy_score, 
                             accuracy_score, 
                             root_mean_squared_error)

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

from autohpsearch.models.nn import AutoHPSearchClassifier, AutoHPSearchRegressor


# %% functions for hypergrid searching


def generate_hypergrid(model_name=None):
    """
    Generate a hyperparameter grid for a given model or models.
    
    Parameters:
    -----------
    model_name : None, str, or list
        The model name(s) to generate hyperparameter grids for.
        If None, grids for all available models are returned.
        If str, grid for the specified model is returned.
        If list, grids for all models in the list are returned.
    
    Returns:
    --------
    dict or list of dict
        A dictionary or list of dictionaries containing:
        - model_name: Name of the model
        - function: The function handle for the model
        - param_grid: Hyperparameter grid for the model
    """
    # Define all available models with their hyperparameter grids
    all_models = {
        'logistic_regression': {
            'model_name': 'logistic_regression',
            'function': LogisticRegression,
            'param_grid': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],                
                'class_weight': [None, 'balanced']
            }
        },       
        'random_forest': {
            'model_name': 'random_forest',
            'function': RandomForestClassifier,
            'param_grid': {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [None, 5, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False],
                'class_weight': [None, 'balanced', 'balanced_subsample']
            }
        },
        'gradient_boosting': {
            'model_name': 'gradient_boosting',
            'function': GradientBoostingClassifier,
            'param_grid': {
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [3, 5, 7, 9],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0],
                'max_features': ['sqrt', 'log2', None]
            }
        },
        'svm': {
            'model_name': 'svm',
            'function': SVC,
            'param_grid': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'degree': [2, 3, 4],  # for poly kernel
                'probability': [True],
                'class_weight': [None, 'balanced']
            }
        },
        'knn': {
            'model_name': 'knn',
            'function': KNeighborsClassifier,
            'param_grid': {
                'n_neighbors': [3, 5, 7, 9, 11, 15],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [10, 20, 30, 40, 50],
                'p': [1, 2]  # 1 for manhattan_distance, 2 for euclidean_distance
            }
        },
        'xgboost': {
            'model_name': 'xgboost',
            'function': xgb.XGBClassifier,
            'param_grid': {
                'n_estimators': [50, 100, 200, 500, 750, 1000],
                'max_depth': [3, 4, 5, 10, 15, 20],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'min_child_weight': [1, 3, 5]  # Minimum sum of instance weight for child nodes
            }
        },
        'dnn': {
            'model_name': 'dnn',
            'function': AutoHPSearchClassifier,
            'param_grid': {
                # Network architecture
                'hidden_layers': [(64, 32), (128, 64), (256, 128, 64), (64, 64, 64)],
                'activation': ['relu', 'tanh', 'elu', 'leaky_relu'],
                'dropout_rate': [0.0, 0.2, 0.3, 0.5],
                
                # Training parameters
                'optimizer': ['adam', 'sgd', 'rmsprop'],
                'learning_rate': [0.0001, 0.001, 0.01],
                'batch_size': [16, 32, 64, 128],
                'epochs': [50, 100, 150]
            }
        }        
    }
    
    # Return hypergrids based on input
    if model_name is None:
        # Return all models
        return list(all_models.values())
    
    elif isinstance(model_name, str):
        # Return single specified model
        model_name = model_name.lower()
        if model_name in all_models:
            return all_models[model_name]
        else:
            available_models = list(all_models.keys())
            raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
    
    elif isinstance(model_name, list):
        # Return multiple specified models
        result = []
        available_models = list(all_models.keys())
        for name in model_name:
            name = name.lower()
            if name in all_models:
                result.append(all_models[name])
            else:
                raise ValueError(f"Model '{name}' not found. Available models: {available_models}")
        return result
    
    else:
        raise TypeError("model_name must be None, a string, or a list of strings")


def tune_hyperparameters(X_train, y_train, X_test, y_test, hypergrid=None, scoring='balanced_accuracy', cv=5):
    """
    Perform hyperparameter tuning using grid search for multiple models and return the best model,
    optimal parameters, and results.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    hypergrid : None, dict, or list of dict, default=None
        The model configuration(s) to tune hyperparameters for.
        This should be the output from generate_hypergrid().
        If None, all available models from generate_hypergrid() are used.
    scoring : str, default='balanced_accuracy'
        Scoring metric to use for grid search
    cv : int, default=5
        Number of cross-validation folds
    
    Returns:
    --------
    dict
        A dictionary containing:
        - best_model: The best performing trained model with optimal parameters
        - optimal_params: Dictionary of optimal parameters for each model
        - results: DataFrame with best scores for each model
    """
    # If hypergrid is None, get all available models
    if hypergrid is None:
        hypergrid = generate_hypergrid()
    # If hypergrid is a single model config (dict), convert to list
    elif isinstance(hypergrid, dict):
        hypergrid = [hypergrid]
    # Otherwise, assume hypergrid is already a list of model configs
    else:
        hypergrid = hypergrid
    
    # Initialize dictionaries to store results
    best_models = {}
    optimal_params = {}
    best_scores = {}
    
    # Iterate through each model
    for config in hypergrid:
        model_name = config['model_name']
        model_func = config['function']
        param_grid = config['param_grid']
        
        print(f"Tuning {model_name}...")
        
        try:
            # Initialize the model
            model = model_func()
            
            # Set up GridSearchCV
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                verbose=1
            )
            
            # Fit GridSearchCV
            grid_search.fit(X_train, y_train)
            
            # Get the best model
            best_model = grid_search.best_estimator_
            
            # Evaluate on test set
            y_pred = best_model.predict(X_test)
            test_score = balanced_accuracy_score(y_test, y_pred)
            
            # Store results
            best_models[model_name] = best_model
            optimal_params[model_name] = grid_search.best_params_
            best_scores[model_name] = {
                'cv_score': grid_search.best_score_,
                'test_score': test_score,
                'rank': grid_search.best_index_ + 1
            }
            
            print(f"  Best CV score: {grid_search.best_score_:.4f}")
            print(f"  Test score: {test_score:.4f}")
            print(f"  Best parameters: {grid_search.best_params_}")
            
        except Exception as e:
            print(f"Error with {model_name}: {str(e)}")
            continue
    
    # Create results dataframe
    results_df = pd.DataFrame.from_dict(
        {model: scores for model, scores in best_scores.items()},
        orient='index'
    )
    
    # Sort by test score
    results_df = results_df.sort_values('test_score', ascending=False)
    
    # Find the best overall model
    if len(results_df) > 0:
        best_model_name = results_df.index[0]
        best_overall_model = best_models[best_model_name]
        print(f"\nBest overall model: {best_model_name} with test score: {results_df.loc[best_model_name, 'test_score']:.4f}")
    else:
        best_overall_model = None
        print("No models were successfully trained.")
    
    return {
        'best_model': best_overall_model,
        'optimal_params': optimal_params,
        'results': results_df
    }