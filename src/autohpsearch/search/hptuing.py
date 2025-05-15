import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (make_scorer, 
                             balanced_accuracy_score, 
                             accuracy_score,
                             mean_squared_error,
                             mean_absolute_error,
                             r2_score)

import numpy as np
import pandas as pd
from tqdm import tqdm

from autohpsearch.search.grids import get_grid

# %% functions for hypergrid searching

def generate_hypergrid(model_name=None, task_type='classification'):
    """
    Generate a hyperparameter grid for a given model or models.
    
    Parameters:
    -----------
    model_name : None, str, or list
        The model name(s) to generate hyperparameter grids for.
        If None, grids for all available models of the specified task_type are returned.
        If str, grid for the specified model is returned.
        If list, grids for all models in the list are returned.
    task_type : str, default='classification'
        The type of machine learning task.
        Options: 'classification' or 'regression'
    
    Returns:
    --------
    dict or list of dict
        A dictionary or list of dictionaries containing:
        - model_name: Name of the model
        - function: The function handle for the model
        - param_grid: Hyperparameter grid for the model
        - task_type: The task type ('classification' or 'regression')
    """
    
    # Retrun models based on task_type parameter
    if task_type == 'classification':
        all_models = get_grid(task_type='classification')
    elif task_type == 'regression':
        all_models = get_grid(task_type='regression')
    else:
        raise ValueError(f"Invalid task_type: {task_type}. Must be 'classification' or 'regression'.")
    
    # Return hypergrids based on input
    if model_name is None:
        # Return all models for the specified task_type
        return list(all_models.values())
    
    elif isinstance(model_name, str):
        # Return single specified model
        model_name = model_name.lower()
        if model_name in all_models:
            return all_models[model_name]
        else:
            available_models = list(all_models.keys())
            raise ValueError(f"Model '{model_name}' not found. Available models for {task_type}: {available_models}")
    
    elif isinstance(model_name, list):
        # Return multiple specified models
        result = []
        available_models = list(all_models.keys())
        for name in model_name:
            name = name.lower()
            if name in all_models:
                result.append(all_models[name])
            else:
                raise ValueError(f"Model '{name}' not found. Available models for {task_type}: {available_models}")
        return result
    
    else:
        raise TypeError("model_name must be None, a string, or a list of strings")


def tune_hyperparameters(X_train, y_train, X_test, y_test, hypergrid=None, scoring=None, cv=5, task_type='classification'):
    """
    Perform hyperparameter tuning using grid search for multiple models and return the best model,
    optimal parameters, and results.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels/targets
    X_test : array-like
        Test features
    y_test : array-like
        Test labels/targets
    hypergrid : None, dict, or list of dict, default=None
        The model configuration(s) to tune hyperparameters for.
        This should be the output from generate_hypergrid().
        If None, all available models from generate_hypergrid() for the specified task_type are used.
    scoring : str, default=None
        Scoring metric to use for grid search
        If None, defaults to 'balanced_accuracy' for classification and 'neg_root_mean_squared_error' for regression
    cv : int, default=5
        Number of cross-validation folds
    task_type : str, default='classification'
        The type of machine learning task.
        Options: 'classification' or 'regression'
    
    Returns:
    --------
    dict
        A dictionary containing:
        - best_model: The best performing trained model with optimal parameters
        - optimal_params: Dictionary of optimal parameters for each model
        - results: DataFrame with best scores for each model
    """
    # Set default scoring metric based on task type
    if scoring is None:
        if task_type == 'classification':
            scoring = 'balanced_accuracy'
        else:  # regression
            scoring = 'neg_root_mean_squared_error'
    
    # If hypergrid is None, get all available models for the specified task type
    if hypergrid is None:
        hypergrid = generate_hypergrid(task_type=task_type)
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
        model_task_type = config.get('task_type', task_type)  # Get task_type from config or use default
        
        if model_task_type != task_type:
            print(f"Skipping {model_name}: Model task type ({model_task_type}) doesn't match requested task type ({task_type})")
            continue
        
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
            
            # Evaluate on test set based on task type
            y_pred = best_model.predict(X_test)
            
            if task_type == 'classification':
                test_score = balanced_accuracy_score(y_test, y_pred)
            else:  # regression
                # Use negative RMSE to be consistent with scoring metric
                test_score = -np.sqrt(mean_squared_error(y_test, y_pred))
            
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
        
        # Display appropriate metrics based on task type
        if task_type == 'classification':
            metric_display = "balanced accuracy"
        else:  # regression
            metric_display = "negative RMSE"
            
        print(f"\nBest overall model: {best_model_name} with test {metric_display}: {results_df.loc[best_model_name, 'test_score']:.4f}")
        
        # Add additional evaluation metrics for regression
        if task_type == 'regression':
            y_pred = best_overall_model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"Additional metrics for best model:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R²: {r2:.4f}")
    else:
        best_overall_model = None
        print("No models were successfully trained.")
    
    return {
        'best_model': best_overall_model,
        'optimal_params': optimal_params,
        'results': results_df
    }