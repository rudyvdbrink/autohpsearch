# %% import libraries

import pandas as pd

from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (make_scorer, 
                             balanced_accuracy_score, 
                             accuracy_score,
                             mean_squared_error,
                             mean_absolute_error,
                             r2_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_auc_score)

import numpy as np
import pandas as pd

from autohpsearch.search.grids import get_grid
from autohpsearch.utils.context import hush
from autohpsearch.search.reporting import (measure_training_time,
                                           measure_prediction_time, 
                                           count_fits)

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

def tune_hyperparameters(X_train, y_train, X_test, y_test=None, hypergrid=None, scoring=None, cv=5, 
                        task_type='classification', search_type='grid', n_iter=10, verbose=False):
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
    y_test : array-like, optional (default=None)
        Test labels/targets
        If set to None, it will not evaluate on test data.
    hypergrid : None, dict, or list of dict, default=None
        The model configuration(s) to tune hyperparameters for.
        This should be the output from generate_hypergrid().
        If None, all available models from generate_hypergrid() for the specified task_type are used.
    scoring : str or callable, default=None
        Scoring metric to use for grid search. Options depend on task_type:
        
        For classification (task_type='classification'):
        - 'balanced_accuracy' (default): Balanced accuracy score
        - 'accuracy': Standard accuracy score
        - 'precision': Precision score (binary or weighted for multiclass)
        - 'recall': Recall score (binary or weighted for multiclass)
        - 'f1': F1 score (binary or weighted average for multiclass)
        - 'roc_auc': ROC AUC score (only for binary classification)
        
        For regression (task_type='regression'):
        - 'neg_root_mean_squared_error' (default): Negative root mean squared error
        - 'neg_mean_absolute_error': Negative mean absolute error
        - 'r2': R-squared score
        - 'neg_mean_squared_error': Negative mean squared error
        
        Also accepts any valid scikit-learn scoring string or a custom scoring function
        with signature scorer(estimator, X, y).
    cv : int, default=5
        Number of cross-validation folds
    task_type : str, default='classification'
        The type of machine learning task.
        Options: 'classification' or 'regression'
    search_type : str, default='grid'
        Type of hyperparameter search to perform.
        Options: 'grid' for grid search, 'random' for random search, 'bayesian' for Bayesian optimization.
    n_iter : int, default=10
        Number of parameter settings sampled when search_type='random' or search_type='bayesian'.
    verbose : bool, default=False
        Whether to display verbose output during fitting and evaluation.
        If True, extended output from scikit-learn will be shown.
    
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

     # Define available scoring metrics for each task type
    classification_metrics = {
        'balanced_accuracy': 'balanced_accuracy',
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1': make_scorer(f1_score, average='weighted'),
        'roc_auc': 'roc_auc'  # Note: only works for binary classification
    }
    
    regression_metrics = {
        'neg_root_mean_squared_error': 'neg_root_mean_squared_error',
        'neg_mean_absolute_error': 'neg_mean_absolute_error',
        'r2': 'r2',
        'neg_mean_squared_error': 'neg_mean_squared_error'
    }

    # If scoring is a string that matches our predefined metrics, use the corresponding scorer
    if isinstance(scoring, str):
        if task_type == 'classification' and scoring in classification_metrics:
            scoring = classification_metrics[scoring]
        elif task_type == 'regression' and scoring in regression_metrics:
            scoring = regression_metrics[scoring]
        # Otherwise, assume it's a valid scikit-learn scoring string
    
    # If hypergrid is None, get all available models for the specified task type
    if hypergrid is None:
        hypergrid = generate_hypergrid(task_type=task_type)
    # If hypergrid is a single model config (dict), convert to list
    elif isinstance(hypergrid, dict):
        hypergrid = [hypergrid]
    # Otherwise, assume hypergrid is already a list of model configs
    else:
        hypergrid = hypergrid

    # Report the number of fits
    total_fits = count_fits(hypergrid, search_type=search_type, cv=cv, n_iter=n_iter)
    print(f"Total number of fits to be performed: {total_fits}")
    
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
            
           # Set up search strategy based on search_type
            if search_type == 'grid':
                search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring=scoring,
                    cv=cv,
                    n_jobs=-1,
                    verbose=1
                )
            elif search_type == 'random':
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    scoring=scoring,
                    cv=cv,
                    n_jobs=-1,
                    verbose=1
                )
            elif search_type == 'bayesian':
                search = BayesSearchCV(
                estimator=model,
                search_spaces=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
                )
            else:
                raise ValueError(f"Invalid search_type: {search_type}. Must be 'grid', 'random', or 'bayesian'.")
                
            # Fit the search
            if verbose:
                search.fit(X_train, y_train)
            else:
                with hush():
                    search.fit(X_train, y_train)
            
            # Get the best model
            best_model = search.best_estimator_                        
            
            # Evaluate on test set based on task type
            y_pred = best_model.predict(X_test)

            # Measure training time
            train_time_ms = measure_training_time(best_model, X_train, y_train)

            # Measure prediction time
            pred_time_ms = measure_prediction_time(best_model, X_test)
            
            # Get evaluation metric
            if y_test is not None:
                if task_type == 'classification':
                    # Default to balanced accuracy for test evaluation
                    if scoring == 'accuracy' or scoring == classification_metrics['accuracy']:
                        test_score = accuracy_score(y_test, y_pred)
                        metric_display = "accuracy"
                    elif scoring == 'precision' or (isinstance(scoring, object) and 'precision' in str(scoring)):
                        test_score = precision_score(y_test, y_pred, average='weighted')
                        metric_display = "precision"
                    elif scoring == 'recall' or (isinstance(scoring, object) and 'recall' in str(scoring)):
                        test_score = recall_score(y_test, y_pred, average='weighted')
                        metric_display = "recall"
                    elif scoring == 'f1' or (isinstance(scoring, object) and 'f1' in str(scoring)):
                        test_score = f1_score(y_test, y_pred, average='weighted')
                        metric_display = "f1"
                    elif scoring == 'roc_auc' or scoring == classification_metrics['roc_auc']:
                        # Only compute if binary classification
                        if len(np.unique(y_test)) == 2:
                            try:
                                y_pred_proba = best_model.predict_proba(X_test)[:,1]
                                test_score = roc_auc_score(y_test, y_pred_proba)
                                metric_display = "roc_auc"
                            except:
                                # Fall back to balanced accuracy if roc_auc fails
                                test_score = balanced_accuracy_score(y_test, y_pred)
                                metric_display = "balanced_accuracy (fallback)"
                        else:
                            # Fall back to balanced accuracy for multiclass
                            test_score = balanced_accuracy_score(y_test, y_pred)
                            metric_display = "balanced_accuracy (fallback)"
                    else:
                        # Default fallback to balanced_accuracy
                        test_score = balanced_accuracy_score(y_test, y_pred)
                        metric_display = "balanced_accuracy"
                else:  # regression
                    if scoring == 'r2' or scoring == regression_metrics['r2']:
                        test_score = r2_score(y_test, y_pred)
                        metric_display = "R²"
                    elif scoring == 'neg_mean_absolute_error' or scoring == regression_metrics['neg_mean_absolute_error']:
                        test_score = -mean_absolute_error(y_test, y_pred)
                        metric_display = "negative MAE"
                    elif scoring == 'neg_mean_squared_error' or scoring == regression_metrics['neg_mean_squared_error']:
                        test_score = -mean_squared_error(y_test, y_pred)
                        metric_display = "negative MSE"
                    else:  # Default to negative RMSE
                        # Use negative RMSE to be consistent with scoring metric
                        test_score = -np.sqrt(mean_squared_error(y_test, y_pred))
                        metric_display = "negative RMSE"
            else:
                # If no test set is provided, use the best cross-validation score
                test_score = np.nan
            
            # Store results
            best_models[model_name] = best_model
            optimal_params[model_name] = search.best_params_
            best_scores[model_name] = {
                'cv_score': search.best_score_,
                'test_score': test_score,
                'train_time_ms': train_time_ms,
                'prediction_time_ms': pred_time_ms,
            }
            
            print(f"  Best CV score: {search.best_score_:.4f}")
            print(f"  Test score: {test_score:.4f}")
            print(f"  Best parameters: {search.best_params_}")
            
        except Exception as e:
            print(f"Error with {model_name}: {str(e)}")
            continue
    
    # Create results dataframe
    results_df = pd.DataFrame.from_dict(
        {model: scores for model, scores in best_scores.items()},
        orient='index'
    )
    
    # Sort by test score (and cv_score in case of ties)
    #results_df = results_df.sort_values(['test_score', 'cv_score', 'prediction_time_ms'], ascending=[False, False, True])
    results_df = results_df.sort_values(['cv_score', 'train_time_ms', 'prediction_time_ms'], ascending=[False, True, True])

    
    # Find the best overall model
    if len(results_df) > 0:
        best_model_name = results_df.index[0]
        best_overall_model = best_models[best_model_name]
        
        # Display appropriate metrics based on task type      
        print(f"\nBest overall model: {best_model_name} with test score: {results_df.loc[best_model_name, 'test_score']:.4f}")
        
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