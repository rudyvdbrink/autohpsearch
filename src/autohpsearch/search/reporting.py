# %% libraries

import time

# %% functions

def measure_training_time(model, X, y):
    """
    Measure the execution time for model training.
    
    Parameters:
    -----------
    model : estimator
        The trained model to evaluate
    X : array-like
        Features to use for training
    
    Returns:
    --------
    float
        Training time in milliseconds
    """   
    # Measure time for training
    start_time = time.time()
    model.fit(X, y)
    end_time = time.time()
    
    # Calculate time in milliseconds
    training_time_ms = (end_time - start_time) * 1000
    
    return training_time_ms

def measure_prediction_time(model, X, n_repeats=100):
    """
    Measure the average execution time for model prediction.
    
    Parameters:
    -----------
    model : estimator
        The trained model to evaluate
    X : array-like
        Features to use for prediction
    n_repeats : int, default=100
        Number of times to repeat the prediction for more reliable timing
    
    Returns:
    --------
    float
        Average prediction time in milliseconds per sample
    """
    # Take a single sample for individual prediction timing
    single_sample = X[0:1]
    
    # Warm up the prediction (first prediction can be slower)
    model.predict(single_sample)
    
    # Measure time for single sample prediction
    start_time = time.time()
    for _ in range(n_repeats):
        model.predict(single_sample)
    end_time = time.time()
    
    # Calculate average time per prediction in milliseconds
    avg_time_ms = ((end_time - start_time) / n_repeats) * 1000
    
    return avg_time_ms

def count_fits(hypergrid, search_type='grid', cv=5, n_iter=10):
    """
    Calculate the total number of fits that would be performed in a hyperparameter search.
    
    Parameters:
    -----------
    hypergrid : dict or list of dict
        The model configuration(s) as produced by generate_hypergrid().
        May be a single model config (dict) or a list of model configs.
    search_type : str, default='grid'
        Type of hyperparameter search to perform.
        Options: 'grid' for grid search, 'random' for random search
    cv : int, defautl=5
        Number of cross-validation folds
    n_iter : int, default=10
        Number of parameter settings sampled when search_type='random'.
        Ignored if search_type is 'grid'.
        
    Returns:
    --------
    int
        Total number of fits that would be performed across all models
    """
    # Convert single model config to list if needed
    if isinstance(hypergrid, dict):
        hypergrid = [hypergrid]
        
    total_fits = 0
    
    # Calculate fits for each model in the hypergrid
    for config in hypergrid:
        # For grid search, count all parameter combinations
        if search_type == 'grid':
            # For grid search, each parameter combination is tested
            param_grid = config['param_grid']
            
            # Calculate number of parameter combinations
            n_combinations = 1
            for param_values in param_grid.values():
                n_combinations *= len(param_values)
                
            # Each combination is tested for each CV fold
            model_fits = n_combinations * cv
            
        # For random search, use n_iter parameter
        elif search_type == 'random' or search_type == 'bayesian':
            # For random search, n_iter parameter combinations are tested
            model_fits = n_iter * cv
            
        else:
            raise ValueError(f"Invalid search_type: {search_type}. Must be 'grid', 'random' or 'bayesian'.")
            
        # Add this model's fits to the total
        total_fits += model_fits
        
    return total_fits

    