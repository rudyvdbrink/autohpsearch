# %% libraries

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

import xgboost as xgb

from autohpsearch.models.nn import AutoHPSearchClassifier, AutoHPSearchRegressor


# %% function that defines the model dictionaries

def get_grid(task_type='classification'):

    classification_models = {
        'logistic_regression': {
            'model_name': 'logistic_regression',
            'function': LogisticRegression,
            'param_grid': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],                
                'class_weight': [None, 'balanced']
            },
            'task_type': 'classification'
        },       
        'random_forest_clf': {
            'model_name': 'random_forest_clf',
            'function': RandomForestClassifier,
            'param_grid': {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [None, 5, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False],
                'class_weight': [None, 'balanced', 'balanced_subsample']
            },
            'task_type': 'classification'
        },
        'gradient_boosting_clf': {
            'model_name': 'gradient_boosting_clf',
            'function': GradientBoostingClassifier,
            'param_grid': {
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [3, 5, 7, 9],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0],
                'max_features': ['sqrt', 'log2', None]
            },
            'task_type': 'classification'
        },
        'svm_clf': {
            'model_name': 'svm_clf',
            'function': SVC,
            'param_grid': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'degree': [2, 3, 4],  # for poly kernel
                'probability': [True],
                'class_weight': [None, 'balanced']
            },
            'task_type': 'classification'
        },
        'knn_clf': {
            'model_name': 'knn_clf',
            'function': KNeighborsClassifier,
            'param_grid': {
                'n_neighbors': [3, 5, 7, 9, 11, 15],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [10, 20, 30, 40, 50],
                'p': [1, 2]  # 1 for manhattan_distance, 2 for euclidean_distance
            },
            'task_type': 'classification'
        },
        'xgboost_clf': {
            'model_name': 'xgboost_clf',
            'function': xgb.XGBClassifier,
            'param_grid': {
                'n_estimators': [50, 100, 200, 500, 750, 1000],
                'max_depth': [3, 4, 5, 10, 15, 20],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'min_child_weight': [1, 3, 5]  # Minimum sum of instance weight for child nodes
            },
            'task_type': 'classification'
        },
        'dnn_clf': {
            'model_name': 'dnn_clf',
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
            },
            'task_type': 'classification'
        }        
    }
    
    # Define regression models with their hyperparameter grids
    regression_models = {
        'linear_regression': {
            'model_name': 'linear_regression',
            'function': LinearRegression,
            'param_grid': {
                'fit_intercept': [True, False],
                'normalize': [True, False],
            },
            'task_type': 'regression'
        },
        'ridge': {
            'model_name': 'ridge',
            'function': Ridge,
            'param_grid': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'fit_intercept': [True, False],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            },
            'task_type': 'regression'
        },
        'lasso': {
            'model_name': 'lasso',
            'function': Lasso,
            'param_grid': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'fit_intercept': [True, False],
                'max_iter': [1000, 2000, 5000]
            },
            'task_type': 'regression'
        },
        'elastic_net': {
            'model_name': 'elastic_net',
            'function': ElasticNet,
            'param_grid': {
                'alpha': [0.001, 0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'fit_intercept': [True, False],
                'max_iter': [1000, 2000, 5000]
            },
            'task_type': 'regression'
        },
        'random_forest_reg': {
            'model_name': 'random_forest_reg',
            'function': RandomForestRegressor,
            'param_grid': {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [None, 5, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2', None],
                'bootstrap': [True, False]
            },
            'task_type': 'regression'
        },
        'gradient_boosting_reg': {
            'model_name': 'gradient_boosting_reg',
            'function': GradientBoostingRegressor,
            'param_grid': {
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [3, 5, 7, 9],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0],
                'max_features': ['auto', 'sqrt', 'log2', None]
            },
            'task_type': 'regression'
        },
        'svr': {
            'model_name': 'svr',
            'function': SVR,
            'param_grid': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'epsilon': [0.01, 0.1, 0.2, 0.5],
                'degree': [2, 3, 4]  # for poly kernel
            },
            'task_type': 'regression'
        },
        'knn_reg': {
            'model_name': 'knn_reg',
            'function': KNeighborsRegressor,
            'param_grid': {
                'n_neighbors': [3, 5, 7, 9, 11, 15],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [10, 20, 30, 40, 50],
                'p': [1, 2]  # 1 for manhattan_distance, 2 for euclidean_distance
            },
            'task_type': 'regression'
        },
        'xgboost_reg': {
            'model_name': 'xgboost_reg',
            'function': xgb.XGBRegressor,
            'param_grid': {
                'n_estimators': [50, 100, 200, 500, 750, 1000],
                'max_depth': [3, 4, 5, 10, 15, 20],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'min_child_weight': [1, 3, 5]  # Minimum sum of instance weight for child nodes
            },
            'task_type': 'regression'
        },
        'dnn_reg': {
            'model_name': 'dnn_reg',
            'function': AutoHPSearchRegressor,
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
            },
            'task_type': 'regression'
        }
    }

    all_models = {**classification_models, **regression_models}

    if task_type == 'classification':
        return classification_models
    elif task_type == 'regression':
        return regression_models
    elif task_type == 'all':
        return all_models
    else:
        raise ValueError("task_type must be either 'classification', 'regression', or 'all'.")
    