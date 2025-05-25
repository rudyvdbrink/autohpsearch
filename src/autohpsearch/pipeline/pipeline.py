import numpy as np
import pandas as pd
from typing import List, Union, Callable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from scipy import stats
from sklearn.metrics import get_scorer

# Import AutoHPSearch functions
from autohpsearch.search.hptuing import tune_hyperparameters, generate_hypergrid


class OutlierRemover(BaseEstimator, TransformerMixin):
    """Class for detecting and removing outliers from data."""
    
    def __init__(self, method: str = 'zscore', threshold: float = 2.5):
        """
        Initialize the outlier remover.
        
        Parameters
        ----------
        method : str, optional (default='zscore')
            Method to use for outlier detection. Options:
            - 'zscore': Remove samples with features having Z-scores above threshold
            - 'iqr': Remove samples with features outside IQR * threshold
        threshold : float, optional (default=2.5)
            Threshold for outlier detection
        """
        self.method = method
        self.threshold = threshold
        self.mask_ = None
        self.numerical_cols_ = None
    
    def _get_numerical_columns(self, X):
        """Identify numerical columns in the dataset."""
        if hasattr(X, 'select_dtypes'):
            # For pandas DataFrame
            return X.select_dtypes(include=['int', 'float']).columns.tolist()
        else:
            # For numpy arrays, assume all columns are numeric
            return list(range(X.shape[1]))
    
    def fit(self, X):
        """
        Identify outliers in the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        Returns
        -------
        self : object
            Returns self
        """
        # Identify numerical columns
        self.numerical_cols_ = self._get_numerical_columns(X)
        
        if len(self.numerical_cols_) == 0:
            # No numerical columns to process, no outliers to remove
            self.mask_ = np.ones(len(X), dtype=bool)
            return self
        
        # Extract only numerical columns for outlier detection
        if hasattr(X, 'iloc'):
            X_num = X.iloc[:, self.numerical_cols_] if isinstance(self.numerical_cols_[0], int) else X[self.numerical_cols_]
        else:
            X_num = X[:, self.numerical_cols_]
            
        X_array = X_num.to_numpy() if hasattr(X_num, 'to_numpy') else np.asarray(X_num)
        
        # Handle NaN values
        if np.isnan(X_array).any():
            # Create a mask for non-NaN values
            non_nan_mask = ~np.any(np.isnan(X_array), axis=1)
            # Initialize mask with all True (i.e. keep all rows)
            self.mask_ = np.ones(len(X), dtype=bool)
            
            # Only calculate outliers for non-NaN rows
            if self.method == 'zscore':
                valid_rows = np.where(non_nan_mask)[0]
                if len(valid_rows) > 0:
                    valid_data = X_array[non_nan_mask]
                    z_scores = np.abs(stats.zscore(valid_data))
                    valid_mask = np.all(z_scores < self.threshold, axis=1)
                    self.mask_[valid_rows] = valid_mask
            elif self.method == 'iqr':
                q1 = np.nanquantile(X_array, 0.25, axis=0)
                q3 = np.nanquantile(X_array, 0.75, axis=0)
                iqr = q3 - q1
                lower_bound = q1 - (self.threshold * iqr)
                upper_bound = q3 + (self.threshold * iqr)
                
                # Apply bounds check only to non-NaN values
                for i, row in enumerate(X_array):
                    if non_nan_mask[i]:
                        self.mask_[i] = np.all(
                            np.logical_or(
                                np.isnan(row),  # NaN values are not outliers
                                np.logical_and(row >= lower_bound, row <= upper_bound)
                            )
                        )
        else:
            # No NaN values, proceed normally
            if self.method == 'zscore':
                z_scores = np.abs(stats.zscore(X_array))
                self.mask_ = np.all(z_scores < self.threshold, axis=1)
            elif self.method == 'iqr':
                q1 = np.quantile(X_array, 0.25, axis=0)
                q3 = np.quantile(X_array, 0.75, axis=0)
                iqr = q3 - q1
                lower_bound = q1 - (self.threshold * iqr)
                upper_bound = q3 + (self.threshold * iqr)
                
                within_bounds = np.logical_and(
                    X_array >= lower_bound,
                    X_array <= upper_bound
                )
                self.mask_ = np.all(within_bounds, axis=1)
            else:
                raise ValueError(f"Unknown outlier detection method: {self.method}")
        
        return self
    
    def transform(self, X, y):
        """
        Remove outliers from the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        X_transformed : array-like of shape (n_samples_new, n_features)
            Data with outliers removed
        """
        if self.mask_ is None:
            return X, y
        
        if hasattr(X, 'iloc') and hasattr(y, 'iloc'):
            return X.iloc[self.mask_], y.iloc[self.mask_]
        elif hasattr(X, 'iloc') and not hasattr(y, 'iloc'):
            return X.iloc[self.mask_], y[self.mask_]
        elif not hasattr(X, 'iloc') and hasattr(y, 'iloc'):
            return X[self.mask_], y.iloc[self.mask_]
        else:
            return X[self.mask_], y[self.mask_]
    
    def fit_transform(self, X, y):
        """
        Fit to data, then transform it.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        X_transformed : array-like of shape (n_samples_new, n_features)
            Data with outliers removed
        y_transformed : array-like of shape (n_samples_new,)
        """
        return self.fit(X).transform(X, y)
    
    def get_mask(self):
        """Return the mask of non-outlier samples."""
        return self.mask_


class TargetTransformer(BaseEstimator, TransformerMixin):
    """Class for applying transformations to the target variable."""
    
    def __init__(self, transform_method: str = 'none', apply_inverse_transform: bool = True):
        """
        Initialize the target transformer.
        
        Parameters
        ----------
        transform_method : str, optional (default='none')
            Transformation to apply. Options:
            - 'none': No transformation
            - 'log': Natural logarithm
            - 'log1p': Natural logarithm of 1 + x
            - 'sqrt': Square root
        apply_inverse_transform : bool, optional (default=True)
            Whether to apply inverse transformation when predicting
        """
        self.transform_method = transform_method
        self.apply_inverse_transform = apply_inverse_transform
        self.lambda_ = None
    
    def fit(self, y):
        """
        Fit the transformer on the target data.
        
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Returns self
        """
        # Convert pandas Series to numpy array
        if hasattr(y, 'values'):
            y = y.values        
            
        if self.transform_method == 'log':
            # Check for non-positive values in data for logarithm
            if np.any(y <= 0):
                raise ValueError("Log transformation requires positive values")
                
        return self
    
    def transform(self, y):
        """
        Apply transformation to the target.
        
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        y_transformed : array-like of shape (n_samples,)
            Transformed target values
        """
        # Store original type to handle pandas Series
        is_pandas = hasattr(y, 'values')
        if is_pandas:
            y = y.values
        
        if self.transform_method == 'none':
            return y
        elif self.transform_method == 'log':
            # Check for zeros or negative values
            if np.any(y <= 0):
                raise ValueError("Log transformation cannot be applied to zero or negative values")
            return np.log(y)
        elif self.transform_method == 'log1p':
            # Check for negative values
            if np.any(y < 0):
                raise ValueError("Log1p transformation cannot be applied to negative values")
            return np.log1p(y)
        elif self.transform_method == 'sqrt':
            # Check for negative values
            if np.any(y < 0):
                raise ValueError("Square root cannot be applied to negative values")
            return np.sqrt(y)
        else:
            raise ValueError(f"Unknown transformation: {self.transform_method}")
    
    def inverse_transform(self, y):
        """
        Apply inverse transformation to the target.
        
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Transformed target values
            
        Returns
        -------
        y_original : array-like of shape (n_samples,)
            Original target values
        """
        # Store original type to handle pandas Series
        is_pandas = hasattr(y, 'values')
        if is_pandas:
            y = y.values
            
        if not self.apply_inverse_transform:
            return y
        
        if self.transform_method == 'none':
            return y
        elif self.transform_method == 'log':
            return np.exp(y)
        elif self.transform_method == 'log1p':
            return np.expm1(y)
        elif self.transform_method == 'sqrt':
            return np.square(y)
        else:
            raise ValueError(f"Unknown transformation: {self.transform_method}")


class AutoMLPipeline:
    """
    A complete machine learning pipeline for automatic preprocessing, model training, 
    and evaluation.
    """
    
    def __init__(self,
                 task_type: str = 'classification',
                 remove_outliers: bool = False,
                 outlier_method: str = 'zscore',
                 outlier_threshold: float = 3.0,
                 num_imputation_strategy: str = 'mean',
                 cat_imputation_strategy: str = 'most_frequent',
                 cat_encoding_method: str = 'auto',
                 max_onehot_cardinality: int = 10,
                 scaling_method: str = 'standard',
                 target_transform: str = 'none',
                 model_name: Union[str, List[str], None] = None,
                 scoring: Union[str, Callable] = None,
                 search_type: str = 'random',
                 n_iter: int = 30,
                 cv: int = 5,
                 verbose: bool = True):
        """
        Initialize the AutoML pipeline.
        
        Parameters
        ----------
        task_type : str, optional (default='classification')
            Type of machine learning task: 'classification' or 'regression'
        remove_outliers : bool, optional (default=False)
            Whether to remove outliers from training data
        outlier_method : str, optional (default='zscore')
            Method for outlier detection: 'zscore' or 'iqr'
        outlier_threshold : float, optional (default=2.5)
            Threshold for outlier detection
        num_imputation_strategy : str, optional (default='mean')
            Strategy for imputing missing values in numerical features:
            'mean', 'median', 'most_frequent', 'constant', 'knn'
        cat_imputation_strategy : str, optional (default='most_frequent')
            Strategy for imputing missing values in categorical features:
            'most_frequent', 'constant'
        cat_encoding_method : str, optional (default='auto')
            Method for encoding categorical variables:
            'onehot', 'ordinal', 'auto' (chooses based on cardinality)
        max_onehot_cardinality : int, optional (default=10)
            Maximum cardinality for one-hot encoding when cat_encoding_method='auto'
        scaling_method : str, optional (default='standard')
            Method for scaling numerical features:
            'standard', 'minmax', 'robust', 'none'
        target_transform : str, optional (default='none')
            Transformation to apply to the target (for regression only):
            'none', 'log', 'log1p', 'sqrt'
        model_name : str, list of str, or None, optional (default=None)
            Model name or list of model names to include in hyperparameter search.
            If None, all models suitable for task_type will be used.
        scoring : str or callable, optional (default=None)
            Scoring metric to use for model evaluation.
            Default: 'balanced_accuracy' for classification, 'r2' for regression
        search_type : str, optional (default='random')
            Type of hyperparameter search: 'random', 'grid', 'bayesian'
        n_iter : int, optional (default=30)
            Number of iterations for random or bayesian search
        cv : int, optional (default=5)
            Number of cross-validation folds
        verbose : bool, optional (default=True)
            Whether to print progress information
        """
        self.task_type = task_type
        
        # Outlier removal settings
        self.remove_outliers = remove_outliers
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        
        # Imputation settings
        self.num_imputation_strategy = num_imputation_strategy
        self.cat_imputation_strategy = cat_imputation_strategy
        
        # Encoding settings
        self.cat_encoding_method = cat_encoding_method
        self.max_onehot_cardinality = max_onehot_cardinality
        
        # Scaling settings
        self.scaling_method = scaling_method
        
        # Target transformation settings
        self.target_transform = target_transform
        
        # Model training settings
        self.model_name = model_name
        
        if scoring is None:
            self.scoring = 'balanced_accuracy' if task_type == 'classification' else 'r2'
        else:
            self.scoring = scoring
        
        self.search_type = search_type
        self.n_iter = n_iter
        self.cv = cv
        self.verbose = verbose
        
        # Initialized during fit
        self.numerical_features_ = None
        self.categorical_features_ = None
        self.outlier_remover_ = None
        self.preprocessor_ = None
        self.target_transformer_ = None
        self.results_ = None
        self.best_model_ = None
        self.feature_names_ = None
        self.outlier_mask_ = None
    
    def _identify_features(self, X):
        """Identify numerical and categorical features in the dataset."""
        if hasattr(X, 'select_dtypes'):
            # DataFrame
            numeric_cols = X.select_dtypes(include=['int', 'float']).columns.tolist()
            categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            
            # Check for numeric columns with low cardinality that might be categorical
            for col in numeric_cols.copy():
                if X[col].nunique() < min(20, len(X[col]) * 0.05):
                    categorical_cols.append(col)
                    numeric_cols.remove(col)
        else:
            # Assume all features are numeric for numpy arrays
            numeric_cols = list(range(X.shape[1]))
            categorical_cols = []
        
        if self.verbose:
            print(f"Identified {len(numeric_cols)} numerical features and {len(categorical_cols)} categorical features")
        
        return numeric_cols, categorical_cols
    
    def _compute_cardinality(self, X):
        """
        Compute cardinality of categorical features and determine which features
        should use one-hot encoding vs. ordinal encoding.
        
        Parameters
        ----------
        X : array-like or DataFrame
            Input features to analyze for cardinality
        
        Returns
        -------
        self : object
            Returns self with onehot_features_ and ordinal_features_ attributes set
        """
        if not self.categorical_features_:
            self.onehot_features_ = []
            self.ordinal_features_ = []
            return self
        
        # Calculate cardinality for each categorical feature
        cardinalities = {}
        
        if hasattr(X, 'iloc'):  # DataFrame
            for col in self.categorical_features_:
                cardinalities[col] = X[col].nunique()
        else:  # numpy array
            for i in self.categorical_features_:
                cardinalities[i] = len(np.unique(X[:, i]))
        
        # Split features based on cardinality threshold
        self.onehot_features_ = [feat for feat, card in cardinalities.items() 
                            if card <= self.max_onehot_cardinality]
        
        self.ordinal_features_ = [feat for feat, card in cardinalities.items() 
                                if card > self.max_onehot_cardinality]
        
        if self.verbose:
            print(f"Using one-hot encoding for {len(self.onehot_features_)} categorical features")
            print(f"Using ordinal encoding for {len(self.ordinal_features_)} categorical features")
    
        return self
    
    def _create_preprocessor(self):
        """Create a preprocessing pipeline based on the settings."""
        transformers = []
        
        # Numerical feature preprocessing
        num_steps = []
        
        # Imputation for numerical features
        if self.num_imputation_strategy == 'knn':
            num_imputer = KNNImputer(n_neighbors=5)
        else:
            num_imputer = SimpleImputer(strategy=self.num_imputation_strategy)
        
        num_steps.append(('imputer', num_imputer))
        
        # Scaling for numerical features
        if self.scaling_method == 'standard':
            num_steps.append(('scaler', StandardScaler()))
        elif self.scaling_method == 'minmax':
            num_steps.append(('scaler', MinMaxScaler()))
        elif self.scaling_method == 'robust':
            num_steps.append(('scaler', RobustScaler()))
        
        if self.numerical_features_:
            transformers.append((
                'num', SklearnPipeline(num_steps), self.numerical_features_
            ))
        
        # Categorical feature preprocessing
        if self.categorical_features_:
            # Handle different encoding methods for categorical features
            if self.cat_encoding_method == 'onehot':
                # Use one-hot encoding for all categorical features
                cat_transformer = SklearnPipeline([
                    ('imputer', SimpleImputer(strategy=self.cat_imputation_strategy)),
                    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ])
                transformers.append(('cat', cat_transformer, self.categorical_features_))
                
            elif self.cat_encoding_method == 'ordinal':
                # Use ordinal encoding for all categorical features
                cat_transformer = SklearnPipeline([
                    ('imputer', SimpleImputer(strategy=self.cat_imputation_strategy)),
                    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                ])
                transformers.append(('cat', cat_transformer, self.categorical_features_))
                
            elif self.cat_encoding_method == 'auto':
                # Use mixed encoding based on feature cardinality (computed earlier)
                cat_encoders = []
                
                # Handle one-hot encoded features
                if self.onehot_features_:
                    onehot_transformer = SklearnPipeline([
                        ('imputer', SimpleImputer(strategy=self.cat_imputation_strategy)),
                        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                    ])
                    transformers.append(('cat_onehot', onehot_transformer, self.onehot_features_))
                
                # Handle ordinal encoded features
                if self.ordinal_features_:
                    ordinal_transformer = SklearnPipeline([
                        ('imputer', SimpleImputer(strategy=self.cat_imputation_strategy)),
                        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                    ])
                    transformers.append(('cat_ordinal', ordinal_transformer, self.ordinal_features_))
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        return preprocessor
    
    def fit(self, X_train, y_train, X_test, y_test):
        """
        Fit the pipeline on training data and evaluate on test data.
        
        Parameters
        ----------
        X_train : array-like or DataFrame of shape (n_samples, n_features)
            Training features
        y_train : array-like of shape (n_samples,)
            Training target values
        X_test : array-like or DataFrame of shape (n_samples, n_features)
            Test features
        y_test : array-like of shape (n_samples,)
            Test target values
            
        Returns
        -------
        self : object
            Returns self
        """
        if self.verbose:
            print("Starting AutoML pipeline fitting process...")
        
        # Store original feature names if available
        if hasattr(X_train, 'columns'):
            self.feature_names_ = X_train.columns.tolist()                

        # Identify numerical and categorical features
        self.numerical_features_, self.categorical_features_ = self._identify_features(X_train)

        # If we are using automatic categorical encoding, compute cardinality
        if self.cat_encoding_method == 'auto':
            self._compute_cardinality(X_train)
        
        # Step 1: Remove outliers (optional, only from training data)
        if self.remove_outliers:
            if self.verbose:
                print(f"Removing outliers using {self.outlier_method} method...")
            
            self.outlier_remover_ = OutlierRemover(
                method=self.outlier_method,
                threshold=self.outlier_threshold
            )
            
            # Get N for reference
            y_train_len = len(y_train)
            
            # Fit and transform on training data only
            self.outlier_remover_.fit(X_train)
            X_train, y_train = self.outlier_remover_.transform(X_train, y_train)
            
            if self.verbose:
                n_removed = y_train_len - len(y_train)
                print(f"Removed {n_removed} outliers ({n_removed/y_train_len*100:.1f}% of training data)")
        
        # Step 2: Create preprocessor for missing value imputation, encoding, and scaling
        if self.verbose:
            print("Creating preprocessing pipeline...")
        
        self.preprocessor_ = self._create_preprocessor()
        
        # Step 3: Apply target transformation (for regression only)
        if self.task_type == 'regression' and self.target_transform != 'none':
            if self.verbose:
                print(f"Applying {self.target_transform} transformation to target variable...")
            
            self.target_transformer_ = TargetTransformer(transform_method=self.target_transform)
            y_train = self.target_transformer_.fit_transform(y_train)
        
        # Step 4: Fit preprocessor on training data
        if self.verbose:
            print("Fitting preprocessor on training data...")
        
        X_train_transformed = self.preprocessor_.fit_transform(X_train)
        
        # Step 5: Generate hyperparameter grid
        if self.verbose:
            print("Generating hyperparameter grid...")
        
        hypergrid = generate_hypergrid(
            model_name=self.model_name,
            task_type=self.task_type
        )
        
        # Step 6: Apply preprocessor to test data
        if self.verbose:
            print("Transforming test data...")
        
        X_test_transformed = self.preprocessor_.transform(X_test)
        
        # Step 7: Tune hyperparameters
        if self.verbose:
            print(f"Running hyperparameter search with {self.search_type} search...")
        
        self.results_ = tune_hyperparameters(
            X_train_transformed, y_train,
            X_test_transformed, y_test,
            hypergrid=hypergrid,
            scoring=self.scoring,
            cv=self.cv,
            task_type=self.task_type,
            search_type=self.search_type,
            n_iter=self.n_iter,
            verbose=self.verbose
        )
        
        # Get the best model
        self.best_model_ = self.results_['best_model']
        
        if self.verbose:
            print("AutoML pipeline fitting complete!")
            print(f"Best model: {self.results_['results'].index[0]}")
            print(f"Test score ({self.scoring}): {self.results_['results']['test_score'].iloc[0]:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Input features
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values
        """
        if self.best_model_ is None:
            raise ValueError("Model has not been fitted. Call 'fit' first.")
        
        # Apply preprocessing
        X_transformed = self.preprocessor_.transform(X)
        
        # Make predictions
        predictions = self.best_model_.predict(X_transformed)
        
        # Inverse transform target (for regression)
        if self.task_type == 'regression' and self.target_transform != 'none' and self.target_transformer_ is not None:
            predictions = self.target_transformer_.inverse_transform(predictions)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X (classification only).
        
        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Input features
            
        Returns
        -------
        y_proba : array-like of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
        
        if self.best_model_ is None:
            raise ValueError("Model has not been fitted. Call 'fit' first.")
        
        if not hasattr(self.best_model_, 'predict_proba'):
            raise ValueError("The best model does not support probability predictions")
        
        # Apply preprocessing
        X_transformed = self.preprocessor_.transform(X)
        
        # Return probability predictions
        return self.best_model_.predict_proba(X_transformed)
    
    def get_results(self):
        """Return the results of the hyperparameter search."""
        if self.results_ is None:
            raise ValueError("Model has not been fitted. Call 'fit' first.")
        
        return self.results_
    
    def get_best_model(self):
        """Return the best model found during hyperparameter search."""
        if self.best_model_ is None:
            raise ValueError("Model has not been fitted. Call 'fit' first.")
        
        return self.best_model_
    
    def score(self, X, y):
        """
        Calculate the score of the best model on given data.
        
        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Input features
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        score : float
            Score of the best model on the given data
        """
        if self.best_model_ is None:
            raise ValueError("Model has not been fitted. Call 'fit' first.")
        
        # Apply preprocessing
        X_transformed = self.preprocessor_.transform(X)
        
        # Transform target if needed (for regression)
        if self.task_type == 'regression' and self.target_transform != 'none' and self.target_transformer_ is not None:
            y_transformed = self.target_transformer_.transform(y)
            score = get_scorer(self.scoring)(self.best_model_, X_transformed, y_transformed)
        else:
            score = get_scorer(self.scoring)(self.best_model_, X_transformed, y)
        
        return score
    
    def save(self, directory=None, filename=None):
        """
        Save the fitted pipeline to a file with automatic versioning.
        
        Parameters
        ----------
        directory : str, optional (default=None)
            Directory where the model will be saved. If None, uses './models'
            in a platform-compatible way.
        filename : str, optional (default=None)
            Base filename for the saved model. If None, a default name will be used.
            The final filename will include a version number.
            
        Returns
        -------
        str
            Path to the saved model file
        
        Notes
        -----
        This method requires the pipeline to be fitted first.
        The saved file includes the entire pipeline with preprocessing components and model.
        """
        import os
        import joblib
        import datetime
        import re
        
        if self.best_model_ is None:
            raise ValueError("Model has not been fitted. Call 'fit' first.")
        
        # Use a platform-compatible default directory
        if directory is None:
            directory = os.path.join('.', 'models')
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Generate default filename if not provided
        if filename is None:
            model_type = type(self.best_model_).__name__
            task_suffix = "clf" if self.task_type == "classification" else "reg"
            timestamp = datetime.datetime.now().strftime("%Y%m%d")
            filename = f"automl_{task_suffix}_{model_type}_{timestamp}"
        
        # Check existing files to determine version number
        pattern = re.compile(f"{re.escape(filename)}_v(\\d+)\\.joblib$")
        version = 1
        
        if os.path.exists(directory):
            for f in os.listdir(directory):
                match = pattern.match(f)
                if match:
                    v = int(match.group(1))
                    version = max(version, v + 1)
        
        # Create final filename with version
        final_filename = f"{filename}_v{version}.joblib"
        file_path = os.path.join(directory, final_filename)
        
        # Get current datetime in UTC
        current_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create a dictionary with all important components
        pipeline_dict = {
            'preprocessor': self.preprocessor_,
            'best_model': self.best_model_,
            'target_transformer': self.target_transformer_,
            'feature_names': self.feature_names_,
            'task_type': self.task_type,
            'numerical_features': self.numerical_features_,
            'categorical_features': self.categorical_features_,
            'scoring': self.scoring,
            'outlier_remover': self.outlier_remover_ if self.remove_outliers else None,
            'results': self.results_,  # Save hyperparameter search results
            'metadata': {
                'created_at': datetime.datetime.now().isoformat(),
                'model_type': type(self.best_model_).__name__,
                'version': version,
                'model_params': self.best_model_.get_params(),
                'scoring': self.scoring,
                'saved_by': 'rudyvdbrink',  # Current user's login
                'saved_at': current_time  # Current UTC date and time
            }
        }
        
        # Save the dictionary
        joblib.dump(pipeline_dict, file_path)
        
        if self.verbose:
            print(f"Pipeline saved to {file_path}")
        
        return file_path
    
    @classmethod
    def load(cls, file_path=None, directory=None, verbose=True):
        """
        Load a saved pipeline from file. 
        
        If file_path is None, the most recently saved pipeline in the specified directory will be loaded.
        
        Parameters
        ----------
        file_path : str, optional (default=None)
            Path to the saved model file. If None, the most recent file in the directory will be loaded.
        directory : str, optional (default=None)
            Directory to search for models when file_path is None. If None, uses './models'
            in a platform-compatible way.
        verbose : bool, optional (default=True)
            Whether to print information about the loaded model
            
        Returns
        -------
        AutoMLPipeline
            Loaded pipeline instance
            
        Raises
        ------
        FileNotFoundError
            If file_path is not found or directory contains no model files
        """
        import joblib
        import os
        import glob
        
        # Use a platform-compatible default directory
        if directory is None:
            directory = os.path.join('.', 'models')
        
        if file_path is None:
            # No specific file provided, find the most recent one in the directory
            if not os.path.exists(directory):
                raise FileNotFoundError(f"Directory not found: {directory}")
            
            # Look for .joblib files in the directory
            model_files = glob.glob(os.path.join(directory, "*.joblib"))
            
            if not model_files:
                raise FileNotFoundError(f"No model files found in directory: {directory}")
            
            # Get the most recently modified file
            file_path = max(model_files, key=os.path.getmtime)
            
            if verbose:
                print(f"Loading most recent model: {os.path.basename(file_path)}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found at {file_path}")
        
        # Load the dictionary
        pipeline_dict = joblib.load(file_path)
        
        # Create a new instance
        pipeline = cls(
            task_type=pipeline_dict['task_type'],
            verbose=verbose
        )
        
        # Restore components
        pipeline.preprocessor_ = pipeline_dict['preprocessor']
        pipeline.best_model_ = pipeline_dict['best_model']
        pipeline.target_transformer_ = pipeline_dict['target_transformer']
        pipeline.feature_names_ = pipeline_dict['feature_names']
        pipeline.numerical_features_ = pipeline_dict['numerical_features']
        pipeline.categorical_features_ = pipeline_dict['categorical_features']
        pipeline.scoring = pipeline_dict['scoring']
        pipeline.outlier_remover_ = pipeline_dict['outlier_remover']
        pipeline.results_ = pipeline_dict.get('results')  # Restore hyperparameter search results if available
        
        if verbose:
            print(f"Pipeline loaded from {file_path}")
            print(f"Model type: {pipeline_dict['metadata']['model_type']}")
            print(f"Created at: {pipeline_dict['metadata']['created_at']}")
            print(f"Version: {pipeline_dict['metadata']['version']}")
            
            # Show who saved the model and when
            if 'saved_by' in pipeline_dict['metadata']:
                print(f"Saved by: {pipeline_dict['metadata']['saved_by']}")
                print(f"Saved at: {pipeline_dict['metadata']['saved_at']}")
        
        return pipeline


# Example usage:
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Create pipeline for classification
    pipeline = AutoMLPipeline(
        task_type='classification',
        remove_outliers=True,
        model_name=['random_forest_clf', 'gradient_boosting_clf'],  # Specify models to use
        scoring='balanced_accuracy',
        n_iter=10,
        verbose=True
    )
    
    # Fit pipeline
    pipeline.fit(X_train, y_train, X_test, y_test)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    
    # Get results
    results = pipeline.get_results()
    print(f"Best model: {results['results'].index[0]}")
    print(f"Test score: {results['results']['test_score'].iloc[0]:.4f}")