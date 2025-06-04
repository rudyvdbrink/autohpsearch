# %% import libraries

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin

# %% classes for data clenaing

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

