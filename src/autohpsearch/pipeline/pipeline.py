# %% import libraries

import numpy as np
import pandas as pd
from typing import List, Union, Callable

from autohpsearch.search.hptuing import tune_hyperparameters, generate_hypergrid
from autohpsearch.pipeline.reporter import DataReporter
from autohpsearch.pipeline.cleaning import Preprocessor


# %% class for an end-to-end pipeline

class AutoMLPipeline:
    """
    A complete machine learning pipeline for automatic preprocessing, model training, 
    and evaluation.
    """
    
    def __init__(self,
                 task_type: str = 'classification',
                 remove_outliers: bool = False,
                 drop_duplicate_rows: bool = False,
                 outlier_method: str = 'zscore',
                 outlier_threshold: float = 3.0,
                 num_imputation_strategy: str = 'mean',
                 cat_imputation_strategy: str = 'most_frequent',
                 cat_encoding_method: str = 'onehot',
                 max_onehot_cardinality: int = 10,
                 apply_smote: bool = False,
                 smote_kwargs: dict = None,
                 scaling_method: str = 'minmax',
                 filter_features: bool = False,
                 filter_threshold: float = 0.95,
                 filter_method: str = 'spearman',
                 target_transform: str = 'none',
                 model_name: Union[str, List[str], None] = None,
                 scoring: Union[str, Callable] = None,
                 search_type: str = 'random',
                 n_iter: int = 30,
                 cv: int = 5,
                 verbose: bool = False):
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
        cat_encoding_method : str, optional (default='onehot')
            Method for encoding categorical variables:
            'onehot', 'ordinal', 'auto' (chooses based on cardinality)
        max_onehot_cardinality : int, optional (default=10)
            Maximum cardinality for one-hot encoding when cat_encoding_method='auto'
        apply_smote : bool, optional
            Whether to apply SMOTE/SMOTENC/SMOTEN to the training data after outlier removal
        smote_kwargs : dict, optional
            Extra keyword arguments for the SMOTEApplier
        scaling_method : str, optional (default='minmax')
            Method for scaling numerical features:
            'standard', 'minmax', 'robust', 'none'
        filter_features : bool, optional (default=False)
            Whether to filter features based on correlation
        filter_threshold : float, optional (default=0.95)
            Threshold for feature filtering (only used if filter_features=True)
        filter_method : str, optional (default='spearman')
            Method for feature filtering: 'spearman', 'pearson', 'kendall'
        target_transform : str, optional (default='none')
            Transformation to apply to the target (for regression only):
            'none', 'log', 'log1p', 'sqrt'
        model_name : str, list of str, or None, optional (default=None)
            Model name or list of model names to include in hyperparameter search.
            If None, all models suitable for task_type will be used.
        scoring : str or callable, optional (default=None)
            Scoring metric to use for model evaluation.
            Default: 'balanced_accuracy' for classification, 'neg_root_mean_squared_error' for regression
        search_type : str, optional (default='random')
            Type of hyperparameter search: 'random', 'grid', 'bayesian'
        n_iter : int, optional (default=30)
            Number of iterations for random or bayesian search
        cv : int, optional (default=5)
            Number of cross-validation folds
        verbose : bool, optional (default=False)
            Whether to print progress information
        """
        self.task_type = task_type
        
        # Outlier removal settings
        self.remove_outliers = remove_outliers
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold

        # Duplicate row removal
        self.drop_duplicate_rows = drop_duplicate_rows
        
        # Imputation settings
        self.num_imputation_strategy = num_imputation_strategy
        self.cat_imputation_strategy = cat_imputation_strategy
        
        # Encoding settings
        self.cat_encoding_method = cat_encoding_method
        self.max_onehot_cardinality = max_onehot_cardinality
        
        # Scaling settings
        self.scaling_method = scaling_method

        # SMOTE settings
        self.apply_smote = apply_smote
        self.smote_kwargs = smote_kwargs if smote_kwargs is not None else {}

        # Feature filtering settings
        self.filter_features = filter_features
        self.filter_threshold = filter_threshold
        self.filter_method = filter_method
        
        # Target transformation settings
        self.target_transform = target_transform
        
        # Model training settings
        self.model_name = model_name
        
        if scoring is None:
            self.scoring = 'balanced_accuracy' if task_type == 'classification' else 'neg_root_mean_squared_error'
        else:
            self.scoring = scoring
        
        self.search_type = search_type
        self.n_iter = n_iter
        self.cv = cv
        self.verbose = verbose

        # Make a dictionary to store the settings for the preprocessor
        self.preprocessor_params = {
            'task_type': self.task_type,
            'remove_outliers': self.remove_outliers,
            'drop_duplicate_rows': self.drop_duplicate_rows,
            'outlier_method': self.outlier_method,
            'outlier_threshold': self.outlier_threshold,
            'num_imputation_strategy': self.num_imputation_strategy,
            'cat_imputation_strategy': self.cat_imputation_strategy,
            'cat_encoding_method': self.cat_encoding_method,
            'max_onehot_cardinality': self.max_onehot_cardinality,
            'apply_smote': self.apply_smote,
            'smote_kwargs': self.smote_kwargs,
            'scaling_method': self.scaling_method,
            'filter_features': self.filter_features,
            'filter_threshold': self.filter_threshold,
            'filter_method': self.filter_method,
            'target_transform': self.target_transform,
            'verbose': self.verbose
        }
        
        # Initialized during fit
        self.numerical_features_ = None
        self.categorical_features_ = None
        self.outlier_remover_ = None
        self.preprocessor_ = None
        self.target_transformer_ = None
        self.results_ = None
        self.best_model_ = None
        self.feature_names_ = None
        self.labels_ = None
        self.label_mapping_ = None
        self.transformed_feature_names_ = None  
        self.outlier_mask_ = None
        self.columns_to_drop_ = None

        # Store data for reporting
        self.X_train_original_ = None
        self.y_train_original_ = None
        self.X_train_processed_ = None
        self.y_train_processed_ = None
        self.X_test_original_ = None
        self.y_test_original_ = None
        self.X_test_processed_ = None
        self.y_test_processed_ = None

    def fit(self, X_train, y_train, X_test, y_test=None):
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
        y_test : array-like of shape (n_samples,), optional (default=None)
            Test target values
            If set to None, the pipeline will not evaluate on test data.
            
        Returns
        -------
        self : object
            Returns self
        """

        if self.verbose:
            print("Starting AutoML pipeline fitting process...")

        # Store original data for reporting
        self.X_train_original_ = X_train.copy() if hasattr(X_train, 'copy') else X_train
        self.y_train_original_ = y_train.copy() if hasattr(y_train, 'copy') else y_train
        self.X_test_original_  = X_test.copy() if hasattr(X_test, 'copy') else X_test
        self.y_test_original_  = y_test.copy() if y_test is not None and hasattr(y_test, 'copy') else y_test

        # Instantiate a preprocessor
        self.preprocessor_ = Preprocessor(**self.preprocessor_params)
        # Clean the data
        X_train_processed, y_train_processed, X_test_processed, y_test_processed = self.preprocessor_.preprocess(X_train, y_train, X_test, y_test)

        # Store data for reporting
        self.X_train_processed_ = X_train_processed
        self.y_train_processed_ = y_train_processed
        self.X_test_processed_  = X_test_processed
        self.y_test_processed_  = y_test_processed

        # Store variables from preprocessor
        self.labels_ = self.preprocessor_.labels_
        self.label_mapping_ = self.preprocessor_.label_mapping_
        self.feature_names_ = self.preprocessor_.feature_names_
        self.transformed_feature_names_ = self.preprocessor_.transformed_feature_names_
        self.numerical_features_ = self.preprocessor_.numerical_features_
        self.categorical_features_ = self.preprocessor_.categorical_features_
        self.outlier_remover_ = self.preprocessor_.outlier_remover_
        self.columns_to_drop_ = self.preprocessor_.columns_to_drop_

        # Step 6: Generate hyperparameter grid
        if self.verbose:
            print("Generating hyperparameter grid...")
        
        hypergrid = generate_hypergrid(
            model_name=self.model_name,
            task_type=self.task_type
        )

        self.results_ = tune_hyperparameters(
            X_train_processed, y_train_processed,
            X_test_processed, y_test_processed,
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
        
        X_transformed = self.apply_preprocessing(X)
        
        # Make predictions
        predictions = np.squeeze(self.best_model_.predict(X_transformed))
        
        # Inverse transform target (for regression)
        if self.task_type == 'regression' and self.target_transform != 'none' and self.target_transformer_ is not None:
            predictions = self.target_transformer_.inverse_transform(predictions)

        # Convert numeric predictions back to original labels
        if self.task_type == 'classification' and self.label_mapping_ is not None:
            predictions = self.preprocessor_._convert_float_to_target(predictions)
        
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
        X_transformed = self.apply_preprocessing(X)
        
        # Return probability predictions
        return self.best_model_.predict_proba(X_transformed)
    
    def apply_preprocessing(self, X):
        # Apply preprocessing
        X_transformed = self.preprocessor_.preprocessor_.transform(X)
        feature_names = self.preprocessor_._extract_feature_names()
        X_transformed = pd.DataFrame(X_transformed, columns=feature_names)

        if self.filter_features and self.preprocessor_.filter_ is not None:
            X_transformed = self.preprocessor_.filter_.filter(X_transformed)

        return X_transformed

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
    
    def get_feature_names(self, input_type='transformed'):
        """
        Get feature names for the pipeline.
        
        Parameters
        ----------
        input_type : str, optional (default='transformed')
            Type of feature names to return:
            - 'original': Original feature names before preprocessing
            - 'transformed': Feature names after preprocessing
            
        Returns
        -------
        list
            List of feature names
        """
        if input_type == 'original':
            if self.feature_names_ is not None:
                return self.feature_names_
            else:
                return [f'feature_{i}' for i in range(len(self.numerical_features_) + len(self.categorical_features_))]
        
        elif input_type == 'transformed':
            if self.transformed_feature_names_ is not None:
                return self.transformed_feature_names_
            else:
                return []
        
        else:
            raise ValueError("input_type must be 'original' or 'transformed'")
        
    def generate_data_report(self, report_directory: str = "reports", version: int = None):
        """
        Generate a comprehensive data report in markdown format.
        
        Parameters
        ----------
        report_directory : str, optional (default="reports")
            Directory where the report will be saved
        version : int, optional (default=None)
            Version number for the report. If None, will auto-increment.
            
        Returns
        -------
        str
            Path to the generated report file
        """
        if self.X_train_original_ is None:
            raise ValueError("No training data available. Please call 'fit' first.")
        
        # Create reporter instance
        reporter = DataReporter(report_directory=report_directory)
        
        # Extract version from pipeline if available and version not specified
        if version is None and hasattr(self, '_pipeline_version'):
            version = self._pipeline_version
        
        # Generate the report
        report_path = reporter.generate_report(
            X_train=self.X_train_original_,
            y_train=self.y_train_original_,
            pipeline=self,
            X_train_processed=self.X_train_processed_,
            y_train_processed=self.y_train_processed_,
            X_test=self.X_test_original_,
            y_test=self.y_test_processed_,
            version=version
        )
        
        if self.verbose:
            print(f"Data report generated: {report_path}")
        
        return report_path
    
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
            The final filename will include a zero-padded version number.
            
        Returns
        -------
        str
            Path to the saved model file
        
        Notes
        -----
        This method requires the pipeline to be fitted first.
        The saved file includes the entire pipeline with preprocessing components and model.
        Version numbers are zero-padded to 4 digits (e.g., 0001, 0034).
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
        
        # Store version for report generation
        self._pipeline_version = version
        
        # Create final filename with zero-padded version number (4 digits)
        final_filename = f"{filename}_v{version:04d}.joblib"
        file_path = os.path.join(directory, final_filename)
        
        # Get current datetime in UTC
        current_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create a dictionary with all important components
        pipeline_dict = {
            **self.preprocessor_params,  # Unpack all fields from the preprocessing dictionary
            'model_name': self.model_name,
            'preprocessor': self.preprocessor_,
            'best_model': self.best_model_,
            'target_transformer': self.target_transformer_,
            'feature_names': self.feature_names_,
            'transformed_feature_names': self.transformed_feature_names_,  # NEW: Save transformed feature names
            'numerical_features': self.numerical_features_,
            'categorical_features': self.categorical_features_,
            'scoring': self.scoring,
            'outlier_remover': self.outlier_remover_ if self.remove_outliers else None,
            'results': self.results_,  # Save hyperparameter search results
            'labels': self.labels_,
            'label_mapping': self.label_mapping_,
            'columns_to_drop': self.columns_to_drop_,
            'metadata': {
                'created_at': datetime.datetime.now().isoformat(),
                'model_type': type(self.best_model_).__name__,
                'version': version,
                'model_params': self.best_model_.get_params(),
                'scoring': self.scoring,
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
        pipeline.transformed_feature_names_ = pipeline_dict.get('transformed_feature_names')  # NEW: Load transformed feature names
        pipeline.numerical_features_ = pipeline_dict['numerical_features']
        pipeline.categorical_features_ = pipeline_dict['categorical_features']
        pipeline.scoring = pipeline_dict['scoring']
        pipeline.outlier_remover_ = pipeline_dict['outlier_remover']
        pipeline.remove_outliers = pipeline_dict['outlier_remover'] is not None
        pipeline.results_ = pipeline_dict.get('results')  # Restore hyperparameter search results if available
        pipeline.apply_smote = pipeline_dict.get('apply_smote', False)
        pipeline.smote_kwargs = pipeline_dict.get('smote_kwargs', {})
        pipeline.model_name = pipeline_dict.get('model_name', None)
        pipeline.labels_ = pipeline_dict.get('labels', None)  # Restore labels if available
        pipeline.label_mapping_ = pipeline_dict.get('label_mapping', None)
        pipeline.filter_features = pipeline_dict.get('filter_features', False)
        pipeline.filter_threshold = pipeline_dict.get('filter_threshold', 0.95)
        pipeline.filter_method = pipeline_dict.get('filter_method', 'spearman')

        if verbose:
            print(f"Pipeline loaded from {file_path}")
            print(f"Model type: {pipeline_dict['metadata']['model_type']}")
            print(f"Created at: {pipeline_dict['metadata']['created_at']}")
            print(f"Version: {pipeline_dict['metadata']['version']}")            
            print(f"Saved at: {pipeline_dict['metadata']['saved_at']}")
            
            # Show feature name information
            if pipeline.transformed_feature_names_:
                print(f"Loaded {len(pipeline.transformed_feature_names_)} transformed feature names")
        
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
    
    # Get feature names
    original_names = pipeline.get_feature_names('original')
    transformed_names = pipeline.get_feature_names('transformed')
    
    print(f"Original feature names: {original_names}")
    print(f"Transformed feature names: {transformed_names}")
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    
    # Get results
    results = pipeline.get_results()
    print(f"Best model: {results['results'].index[0]}")
    print(f"Test score: {results['results']['test_score'].iloc[0]:.4f}")