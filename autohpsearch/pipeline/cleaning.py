# %% import libraries

import numpy as np
import pandas as pd
import re

from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.over_sampling import SMOTE, SMOTENC, SMOTEN

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline

from bs4 import BeautifulSoup
from textblob import TextBlob
import unicodedata
import contractions
import stanza

# %% Classes for data cleaning steps

class TextCleaner:
    def __init__(self, 
                 named_entity_recognition=False, 
                 remove_html_tags=False, 
                 remove_urls=False, 
                 remove_email_addresses=False, 
                 remove_special_characters=False, 
                 remove_numbers=False, 
                 whitespace_normalization=False, 
                 correct_encoding_issues=False, 
                 expand_contractions=False, 
                 spelling_correction=False):
        """
        Initialize the TextCleaner with specified cleaning steps.
        """
        self.steps = {
            "named_entity_recognition": named_entity_recognition,
            "remove_html_tags": remove_html_tags,
            "remove_urls": remove_urls,
            "remove_email_addresses": remove_email_addresses,
            "remove_special_characters": remove_special_characters,
            "remove_numbers": remove_numbers,
            "whitespace_normalization": whitespace_normalization,
            "correct_encoding_issues": correct_encoding_issues,
            "expand_contractions": expand_contractions,
            "spelling_correction": spelling_correction
        }

        if self.steps["named_entity_recognition"]:                
                stanza.download('en')  # Ensure the English model is downloaded
                self.ner_pipeline = stanza.Pipeline(lang='en', processors='tokenize,ner')
        
        if self.steps["remove_special_characters"]:
            # Define a regex pattern to match special characters
            self.char_pattern = r"[^\w\s]"

    def transform(self, text_series):
        """
        Apply the specified cleaning steps to a pandas Series containing text.
        """
        # Ensure the input is a pandas Series
        if not isinstance(text_series, pd.Series):
            # Convert to pandas series
            text_series = pd.Series(text_series)        

        def clean_text(text):
            # Skip processing if the text is NaN or not a valid string
            if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
                return text
            
            if self.steps["named_entity_recognition"]:
                text = self._apply_NER(text)
            
            if self.steps["remove_html_tags"]:
                text = BeautifulSoup(text, "html.parser").get_text()
            
            if self.steps["remove_urls"]:
                text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
            
            if self.steps["remove_email_addresses"]:
               text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "(EMAIL ADDRESS)", text)
            
            if self.steps["remove_special_characters"]:
                text = re.sub(self.char_pattern, " ", text)
            
            if self.steps["remove_numbers"]:
                text = re.sub(r"\d+", "", text)
            
            if self.steps["whitespace_normalization"]:
                text = re.sub(r"\s+", " ", text).strip()
            
            if self.steps["correct_encoding_issues"]:
                text = unicodedata.normalize("NFKD", text)
            
            if self.steps["expand_contractions"]:
                text = contractions.fix(text)
            
            if self.steps["spelling_correction"]:
                text = str(TextBlob(text).correct())
            
            return text

        return text_series.apply(clean_text)
    
    def _apply_NER(self, text):
        """
        Apply Named Entity Recognition (NER) to the input text using Stanza.
        Replace named entities in the text with their placeholders.
        """        
        
        # Process the text with the Stanza pipeline
        doc = self.ner_pipeline(text)

        # Replace named entities with their placeholders
        for sentence in doc.sentences:
            for ent in sentence.ents:
                if ent.type != "ORDINAL":  # Skip replacing numbers classified as ORDINAL
                    text = text.replace(ent.text, f"({ent.type})")

        print(text)
        
        return text

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

class SMOTEApplier(BaseEstimator, TransformerMixin):
    """
    Applies the appropriate SMOTE variant (SMOTE, SMOTENC, or SMOTEN) using feature types from the pipeline.
    """

    def __init__(self, pipeline, random_state=None, sampling_strategy='auto'):
        """
        Parameters
        ----------
        pipeline : AutoMLPipeline instance
            The pipeline instance (should have .numerical_features_ and .categorical_features_)
        random_state : int, optional
            Random state for reproducibility
        sampling_strategy : str or float or dict, optional
            Sampling strategy for SMOTE variants
        """
        self.pipeline = pipeline
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self.smote_ = None
        self.categorical_features_ = None

    def _detect_feature_types(self, X):
        # Use the pipeline's feature lists and map to indices if DataFrame
        if hasattr(X, "columns"):
            # Use column names from pipeline
            cat_cols = self.pipeline.categorical_features_
            num_cols = self.pipeline.numerical_features_
            # Get indices for categorical columns
            self.categorical_features_ = [X.columns.get_loc(col) for col in cat_cols]
            return len(num_cols), len(cat_cols)
        else:
            # Assume all float/integer for numpy arrays
            return X.shape[1], 0

    def fit(self, X, y=None):
        n_num, n_cat = self._detect_feature_types(X)
        if n_cat == 0:
            self.smote_ = SMOTE(random_state=self.random_state, sampling_strategy=self.sampling_strategy)
        elif n_num == 0:
            self.smote_ = SMOTEN(random_state=self.random_state, sampling_strategy=self.sampling_strategy)
        else:
            self.smote_ = SMOTENC(
                categorical_features=self.categorical_features_,
                random_state=self.random_state,
                sampling_strategy=self.sampling_strategy
            )
        self.smote_.fit(X, y)
        return self

    def transform(self, X, y):
        X_res, y_res = self.smote_.fit_resample(X, y)
        if isinstance(X, pd.DataFrame):
            X_res = pd.DataFrame(X_res, columns=X.columns)
        if isinstance(y, pd.Series):
            y_res = pd.Series(y_res, name=y.name)
        return X_res, y_res

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)

class FeatureFilter:
    def __init__(self, thresh=0.95, method='spearman'):
        """
        Initializes the FeatureFilter class.

        Args:
            thresh (float): Spearman's rho to use as threshold. Defaults to 0.95.
            method (str): Correlation method to use ('spearman', 'pearson', etc.). Defaults to 'spearman'.
        """
        self.thresh = thresh
        self.method = method
        self.columns_to_drop = []

    def _convert_to_dataframe(self, X):
        """
        Converts numpy arrays to pandas DataFrames if necessary.

        Args:
            X (numpy.ndarray or pandas.DataFrame): Input data.

        Returns:
            pandas.DataFrame: Converted DataFrame.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame or a numpy array.")
        return X

    def fit(self, X):
        """
        Determines which columns to drop based on correlation threshold.

        Args:
            X (numpy.ndarray or pandas.DataFrame): Design matrix.

        Returns:
            list: List of features to drop.
        """
        # Convert numpy array to DataFrame if necessary
        X = self._convert_to_dataframe(X)

        # Compute correlation matrix and get upper triangle
        cm = X.corr(method=self.method).abs()
        upper = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))

        # Identify columns to drop
        self.columns_to_drop = [column for column in upper.columns if any(upper[column] > self.thresh)]

        return self.columns_to_drop

    def filter(self, X):
        """
        Drops the columns identified by the fit method.

        Args:
            X (numpy.ndarray or pandas.DataFrame): Design matrix.

        Returns:
            pandas.DataFrame or numpy.ndarray: Design matrix with features removed.
        """
        # Convert numpy array to DataFrame if necessary
        X = self._convert_to_dataframe(X)

        if not self.columns_to_drop:            
            return X

        # Drop features
        X_filtered = X.drop(self.columns_to_drop, axis=1)

        # Return as numpy array if the original input was a numpy array
        if isinstance(X, np.ndarray):
            return X_filtered.to_numpy()
        return X_filtered
    
# %% Class for a complete preprocessing pipeline

class Preprocessor:
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
                 verbose: bool = False
                 ):
        """
        Initialize the Preprocessor with configuration options.

       Parameters
        ----------
        task_type : str, optional (default='classification')
            Type of machine learning task: 'classification' or 'regression'
        remove_outliers : bool, optional (default=False)
            Whether to remove outliers from training data
        drop_duplicate_rows : bool, optional (default=False)
            Whether to drop duplicate rows from the training data
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
        verbose : bool, optional (default=False)
            Whether to print progress information
        """

        self.task_type = task_type
        self.drop_duplicate_rows = drop_duplicate_rows
        self.cat_encoding_method = cat_encoding_method
        self.max_onehot_cardinality = max_onehot_cardinality
        self.num_imputation_strategy = num_imputation_strategy
        self.cat_imputation_strategy=cat_imputation_strategy
        self.remove_outliers = remove_outliers
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.scaling_method = scaling_method
        self.apply_smote = apply_smote
        self.smote_kwargs = smote_kwargs or {}
        self.filter_features = filter_features
        self.filter_threshold = filter_threshold
        self.filter_method = filter_method
        self.target_transform = target_transform
        self.verbose = verbose

        # Attributes added later
        self.feature_names_ = None
        self.numerical_features_ = None
        self.categorical_features_ = None
        self.labels_ = None
        self.label_mapping_ = None
        self.transformed_feature_names_ = None
        self.outlier_remover_ = None
        self.columns_to_drop_ = None
        self.num_rows_dropped_ = None
        self.preprocessor_ = None

    def preprocess(self, X_train, y_train, X_test, y_test=None):
        """
        Perform preprocessing on the training and test data.

        Parameters:
        - X_train (pd.DataFrame): Training features.
        - y_train (pd.Series): Training target variable.
        - X_test (pd.DataFrame): Test features.
        - y_test (pd.Series): Test target variable.

        Returns:
        - X_train_transformed (pd.DataFrame): Transformed training features.
        - y_train_transformed (pd.Series): Transformed training target variable.
        - X_test_transformed (pd.DataFrame): Transformed test features.
        - y_test_transformed (pd.Series): Transformed test target variable.
        """
        # Store original feature names if available
        if hasattr(X_train, 'columns'):
            self.feature_names_ = X_train.columns.tolist()                

        # Identify numerical and categorical features
        self.numerical_features_, self.categorical_features_ = self._identify_features(X_train)

        # Identify targets
        if self.task_type == 'classification':            
            # Convert target to numeric if necessary (for classification)
            y_train, y_test, self.labels_ = self._convert_target_to_float(y_train=y_train, y_test=y_test)

            # If the target was numeric, get labels from training data
            if self.labels_ is None:
                self.labels_ = self._extract_labels(y_train)

        # If we are using automatic categorical encoding, compute cardinality
        if self.cat_encoding_method == 'auto':
            self._compute_cardinality(X_train)

        # Step 1: Drop duplicate rows (if requested)
        if self.drop_duplicate_rows:
            if self.verbose:
                print("Dropping duplicate rows from training data...")

            convert_to_dataframe = False
            converted_to_series = False
            
            # If X_train is a numpy array, convert it to DataFrame temporarily
            if isinstance(X_train, np.ndarray):
                X_train = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
                convert_to_dataframe = True

            if isinstance(y_train, np.ndarray):    
                y_train = pd.Series(y_train, name='target')
                converted_to_series = True
            
            initial_shape = X_train.shape
            X_train = X_train.drop_duplicates()
            y_train = y_train[X_train.index]
            self.num_rows_dropped_ = initial_shape[0] - X_train.shape[0]
            if self.verbose:
                print(f"Dropped {self.num_rows_dropped_} duplicate rows from training data ({self.num_rows_dropped_/initial_shape[0]*100:.1f}% of training data)")
        
            # If we converted to DataFrame, convert back to numpy array
            if convert_to_dataframe:
                X_train = X_train.to_numpy()

            if converted_to_series:
                y_train = y_train.to_numpy()

        # Step 2: Remove outliers (optional, only from training data)
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
        
        # Step 3: Class balancing using SMOTE (if requested)
        if self.apply_smote:
            if self.task_type != 'classification':
                raise ValueError("SMOTE can only be applied to classification tasks")

            if self.verbose:
                print("Applying SMOTE-based oversampling to the training data...")
            if self.smote_kwargs is None:
                self.smote_kwargs = {}

            self.smote_applier_ = SMOTEApplier(pipeline=self, **self.smote_kwargs)
            X_train, y_train = self.smote_applier_.fit_transform(X_train, y_train)
            if self.verbose:
                print(f"Training samples after SMOTE: {len(X_train)}")
                
        # Step 4: Create preprocessor for missing value imputation, encoding, and scaling
        if self.verbose:
            print("Fitting preprocessor on the data...")
        
        self.preprocessor_ = self._create_preprocessor()        
        X_train_transformed = self.preprocessor_.fit_transform(X_train)
        X_test_transformed  = self.preprocessor_.transform(X_test)
        
        # Step 5: Apply target transformation (for regression only)
        if self.task_type == 'regression' and self.target_transform != 'none':
            if self.verbose:
                print(f"Applying {self.target_transform} transformation to target variable...")
            
            self.target_transformer_ = TargetTransformer(transform_method=self.target_transform)
            y_train = self.target_transformer_.fit_transform(y_train)
            #y_test_processed = self.target_transformer_.transform(y_test)
        
        if self.task_type == 'regression' and self.target_transform != 'none' and self.target_transformer_ is not None:
            if y_test is not None:
                y_test = self.target_transformer_.transform(y_test)
        
        
        # Step 6: Extract and store transformed feature names
        if self.verbose:
            print("Extracting feature names after preprocessing...")
        
        self.transformed_feature_names_ = self._extract_feature_names()
        
        if self.verbose and self.transformed_feature_names_:
            print(f"Extracted {len(self.transformed_feature_names_)} feature names after preprocessing")
            if len(self.transformed_feature_names_) <= 20:
                print(f"Feature names: {self.transformed_feature_names_}")       

        # Make the transformed data a DataFrame with the new feature names
        X_train_transformed = pd.DataFrame(X_train_transformed, columns=self.transformed_feature_names_)
        X_test_transformed  = pd.DataFrame(X_test_transformed, columns=self.transformed_feature_names_)

        # Step 7: Filter features
        if self.filter_features:
            self.filter_ = FeatureFilter(thresh=self.filter_threshold, method=self.filter_method)
            if self.verbose:
                print(f"Filtering features with {self.filter_method} method and threshold {self.filter_threshold}...")
            self.columns_to_drop_ = self.filter_.fit(X_train_transformed)
            if self.columns_to_drop_ is not None:
                X_train_transformed = self.filter_.filter(X_train_transformed)
                if X_test_transformed is not None:
                    X_test_transformed = self.filter_.filter(X_test_transformed)
                if self.verbose:
                    print(f"Dropped {len(self.columns_to_drop_)} features based on correlation threshold")

                # Update the transformed feature names after filtering
                self.transformed_feature_names_ = [col for col in self.transformed_feature_names_ if col not in self.columns_to_drop_]              

        return X_train_transformed, y_train, X_test_transformed, y_test

    def _identify_features(self, X):
        """Identify numerical and categorical features in the dataset."""
        if hasattr(X, 'select_dtypes'):
            # DataFrame
            numeric_cols = X.select_dtypes(include=['int', 'float']).columns.tolist()
            categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            
        else:
            # Assume all features are numeric for numpy arrays
            numeric_cols = list(range(X.shape[1]))
            categorical_cols = []
        
        if self.verbose:
            print(f"Identified {len(numeric_cols)} numerical features and {len(categorical_cols)} categorical features")
        
        return numeric_cols, categorical_cols

    def _convert_target_to_float(self, y_train, y_test):
        """
        Convert the target variables to numeric values if they are categorical and ensure consistent labels across train and test sets.
        Also creates self.label_mapping_, which keeps track of numerical and string versions of the target.

        Parameters
        ----------
        y_train : array-like
            Target variable for training data.
        y_test : array-like
            Target variable for testing data.

        Returns
        -------
        y_train_converted : array-like
            Target variable for training data converted to numeric values if necessary.
        y_test_converted : array-like
            Target variable for testing data converted to numeric values if necessary.
        labels : list
            List of labels corresponding to the numeric values.
        """
        if self.verbose:
            print("Converting categorical target variables to numeric values...")
        
        # Concatenate y_train and y_test to ensure consistent factorization
        if y_test is not None:
            y_combined = pd.concat([pd.Series(y_train), pd.Series(y_test)], ignore_index=True)
        else:
            y_combined = pd.Series(y_train)
        
        # Use pandas.factorize to convert to numeric values
        y_combined_converted, labels = pd.factorize(y_combined)
        
        # Create label mapping dictionary
        self.label_mapping_ = {
            "to_numeric": {label: i for i, label in enumerate(labels)},
            "to_string": {i: label for i, label in enumerate(labels)}
        }
        
        # Split back into y_train and y_test
        y_train_converted = y_combined_converted[:len(y_train)]

        if y_test is not None:
            y_test_converted = y_combined_converted[len(y_train):]
        else:
            y_test_converted = None
        
        return y_train_converted, y_test_converted, labels.tolist()       
       
    def _convert_float_to_target(self, y):
        """
        Convert a numeric target variable back to its original categorical labels using self.label_mapping_.

        Parameters
        ----------
        y : array-like
            Numeric target variable to be converted back to categorical labels.

        Returns
        -------
        y_converted : array-like
            Target variable converted back to categorical labels.
        """
        if self.verbose:
            print("Converting numeric target variable back to categorical labels...")

        # Ensure self.label_mapping_ exists
        if self.label_mapping_ is None or "to_string" not in self.label_mapping_:
            raise ValueError("Label mapping is not defined. Ensure _convert_target_to_float was called first.")

        # Use the mapping dictionary to convert numeric values back to labels
        y_converted = pd.Series(y).map(self.label_mapping_["to_string"])

        return y_converted

    def _extract_labels(self, y):
        """Extract unique labels from the target variable."""
        if hasattr(y, 'unique'):  # Check if y is a pandas Series
            labels = y.unique().tolist()
        else:  # Assume y is a numpy array
            import numpy as np
            labels = np.unique(y).tolist()
        
        if self.verbose:
            print(f"Extracted {len(labels)} unique labels from the target variable")
        
        return labels   

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

        # Apply scaling to all features after encoding
        cleaning_pipeline = SklearnPipeline([
            ('pre_encoding', preprocessor),
            ('scaler', StandardScaler() if self.scaling_method == 'standard' 
                    else MinMaxScaler() if self.scaling_method == 'minmax' 
                    else RobustScaler())
        ])
        
        return cleaning_pipeline
  
    def _extract_feature_names(self):
        """
        Extract feature names from the fitted preprocessor.
        
        Returns
        -------
        list
            List of feature names after preprocessing
        """
        if self.preprocessor_ is None:
            return []
        
        feature_names = []
        
        # Iterate through each transformer in the ColumnTransformer
        for transformer_name, transformer, feature_indices in self.preprocessor_.named_steps['pre_encoding'].transformers_:
            if transformer_name == 'remainder':
                continue
                
            # Get the feature names for this transformer
            if transformer_name == 'num':
                # Numerical features keep their original names
                if isinstance(feature_indices, list):
                    # Feature names were provided
                    if all(isinstance(idx, str) for idx in feature_indices):
                        num_feature_names = feature_indices
                    else:
                        # Integer indices, create generic names
                        num_feature_names = [f'num_feature_{idx}' for idx in feature_indices]
                else:
                    # Single feature
                    num_feature_names = [str(feature_indices)]
                
                feature_names.extend(num_feature_names)
                
            elif transformer_name in ['cat', 'cat_onehot']:
                # Categorical features with one-hot encoding
                try:
                    # Try to get feature names from the encoder
                    encoder = transformer.named_steps['encoder']
                    if hasattr(encoder, 'get_feature_names_out'):
                        # sklearn >= 1.0
                        input_features = feature_indices if isinstance(feature_indices, list) else [str(feature_indices)]
                        cat_feature_names = encoder.get_feature_names_out(input_features).tolist()
                    elif hasattr(encoder, 'get_feature_names'):
                        # sklearn < 1.0
                        input_features = feature_indices if isinstance(feature_indices, list) else [str(feature_indices)]
                        cat_feature_names = encoder.get_feature_names(input_features).tolist()
                    else:
                        # Fallback: estimate based on categories
                        if hasattr(encoder, 'categories_'):
                            cat_feature_names = []
                            for i, (feature, categories) in enumerate(zip(feature_indices, encoder.categories_)):
                                for category in categories:
                                    cat_feature_names.append(f'{feature}_{category}')
                        else:
                            # Ultimate fallback
                            cat_feature_names = [f'cat_feature_{i}' for i in range(len(feature_indices))]
                except:
                    # If all else fails, create generic names
                    cat_feature_names = [f'cat_feature_{i}' for i in range(len(feature_indices))]
                
                feature_names.extend(cat_feature_names)
                
            elif transformer_name == 'cat_ordinal':
                # Categorical features with ordinal encoding keep original names
                if isinstance(feature_indices, list):
                    # Feature names were provided
                    if all(isinstance(idx, str) for idx in feature_indices):
                        ordinal_feature_names = feature_indices
                    else:
                        # Integer indices, create generic names
                        ordinal_feature_names = [f'ordinal_feature_{idx}' for idx in feature_indices]
                else:
                    # Single feature
                    ordinal_feature_names = [str(feature_indices)]
                
                feature_names.extend(ordinal_feature_names)
        
        return feature_names
    