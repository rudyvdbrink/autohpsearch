import unittest
import numpy as np
import pandas as pd

from autohpsearch.pipeline.pipeline import OutlierRemover

class TestOutlierRemover(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data with known outliers
        np.random.seed(42)
        # Normal data points
        self.normal_data = np.random.normal(0, 1, (100, 3))
        # Add some outliers
        self.outliers = np.array([
            [10, 10, 10],
            [-10, -10, -10],
            [8, -8, 8]
        ])
        self.data_with_outliers = np.vstack([self.normal_data, self.outliers])
        
        # Create DataFrame version of the data
        self.df_with_outliers = pd.DataFrame(
            self.data_with_outliers, 
            columns=['feature1', 'feature2', 'feature3']
        )
        
        # Data with NaN values
        self.data_with_nans = self.data_with_outliers.copy()
        self.data_with_nans[0, 0] = np.nan
        self.data_with_nans[1, 1] = np.nan
        self.df_with_nans = pd.DataFrame(
            self.data_with_nans,
            columns=['feature1', 'feature2', 'feature3']
        )
        
    def test_initialization(self):
        """Test initialization with different parameters."""
        # Default parameters
        remover = OutlierRemover()
        self.assertEqual(remover.method, 'zscore')
        self.assertEqual(remover.threshold, 3.0)
        
        # Custom parameters
        remover = OutlierRemover(method='iqr', threshold=1.5)
        self.assertEqual(remover.method, 'iqr')
        self.assertEqual(remover.threshold, 1.5)
        
    def test_zscore_outlier_detection_numpy(self):
        """Test z-score outlier detection with numpy arrays."""
        remover = OutlierRemover(method='zscore', threshold=3.0)
        remover.fit(self.data_with_outliers)
        
        # Should identify 3 outliers
        self.assertEqual(np.sum(~remover.mask_), 3)
        
        # Transform should remove outliers
        transformed = remover.transform(self.data_with_outliers)
        self.assertEqual(transformed.shape[0], 100)  # 103 - 3 outliers
        
    def test_zscore_outlier_detection_pandas(self):
        """Test z-score outlier detection with pandas DataFrame."""
        remover = OutlierRemover(method='zscore', threshold=3.0)
        remover.fit(self.df_with_outliers)
        
        # Should identify 3 outliers
        self.assertEqual(np.sum(~remover.mask_), 3)
        
        # Transform should remove outliers
        transformed = remover.transform(self.df_with_outliers)
        self.assertEqual(transformed.shape[0], 100)  # 103 - 3 outliers
        self.assertTrue(isinstance(transformed, pd.DataFrame))
        
    def test_iqr_outlier_detection_numpy(self):
        """Test IQR outlier detection with numpy arrays."""
        remover = OutlierRemover(method='iqr', threshold=1.5)
        remover.fit(self.data_with_outliers)
        
        # Should identify the outliers
        self.assertTrue(np.sum(~remover.mask_) > 0)
        
        # Transform should remove outliers
        transformed = remover.transform(self.data_with_outliers)
        self.assertTrue(transformed.shape[0] < self.data_with_outliers.shape[0])
        
    def test_iqr_outlier_detection_pandas(self):
        """Test IQR outlier detection with pandas DataFrame."""
        remover = OutlierRemover(method='iqr', threshold=1.5)
        remover.fit(self.df_with_outliers)
        
        # Should identify the outliers
        self.assertTrue(np.sum(~remover.mask_) > 0)
        
        # Transform should remove outliers
        transformed = remover.transform(self.df_with_outliers)
        self.assertTrue(transformed.shape[0] < self.df_with_outliers.shape[0])
        self.assertTrue(isinstance(transformed, pd.DataFrame))
        
    def test_fit_transform(self):
        """Test the fit_transform method."""
        remover = OutlierRemover()
        transformed = remover.fit_transform(self.data_with_outliers)
        
        # Should have removed the outliers
        self.assertTrue(transformed.shape[0] < self.data_with_outliers.shape[0])
        
    def test_nan_handling_zscore(self):
        """Test handling of NaN values with z-score method."""
        remover = OutlierRemover(method='zscore')
        remover.fit(self.data_with_nans)
        
        # Z-score method should handle NaNs
        transformed = remover.transform(self.data_with_nans)
        self.assertTrue(transformed.shape[0] < self.data_with_nans.shape[0])
        
    def test_nan_handling_iqr(self):
        """Test handling of NaN values with IQR method."""
        remover = OutlierRemover(method='iqr')
        remover.fit(self.data_with_nans)
        
        # IQR method should handle NaNs
        transformed = remover.transform(self.data_with_nans)
        self.assertTrue(transformed.shape[0] < self.data_with_nans.shape[0])
        
    def test_nan_handling_pandas(self):
        """Test handling of NaN values with pandas DataFrame."""
        remover = OutlierRemover()
        remover.fit(self.df_with_nans)
        
        # Should handle NaNs in DataFrames
        transformed = remover.transform(self.df_with_nans)
        self.assertTrue(transformed.shape[0] < self.df_with_nans.shape[0])
        
    def test_invalid_method(self):
        """Test handling of invalid outlier detection method."""
        remover = OutlierRemover(method='invalid_method')
        with self.assertRaises(ValueError):
            remover.fit(self.data_with_outliers)
            
    def test_empty_dataset(self):
        """Test handling of empty datasets."""
        remover = OutlierRemover()
        empty_data = np.array([]).reshape(0, 3)
        remover.fit(empty_data)
        transformed = remover.transform(empty_data)
        self.assertEqual(transformed.shape, (0, 3))
        
    def test_no_numerical_columns(self):
        """Test handling of datasets with no numerical columns."""
        remover = OutlierRemover()
        # Create a DataFrame with only string columns
        df = pd.DataFrame({
            'col1': ['a', 'b', 'c'],
            'col2': ['x', 'y', 'z']
        })
        remover.fit(df)
        transformed = remover.transform(df)
        # Should keep all rows as there are no numerical columns to find outliers in
        self.assertEqual(transformed.shape[0], df.shape[0])
        
    def test_get_mask(self):
        """Test the get_mask method."""
        remover = OutlierRemover()
        remover.fit(self.data_with_outliers)
        mask = remover.get_mask()
        
        # Mask should be a boolean array
        self.assertTrue(isinstance(mask, np.ndarray))
        self.assertEqual(mask.dtype, np.bool_)
        
        # Length should match number of samples
        self.assertEqual(len(mask), len(self.data_with_outliers))
        
    def test_mixed_type_dataframe(self):
        """Test handling of DataFrames with mixed column types."""
        mixed_df = self.df_with_outliers.copy()
        mixed_df['category'] = ['A'] * 50 + ['B'] * 50 + ['C'] * 3
        
        remover = OutlierRemover()
        remover.fit(mixed_df)
        
        # Should only consider numerical columns for outlier detection
        transformed = remover.transform(mixed_df)
        self.assertTrue(transformed.shape[0] < mixed_df.shape[0])
        self.assertTrue('category' in transformed.columns)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)  