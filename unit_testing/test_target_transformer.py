import unittest
import numpy as np
import pandas as pd

from autohpsearch.pipeline.cleaning import TargetTransformer

class TestTargetTransformer(unittest.TestCase):
    """Unit tests for TargetTransformer class."""
    
    def setUp(self):
        """Set up test fixtures with various data types."""
        # Create different test data sets
        self.small_positive = np.array([1.5, 2.0, 3.0, 4.5, 5.0])
        self.large_positive = np.array([100.0, 200.0, 500.0, 1000.0])
        self.with_ones = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.with_zeros = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        self.small_values = np.array([0.1, 0.01, 0.001, 0.0001, 0.00001])
        self.with_neg_values = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
        
        # Pandas Series for testing
        self.pd_series = pd.Series([1.5, 2.0, 3.0, 4.5, 5.0])
        
        # Random data for Box-Cox
        np.random.seed(42)
        self.boxcox_data = np.random.lognormal(size=100)  # All positive
    
    def test_initialization(self):
        """Test initialization with different parameters."""
        # Default parameters
        transformer = TargetTransformer()
        self.assertEqual(transformer.transform_method, 'none')
        self.assertTrue(transformer.apply_inverse_transform)
        
        # Custom parameters
        transformer = TargetTransformer(transform_method='log', apply_inverse_transform=False)
        self.assertEqual(transformer.transform_method, 'log')
        self.assertFalse(transformer.apply_inverse_transform)
    
    def test_no_transformation(self):
        """Test that 'none' transformation doesn't change the data."""
        transformer = TargetTransformer(transform_method='none')
        transformer.fit(self.small_positive)
        transformed = transformer.transform(self.small_positive)
        np.testing.assert_array_equal(transformed, self.small_positive)
        
        # Test inverse transform
        inverse = transformer.inverse_transform(transformed)
        np.testing.assert_array_equal(inverse, self.small_positive)
    
    def test_log_transformation(self):
        """Test logarithmic transformation."""
        transformer = TargetTransformer(transform_method='log')
        transformer.fit(self.small_positive)
        
        # Check transformation
        transformed = transformer.transform(self.small_positive)
        expected = np.log(self.small_positive)
        np.testing.assert_array_almost_equal(transformed, expected)
        
        # Check inverse transformation
        inverse = transformer.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(inverse, self.small_positive)
    
    def test_log1p_transformation(self):
        """Test log1p transformation."""
        transformer = TargetTransformer(transform_method='log1p')
        transformer.fit(self.small_positive)
        
        # Check transformation
        transformed = transformer.transform(self.small_positive)
        expected = np.log1p(self.small_positive)
        np.testing.assert_array_almost_equal(transformed, expected)
        
        # Check inverse transformation
        inverse = transformer.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(inverse, self.small_positive)
        
        # Test with zeros (log1p can handle zeros)
        transformer.fit(self.with_zeros)
        transformed = transformer.transform(self.with_zeros)
        expected = np.log1p(self.with_zeros)
        np.testing.assert_array_almost_equal(transformed, expected)
        
        inverse = transformer.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(inverse, self.with_zeros)
    
    def test_sqrt_transformation(self):
        """Test square root transformation."""
        transformer = TargetTransformer(transform_method='sqrt')
        transformer.fit(self.small_positive)
        
        # Check transformation
        transformed = transformer.transform(self.small_positive)
        expected = np.sqrt(self.small_positive)
        np.testing.assert_array_almost_equal(transformed, expected)
        
        # Check inverse transformation
        inverse = transformer.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(inverse, self.small_positive)
        
        # Test with zeros (sqrt can handle zeros)
        transformer.fit(self.with_zeros)
        transformed = transformer.transform(self.with_zeros)
        expected = np.sqrt(self.with_zeros)
        np.testing.assert_array_almost_equal(transformed, expected)   
    
    def test_pandas_series(self):
        """Test handling of pandas Series."""
        transformer = TargetTransformer(transform_method='log')
        transformer.fit(self.pd_series)
        
        transformed = transformer.transform(self.pd_series)
        # Check it returns the expected values
        np.testing.assert_array_almost_equal(transformed, np.log(self.pd_series.values))
        
        # Check the type is preserved as numpy array
        self.assertIsInstance(transformed, np.ndarray)
    
    def test_invalid_transform_method(self):
        """Test error handling for invalid transformation methods."""
        transformer = TargetTransformer(transform_method='invalid_method')
        
        with self.assertRaises(ValueError):
            transformer.fit(self.small_positive)
            transformer.transform(self.small_positive)
    
    def test_log_with_zeros(self):
        """Test logarithm with zeros raises appropriate error."""
        transformer = TargetTransformer(transform_method='log')
        transformer.fit(self.with_ones)  # Fit with safe data
        
        # Should raise warning or error when transforming zeros
        with self.assertRaises(Exception):  # Could be RuntimeWarning or other errors
            transformer.transform(self.with_zeros)  
    
    def test_inverse_transform_disabled(self):
        """Test behavior when inverse_transform is disabled."""
        transformer = TargetTransformer(transform_method='log', apply_inverse_transform=False)
        transformer.fit(self.small_positive)
        
        transformed = transformer.transform(self.small_positive)
        inverse = transformer.inverse_transform(transformed)
        
        # Should return the transformed data unchanged
        np.testing.assert_array_almost_equal(inverse, transformed)
    
    def test_small_values(self):
        """Test behavior with very small values."""
        transformer = TargetTransformer(transform_method='log')
        transformer.fit(self.small_values)
        
        transformed = transformer.transform(self.small_values)
        inverse = transformer.inverse_transform(transformed)
        
        # Check round-trip accuracy
        np.testing.assert_array_almost_equal(inverse, self.small_values)
    
    def test_large_values(self):
        """Test behavior with large values."""
        transformer = TargetTransformer(transform_method='log')
        transformer.fit(self.large_positive)
        
        transformed = transformer.transform(self.large_positive)
        inverse = transformer.inverse_transform(transformed)
        
        # Check round-trip accuracy
        np.testing.assert_array_almost_equal(inverse, self.large_positive)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)  