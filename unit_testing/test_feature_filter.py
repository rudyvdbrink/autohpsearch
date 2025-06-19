import unittest
import numpy as np
import pandas as pd
from autohpsearch.pipeline.cleaning import FeatureFilter  # Replace with the actual file name if saved

class TestFeatureFilter(unittest.TestCase):
    def setUp(self):
        # Create synthetic data for testing
        self.data_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1, 2, 3, 4, 5],  # Perfectly correlated with A
            'C': [5, 4, 3, 2, 1],
            'D': [1, 1, 1, 1, 1]   # No variance
        })

        self.data_np = np.array([
            [1, 1, 5, 1],
            [2, 2, 4, 1],
            [3, 3, 3, 1],
            [4, 4, 2, 1],
            [5, 5, 1, 1]
        ])

        # Initialize the FeatureFilter
        self.filterer = FeatureFilter(thresh=0.95, method='spearman')

    def test_fit_dataframe(self):
        """Test fit method with pandas DataFrame."""
        columns_to_drop = self.filterer.fit(self.data_df)
        self.assertEqual(columns_to_drop, ['B', 'C'], "Incorrect columns identified for dropping in DataFrame.")

    def test_filter_dataframe(self):
        """Test filter method with pandas DataFrame."""
        self.filterer.fit(self.data_df)
        filtered_df = self.filterer.filter(self.data_df)
        self.assertEqual(filtered_df.shape[1], 2, "Incorrect number of columns after filtering in DataFrame.")
        self.assertNotIn('B', filtered_df.columns, "Column B was not dropped from DataFrame.")

    def test_fit_numpy_array(self):
        """Test fit method with numpy array."""
        columns_to_drop = self.filterer.fit(self.data_np)
        self.assertEqual(columns_to_drop, ['feature_1', 'feature_2'], "Incorrect columns identified for dropping in numpy array.")

    def test_filter_numpy_array(self):
        """Test filter method with numpy array."""
        self.filterer.fit(self.data_np)
        filtered_np = self.filterer.filter(self.data_np)
        self.assertEqual(filtered_np.shape[1], 2, "Incorrect number of columns after filtering in numpy array.")

if __name__ == '__main__':
    unittest.main()