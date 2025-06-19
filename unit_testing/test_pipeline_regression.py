import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from autohpsearch.pipeline.pipeline import AutoMLPipeline

class TestAutoMLPipelineRegression(unittest.TestCase):
    def setUp(self):
        # Generate synthetic regression data
        rows, cols = 100, 10
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(rows, cols), columns=[f'feature_{i}' for i in range(cols)])
        y = np.random.rand(rows) * 100  # Continuous target variable

        # Add some missing values
        mask = np.random.random((rows, cols)) < 0.05  # 5% missing values
        X = X.mask(mask)

        # Convert some numerical columns to categorical
        X['mean radius'] = np.random.rand(rows) * 20
        X['mean radius_cat'] = pd.cut(X['mean radius'], 
                                      bins=[0, 10, 15, 20, 100], 
                                      labels=['tiny', 'small', 'medium', 'large'])

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        # Initialize the pipeline
        self.pipeline = AutoMLPipeline(
            task_type='regression',
            model_name=['random_forest_reg', 'gradient_boosting_reg'],        
        )

    def test_pipeline_fit(self):
        """Test pipeline fitting."""
        self.pipeline.fit(self.X_train, self.y_train, self.X_test, self.y_test)
        self.assertTrue(hasattr(self.pipeline, 'best_model_'), "Pipeline did not fit properly.")

    def test_missing_values_handling(self):
        """Test missing value handling."""
        self.pipeline.fit(self.X_train, self.y_train, self.X_test, self.y_test)
        self.assertFalse(np.isnan(self.pipeline.X_train_processed_).any().any(), "Missing values were not handled properly.")

    def test_categorical_encoding(self):
        """Test categorical encoding."""
        self.pipeline.fit(self.X_train, self.y_train, self.X_test, self.y_test)
        encoded_columns = [col for col in self.X_train.columns if '_cat' in col]
        self.assertTrue(len(encoded_columns) > 0, "Categorical encoding failed.")

    def test_processed_data_length(self):
        """Test if X_train_processed_ and y_train_processed_ have the same length."""
        self.pipeline.fit(self.X_train, self.y_train, self.X_test, self.y_test)
        self.assertEqual(
            len(self.pipeline.X_train_processed_),
            len(self.pipeline.y_train_processed_),
            "Mismatch in X_train_processed_ and y_train_processed_ length."
        )

    def test_generate_data_report(self):
        """Test the generate_data_report method."""
        self.pipeline.fit(self.X_train, self.y_train, self.X_test, self.y_test)
        report = self.pipeline.generate_data_report()
        self.assertIsNotNone(report, "Data report was not generated.")       

if __name__ == '__main__':
    unittest.main()